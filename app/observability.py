"""Structured logging, and the counters behind the admin endpoint.

Logs are JSON, one object per line, because the questions worth asking of them are
aggregate ones - "what is the escalation rate for REFUND questions this week", "which
turns took over two seconds" - and grepping prose for that is a bad time. Every log
line for a turn carries the request id and the conversation id, so a customer
complaint can be traced from a single timestamp to the exact chunks the bot retrieved.

On the metric that is *not* here: **false-answer rate cannot be computed from live
traffic.** A false answer is one the bot gave confidently and wrongly, and nothing in
a request tells us it was wrong - that needs ground truth or a human saying so. The
PRD asks for it, and the honest place to produce it is the Phase 11 eval, which has
labelled queries. Reporting a made-up number here, or silently redefining it as
something cheaper to measure, would be worse than reporting nothing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass

from redis.asyncio import Redis

# Uvicorn installs its own handlers with its own plain-text format, so without this
# half the log stream is JSON and half is prose - and an aggregator can parse
# neither reliably. Clearing their handlers and letting them propagate sends them
# through the root JSON handler like everything else.
UVICORN_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")

# Set per request by the middleware, read by the formatter, so a log line emitted
# deep in the pipeline still knows which request it belongs to without every
# function having to pass an id down.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        # Anything passed as extra={...} rides along as a first-class field rather
        # than being interpolated into the message string, which is what makes the
        # line queryable.
        payload.update(getattr(record, "fields", {}))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class Timer:
    """Wall-clock milliseconds, because that is what a customer waits."""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@dataclass(frozen=True)
class TurnMetrics:
    escalated: bool
    reason: str | None
    category: str | None  # of the best chunk, even when the turn escalated
    confidence: float
    latency_ms: float


class MetricsStore:
    """Counters in Redis. Aggregation only - no per-turn rows, no PII.

    Deliberately not a time series. The PRD asks for rates, and rates come out of
    counters; a proper time series belongs in Prometheus rather than in a hash we
    hand-rolled. What is here is enough to answer "what fraction of turns did we
    deflect, and where are we escalating most".
    """

    TOTALS = "metrics:totals"
    REASONS = "metrics:reasons"
    CATEGORY_ANSWERED = "metrics:category:answered"
    CATEGORY_ESCALATED = "metrics:category:escalated"
    LATENCY = "metrics:latency"

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def record(self, metrics: TurnMetrics) -> None:
        pipe = self._redis.pipeline()
        pipe.hincrby(self.TOTALS, "turns", 1)

        if metrics.escalated:
            pipe.hincrby(self.TOTALS, "escalated", 1)
            pipe.hincrby(self.REASONS, metrics.reason or "unknown", 1)
            if metrics.category:
                pipe.hincrby(self.CATEGORY_ESCALATED, metrics.category, 1)
        else:
            pipe.hincrby(self.TOTALS, "answered", 1)
            if metrics.category:
                pipe.hincrby(self.CATEGORY_ANSWERED, metrics.category, 1)

        # Sum and count, so the mean is a division rather than a stored average
        # that drifts. Not a histogram: p95 would need one, and a hash cannot fake it.
        pipe.hincrbyfloat(self.LATENCY, "sum_ms", metrics.latency_ms)
        pipe.hincrby(self.LATENCY, "count", 1)

        await pipe.execute()

    async def snapshot(self) -> dict:
        pipe = self._redis.pipeline()
        pipe.hgetall(self.TOTALS)
        pipe.hgetall(self.REASONS)
        pipe.hgetall(self.CATEGORY_ANSWERED)
        pipe.hgetall(self.CATEGORY_ESCALATED)
        pipe.hgetall(self.LATENCY)
        totals, reasons, answered_by_cat, escalated_by_cat, latency = await pipe.execute()

        turns = int(totals.get("turns", 0))
        answered = int(totals.get("answered", 0))
        escalated = int(totals.get("escalated", 0))

        categories = {}
        for category in set(answered_by_cat) | set(escalated_by_cat):
            a = int(answered_by_cat.get(category, 0))
            e = int(escalated_by_cat.get(category, 0))
            categories[category] = {
                "turns": a + e,
                "answered": a,
                "escalated": e,
                "deflection_rate": _rate(a, a + e),
                # See the module docstring. This needs labels, and live traffic has
                # none. Phase 11's eval is where it comes from.
                "false_answer_rate": None,
            }

        return {
            "turns": turns,
            "answered": answered,
            "escalated": escalated,
            "deflection_rate": _rate(answered, turns),
            "escalation_rate": _rate(escalated, turns),
            "escalation_reasons": {k: int(v) for k, v in reasons.items()},
            "mean_latency_ms": _mean(latency),
            "by_category": categories,
            "false_answer_rate": None,
            "false_answer_rate_note": (
                "Not measurable from live traffic: it needs ground truth or user "
                "feedback to know an answer was wrong. Produced by the Phase 11 eval "
                "harness against labelled Bitext queries."
            ),
        }


def _rate(part: int, whole: int) -> float | None:
    """None, not 0.0, when nothing has happened yet.

    A 0% deflection rate and "no traffic" are very different things, and a dashboard
    that renders them identically will have someone debugging a bot that is fine.
    """
    if whole == 0:
        return None
    return round(part / whole, 4)


def _mean(latency: dict) -> float | None:
    count = int(latency.get("count", 0))
    if count == 0:
        return None
    return round(float(latency.get("sum_ms", 0.0)) / count, 1)

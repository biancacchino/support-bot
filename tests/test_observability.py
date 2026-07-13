"""Logging and metrics.

The load-bearing assertions here are about honesty, not arithmetic:

- an empty bot reports `null` rates, not 0%, because "nothing has happened" and
  "everything failed" must not render identically on a dashboard
- false-answer rate is `null` with a stated reason, because it is not computable
  from live traffic and a plausible-looking made-up number is worse than a gap
"""

import json
import logging
import time

import pytest
from fakeredis.aioredis import FakeRedis

from app.observability import (
    UVICORN_LOGGERS,
    JsonFormatter,
    MetricsStore,
    Timer,
    TurnMetrics,
    configure_logging,
    request_id_var,
)


@pytest.fixture
async def metrics():
    redis = FakeRedis(decode_responses=True)
    yield MetricsStore(redis)
    await redis.aclose()


def turn(
    *,
    escalated: bool = False,
    reason: str | None = None,
    category: str | None = "REFUND",
    confidence: float = 0.9,
    latency_ms: float = 100.0,
) -> TurnMetrics:
    return TurnMetrics(
        escalated=escalated,
        reason=reason,
        category=category,
        confidence=confidence,
        latency_ms=latency_ms,
    )


# --- structured logs --------------------------------------------------------


def test_a_log_line_is_json_with_its_request_id(caplog):
    record = logging.LogRecord("t", logging.INFO, "f", 1, "turn answered", None, None)
    record.fields = {"conversation_id": "c1", "escalated": False, "confidence": 0.91}
    token = request_id_var.set("req-abc")

    try:
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert line["message"] == "turn answered"
    assert line["request_id"] == "req-abc"
    assert line["conversation_id"] == "c1"  # a field, not buried in the message text
    assert line["escalated"] is False
    assert line["confidence"] == 0.91


def test_timing_is_measured_not_guessed():
    """`>= 0` was the old assertion, and a Timer that returned a constant 0.0 passed it.

    Sleeping for a known interval is the cheapest thing that actually fails if the
    clock is never read: the elapsed time has to be at least the sleep, and has to
    be in milliseconds rather than seconds.
    """
    with Timer() as timer:
        time.sleep(0.02)

    assert timer.elapsed_ms >= 20  # not 0.02: this is milliseconds, not seconds
    assert timer.elapsed_ms < 5_000  # and it is an interval, not a wall-clock timestamp


def test_uvicorn_logs_go_through_the_json_formatter_too():
    """Otherwise half the stream is JSON and half is prose, and neither parses.

    Uvicorn installs its own plain-text handlers. Left alone, its access lines
    ("INFO: ... POST /chat 200 OK") sit alongside our JSON, and an aggregator can
    make sense of neither.
    """
    configure_logging("INFO")

    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        assert uvicorn_logger.handlers == []
        assert uvicorn_logger.propagate is True

    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


# --- counters ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_bot_reports_nothing_rather_than_zero(metrics):
    """No traffic and total failure must not look the same.

    A 0% deflection rate is an emergency. No traffic is a Tuesday. A dashboard that
    renders them identically will have someone debugging a bot that is fine.
    """
    snapshot = await metrics.snapshot()

    assert snapshot["turns"] == 0
    assert snapshot["deflection_rate"] is None
    assert snapshot["escalation_rate"] is None
    assert snapshot["mean_latency_ms"] is None


@pytest.mark.asyncio
async def test_deflection_and_escalation_rates(metrics):
    await metrics.record(turn(escalated=False))
    await metrics.record(turn(escalated=False))
    await metrics.record(turn(escalated=False))
    await metrics.record(turn(escalated=True, reason="low_confidence"))

    snapshot = await metrics.snapshot()

    assert snapshot["turns"] == 4
    assert snapshot["deflection_rate"] == 0.75
    assert snapshot["escalation_rate"] == 0.25


@pytest.mark.asyncio
async def test_escalations_are_broken_down_by_reason(metrics):
    await metrics.record(turn(escalated=True, reason="low_confidence"))
    await metrics.record(turn(escalated=True, reason="low_confidence"))
    await metrics.record(turn(escalated=True, reason="human_owned"))

    assert (await metrics.snapshot())["escalation_reasons"] == {
        "low_confidence": 2,
        "human_owned": 1,
    }


@pytest.mark.asyncio
async def test_rates_are_broken_down_by_category(metrics):
    """"We keep escalating REFUND questions" is the finding this endpoint exists for."""
    await metrics.record(turn(category="ORDER", escalated=False))
    await metrics.record(turn(category="REFUND", escalated=False))
    await metrics.record(turn(category="REFUND", escalated=True, reason="low_confidence"))
    await metrics.record(turn(category="REFUND", escalated=True, reason="low_confidence"))

    by_category = (await metrics.snapshot())["by_category"]

    assert by_category["ORDER"] == {
        "turns": 1,
        "answered": 1,
        "escalated": 0,
        "deflection_rate": 1.0,
        "false_answer_rate": None,
    }
    assert by_category["REFUND"]["turns"] == 3
    assert by_category["REFUND"]["deflection_rate"] == pytest.approx(0.3333, abs=1e-4)


@pytest.mark.asyncio
async def test_an_escalated_turn_still_records_its_category(metrics):
    """Otherwise the one thing you want to know - where we fail - is unavailable."""
    await metrics.record(turn(category="REFUND", escalated=True, reason="low_confidence"))

    assert (await metrics.snapshot())["by_category"]["REFUND"]["escalated"] == 1


@pytest.mark.asyncio
async def test_mean_latency_is_a_division_not_a_stored_average(metrics):
    await metrics.record(turn(latency_ms=100.0))
    await metrics.record(turn(latency_ms=300.0))

    assert (await metrics.snapshot())["mean_latency_ms"] == 200.0


@pytest.mark.asyncio
async def test_false_answer_rate_is_null_and_says_why(metrics):
    """The PRD asks for it; live traffic cannot produce it.

    Knowing an answer was *wrong* needs ground truth or a human saying so. Reporting
    a number here - or silently redefining it as something cheaper to measure, like
    "escalation rate" - would be worse than reporting nothing, because someone would
    put it in a slide.
    """
    await metrics.record(turn(escalated=False))

    snapshot = await metrics.snapshot()

    assert snapshot["false_answer_rate"] is None
    assert "ground truth" in snapshot["false_answer_rate_note"]
    assert "Phase 11" in snapshot["false_answer_rate_note"]

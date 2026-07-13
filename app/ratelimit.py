"""Redis-backed rate limiting, on two axes that protect different things.

A per-client limit stops one caller hammering the bot. It does *not* protect the
Gemini quota, and confusing the two is the mistake worth avoiding here: the free
tier is a project-wide budget, so a hundred callers each politely under their own
limit will still exhaust it between them, and every one of them then gets a 500
from an upstream 429 nobody was watching for.

So there are two limiters:

- per client (IP or API key), which is about fairness and abuse
- a global upstream budget, which is about not spending a quota we do not have

The upstream budget is sized in *turns*, not requests, because one turn can cost
two Gemini calls: condensing the follow-up, and writing the answer. Sizing a
15-RPM budget as 15 turns would silently overspend it by a factor of two on any
multi-turn conversation.

Fixed windows, not a token bucket. A fixed window is two Redis commands and is
trivially correct under concurrency; a token bucket would be smoother at the
boundary and is not worth the Lua script here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until the window resets; 0 when allowed


class RateLimiter:
    """A fixed-window counter per key."""

    def __init__(self, redis: Redis, limit: int, window_seconds: int, namespace: str) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_seconds
        self._namespace = namespace

    def _key(self, identity: str) -> str:
        return f"ratelimit:{self._namespace}:{identity}"

    async def check(self, identity: str) -> Decision:
        """Count this request against `identity`, and say whether it may proceed."""
        key = self._key(identity)

        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = await pipe.execute()

        # First request in a window, or a key that somehow lost its expiry: give it
        # one. Without this a counter could live forever and lock the caller out
        # permanently, which is a far worse failure than letting one extra request
        # through.
        if count == 1 or ttl < 0:
            await self._redis.expire(key, self._window)
            ttl = self._window

        if count > self._limit:
            logger.info("rate limited %s (%d > %d in %ds)", key, count, self._limit, self._window)
            return Decision(allowed=False, limit=self._limit, remaining=0, retry_after=max(ttl, 1))

        return Decision(
            allowed=True,
            limit=self._limit,
            remaining=self._limit - count,
            retry_after=0,
        )


class TurnLimiter:
    """The limits a single chat turn has to clear, checked cheapest-first."""

    def __init__(
        self,
        per_minute: RateLimiter,
        per_day: RateLimiter,
        upstream: RateLimiter,
        upstream_day: RateLimiter,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._upstream = upstream
        self._upstream_day = upstream_day

    async def check(self, identity: str) -> Decision:
        """Allow the turn, or return the first limit that says no.

        Client limits are checked before the shared upstream budget on purpose. A
        caller who is over their own limit should not get to consume a slot of a
        budget everyone else is sharing, even to be told no.

        The upstream budget is guarded per minute *and* per day, because Gemini's
        free tier is capped on both and they fail differently. The per-minute limiter
        alone cannot hold the daily line: 15 RPM sustained is far more than 1,000 RPD
        allows, so a day of steady, individually-legal traffic exhausts the quota and
        every turn after it fails upstream. This was the gap - `gemini_rpd` was in the
        config, documented, and read by nothing.
        """
        for limiter, key in (
            (self._per_minute, identity),
            (self._per_day, identity),
            (self._upstream, "global"),
            (self._upstream_day, "global"),
        ):
            decision = await limiter.check(key)
            if not decision.allowed:
                return decision

        return decision

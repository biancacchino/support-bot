"""Rate limiter tests, against a real Redis implementation (fakeredis).

The interesting cases are not "does it count to ten". They are the ones where a
limiter fails open or fails closed by accident: a key that lost its TTL and locks
a caller out forever, or a client limit that quietly consumes the shared upstream
budget it was supposed to protect.
"""

import pytest
from fakeredis.aioredis import FakeRedis

from app.ratelimit import RateLimiter, TurnLimiter


@pytest.fixture
async def redis():
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def limiter(redis, limit: int, window: int = 60, namespace: str = "test") -> RateLimiter:
    return RateLimiter(redis, limit=limit, window_seconds=window, namespace=namespace)


# --- counting ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_requests_under_the_limit_are_allowed(redis):
    rl = limiter(redis, limit=3)

    decisions = [await rl.check("alice") for _ in range(3)]

    assert all(d.allowed for d in decisions)
    assert [d.remaining for d in decisions] == [2, 1, 0]


@pytest.mark.asyncio
async def test_the_request_over_the_limit_is_refused(redis):
    rl = limiter(redis, limit=2)
    await rl.check("alice")
    await rl.check("alice")

    decision = await rl.check("alice")

    assert decision.allowed is False
    assert decision.remaining == 0
    assert decision.retry_after > 0  # and it says when to come back


@pytest.mark.asyncio
async def test_callers_are_limited_separately(redis):
    rl = limiter(redis, limit=1)
    await rl.check("alice")

    assert (await rl.check("bob")).allowed is True


@pytest.mark.asyncio
async def test_namespaces_do_not_share_a_counter(redis):
    """The per-minute and per-day limiters count the same caller independently."""
    await limiter(redis, limit=1, namespace="minute").check("alice")

    assert (await limiter(redis, limit=1, namespace="day").check("alice")).allowed is True


@pytest.mark.asyncio
async def test_the_window_expires(redis):
    rl = limiter(redis, limit=1, window=60)
    await rl.check("alice")

    assert await redis.ttl("ratelimit:test:alice") > 0


@pytest.mark.asyncio
async def test_a_counter_that_lost_its_expiry_does_not_lock_the_caller_out_forever(redis):
    """Fail open, not closed.

    A key with no TTL - a crash between INCR and EXPIRE, a manual poke - would
    otherwise count up forever and ban the caller permanently. Letting one extra
    request through is a far better failure than a permanent, silent ban.
    """
    rl = limiter(redis, limit=5)
    await redis.set("ratelimit:test:alice", 99)  # no TTL, count already over the limit

    decision = await rl.check("alice")

    assert await redis.ttl("ratelimit:test:alice") > 0  # expiry restored
    assert decision.allowed is False  # this request still counts as over
    assert decision.retry_after > 0  # but it will drain, rather than never resetting


# --- the three limits a turn has to clear -----------------------------------


def turn_limiter(
    redis, *, per_minute: int, per_day: int, upstream: int, upstream_day: int = 1_000
) -> TurnLimiter:
    return TurnLimiter(
        per_minute=RateLimiter(redis, per_minute, 60, "client-minute"),
        per_day=RateLimiter(redis, per_day, 86_400, "client-day"),
        upstream=RateLimiter(redis, upstream, 60, "upstream-minute"),
        upstream_day=RateLimiter(redis, upstream_day, 86_400, "upstream-day"),
    )


@pytest.mark.asyncio
async def test_a_turn_within_every_limit_is_allowed(redis):
    tl = turn_limiter(redis, per_minute=10, per_day=100, upstream=7)

    assert (await tl.check("ip:1.2.3.4")).allowed is True


@pytest.mark.asyncio
async def test_the_daily_limit_still_bites_when_the_minute_one_does_not(redis):
    tl = turn_limiter(redis, per_minute=100, per_day=2, upstream=100)
    await tl.check("ip:1.2.3.4")
    await tl.check("ip:1.2.3.4")

    assert (await tl.check("ip:1.2.3.4")).allowed is False


@pytest.mark.asyncio
async def test_the_shared_upstream_budget_stops_polite_callers_from_exhausting_gemini(redis):
    """The reason a per-client limit is not enough.

    Three different callers, each comfortably inside their own limit, still burn
    the same project-wide Gemini quota between them. Without the global budget the
    fourth turn reaches Gemini, gets a 429 from Google, and becomes a 500 for
    someone who did nothing wrong.
    """
    tl = turn_limiter(redis, per_minute=10, per_day=100, upstream=3)

    assert (await tl.check("ip:1.1.1.1")).allowed is True
    assert (await tl.check("ip:2.2.2.2")).allowed is True
    assert (await tl.check("ip:3.3.3.3")).allowed is True

    fourth = await tl.check("ip:4.4.4.4")

    assert fourth.allowed is False
    assert fourth.retry_after > 0


@pytest.mark.asyncio
async def test_a_client_over_their_own_limit_does_not_spend_the_shared_budget(redis):
    """Checked cheapest-first, and for a reason.

    Someone hammering us past their own limit should not get to consume a slot of
    the budget everyone else is sharing, even just to be told no.
    """
    tl = turn_limiter(redis, per_minute=1, per_day=100, upstream=5)
    await tl.check("ip:1.1.1.1")  # uses their one, and one upstream slot

    await tl.check("ip:1.1.1.1")  # refused on their own limit
    await tl.check("ip:1.1.1.1")

    # The upstream budget has been charged once, not three times.
    assert await redis.get("ratelimit:upstream-minute:global") == "1"


@pytest.mark.asyncio
async def test_the_daily_gemini_budget_holds_when_the_per_minute_one_never_fires(redis):
    """The gap Phase 12 found: `gemini_rpd` was configured and read by nothing.

    A per-minute upstream budget cannot enforce a daily one. Traffic that never
    trips 7 turns/minute can still walk past 1,000 requests/day quite comfortably,
    and the failure mode is the worst kind: every turn suddenly 500s late in the day
    because Google is refusing us, and nothing in our own logs says why.
    """
    tl = turn_limiter(redis, per_minute=10, per_day=100, upstream=10, upstream_day=2)

    assert (await tl.check("ip:1.1.1.1")).allowed is True
    assert (await tl.check("ip:2.2.2.2")).allowed is True

    # Nobody is over their own limit, and nobody is over the per-minute budget.
    exhausted = await tl.check("ip:3.3.3.3")

    assert exhausted.allowed is False
    assert exhausted.retry_after > 0

"""API tests: the two response shapes, and the two ways a turn can be refused.

The bot is stubbed here. What is under test is the HTTP contract - status codes,
the shape a client has to branch on, and whether a limit produces a usable answer
rather than a stack trace - not the pipeline, which has its own suite.
"""

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import get_bot, get_limiter, get_metrics, router
from app.bot import Answered, Escalated, EscalationReason
from app.config import get_settings
from app.conversation import Turn
from app.llm import UpstreamRateLimited
from app.observability import MetricsStore
from app.ratelimit import RateLimiter, TurnLimiter
from app.reranker import Ranked
from app.retrieval import Candidate

ANSWERED = Answered(
    text="Refunds take five working days.",
    citations=("refund-policy",),
    confidence=0.91,
    condensed_query="how long does a refund take",
    category="REFUND",
)

ESCALATED = Escalated(
    reason=EscalationReason.LOW_CONFIDENCE,
    query="what is your ceo paid",
    condensed_query="what is your ceo paid",
    confidence=0.02,
    history=[Turn(question="hello", answer="Hi!", escalated=False)],
    chunks=[
        Ranked(
            candidate=Candidate(
                doc_id="placing-an-order",
                title="Placing an order",
                category="ORDER",
                intents=("place_order",),
                heading="How to order",
                chunk_index=0,
                text="Add items to your basket.",
                score=0.44,
            ),
            score=0.02,
        )
    ],
)


class StubBot:
    def __init__(self, result=ANSWERED, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.seen: list[tuple[str, str]] = []

    async def handle(self, query: str, conversation_id: str):
        self.seen.append((query, conversation_id))
        if self._raises:
            raise self._raises
        return self._result


def build_app(bot, limiter, metrics: MetricsStore | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_bot] = lambda: bot
    app.dependency_overrides[get_limiter] = lambda: limiter
    app.dependency_overrides[get_metrics] = lambda: metrics or MetricsStore(FakeRedis(decode_responses=True))
    return app


def generous(redis) -> TurnLimiter:
    return TurnLimiter(
        per_minute=RateLimiter(redis, 100, 60, "client-minute"),
        per_day=RateLimiter(redis, 1000, 86_400, "client-day"),
        upstream=RateLimiter(redis, 100, 60, "upstream-minute"),
    )


@pytest.fixture
async def redis():
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


async def call(app, **body):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/chat", json=body)


# --- the two shapes ---------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answered_turn_comes_back_answered_and_cited(redis):
    response = await call(build_app(StubBot(ANSWERED), generous(redis)), message="how long for a refund")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Refunds take five working days."
    assert body["citations"] == ["refund-policy"]
    assert body["conversation_id"]  # minted for us


@pytest.mark.asyncio
async def test_an_escalated_turn_is_a_different_shape_entirely(redis):
    """A client cannot mistake one for the other: there is no `answer` to render."""
    response = await call(build_app(StubBot(ESCALATED), generous(redis)), message="what is your ceo paid")

    body = response.json()
    assert body["status"] == "escalated"
    assert body["reason"] == "low_confidence"
    assert "answer" not in body
    assert "citations" not in body


@pytest.mark.asyncio
async def test_an_escalation_hands_over_the_evidence(redis):
    """The chunks that lost, with their scores, and the conversation so far."""
    response = await call(build_app(StubBot(ESCALATED), generous(redis)), message="q")

    body = response.json()
    assert body["retrieved"] == [{"doc_id": "placing-an-order", "title": "Placing an order", "score": 0.02}]
    assert [t["question"] for t in body["history"]] == ["hello"]


@pytest.mark.asyncio
async def test_a_conversation_id_is_passed_through_so_a_turn_can_be_a_follow_up(redis):
    bot = StubBot(ANSWERED)

    await call(build_app(bot, generous(redis)), message="how long will it take", conversation_id="c1")

    assert bot.seen == [("how long will it take", "c1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "x" * 2001}])
async def test_a_malformed_turn_is_rejected(redis, body):
    response = await call(build_app(StubBot(), generous(redis)), **body)

    assert response.status_code == 422


# --- being refused ----------------------------------------------------------


@pytest.mark.asyncio
async def test_over_the_limit_is_a_clean_429_with_a_retry_after(redis):
    """A limit is not a crash and not a hang. It is an answer: come back in N seconds."""
    stingy = TurnLimiter(
        per_minute=RateLimiter(redis, 1, 60, "client-minute"),
        per_day=RateLimiter(redis, 100, 86_400, "client-day"),
        upstream=RateLimiter(redis, 100, 60, "upstream-minute"),
    )
    app = build_app(StubBot(), stingy)

    assert (await call(app, message="first")).status_code == 200
    response = await call(app, message="second")

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert response.json()["retry_after_seconds"] > 0


@pytest.mark.asyncio
async def test_a_rate_limited_turn_never_reaches_the_bot(redis):
    """The whole point: the refusal must be cheaper than the work it prevents."""
    bot = StubBot()
    stingy = TurnLimiter(
        per_minute=RateLimiter(redis, 1, 60, "client-minute"),
        per_day=RateLimiter(redis, 100, 86_400, "client-day"),
        upstream=RateLimiter(redis, 100, 60, "upstream-minute"),
    )
    app = build_app(bot, stingy)

    await call(app, message="first")
    await call(app, message="second")

    assert len(bot.seen) == 1  # the second turn cost no Gemini call at all


@pytest.mark.asyncio
async def test_gemini_refusing_us_is_a_429_and_not_a_500(redis):
    """Their request was fine. We simply cannot serve it this second."""
    bot = StubBot(raises=UpstreamRateLimited(retry_after=30))

    response = await call(build_app(bot, generous(redis)), message="how long for a refund")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"
    assert "limit" in response.json()["detail"]


# --- what the turn leaves behind --------------------------------------------


async def get(app, path: str, headers: dict | None = None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers or {})


@pytest.mark.asyncio
async def test_an_answered_turn_is_counted(redis):
    metrics = MetricsStore(redis)
    app = build_app(StubBot(ANSWERED), generous(redis), metrics)

    await call(app, message="how long for a refund")

    snapshot = await metrics.snapshot()
    assert snapshot["turns"] == 1
    assert snapshot["deflection_rate"] == 1.0
    assert snapshot["by_category"]["REFUND"]["answered"] == 1


@pytest.mark.asyncio
async def test_an_escalated_turn_is_counted_with_its_reason_and_category(redis):
    metrics = MetricsStore(redis)
    app = build_app(StubBot(ESCALATED), generous(redis), metrics)

    await call(app, message="what is your ceo paid")

    snapshot = await metrics.snapshot()
    assert snapshot["escalation_rate"] == 1.0
    assert snapshot["escalation_reasons"] == {"low_confidence": 1}
    assert snapshot["by_category"]["ORDER"]["escalated"] == 1


@pytest.mark.asyncio
async def test_a_rate_limited_turn_is_not_counted_as_a_turn(redis):
    """It never reached the bot, so it is not a deflection or an escalation.

    Counting refusals as turns would quietly inflate the escalation rate with
    traffic the bot never saw, and the number would look like a product problem
    rather than a capacity one.
    """
    metrics = MetricsStore(redis)
    stingy = TurnLimiter(
        per_minute=RateLimiter(redis, 1, 60, "client-minute"),
        per_day=RateLimiter(redis, 100, 86_400, "client-day"),
        upstream=RateLimiter(redis, 100, 60, "upstream-minute"),
    )
    app = build_app(StubBot(), stingy, metrics)

    await call(app, message="first")
    await call(app, message="second")  # 429

    assert (await metrics.snapshot())["turns"] == 1


@pytest.mark.asyncio
async def test_the_metrics_endpoint_reports_the_snapshot(redis):
    metrics = MetricsStore(redis)
    app = build_app(StubBot(ANSWERED), generous(redis), metrics)
    await call(app, message="how long for a refund")

    body = (await get(app, "/admin/metrics")).json()

    assert body["turns"] == 1
    assert body["deflection_rate"] == 1.0
    assert body["false_answer_rate"] is None  # and it says why, see test_observability


@pytest.fixture
def admin_key(monkeypatch):
    """Turn the admin gate on for a test, and reset the cached settings after."""
    monkeypatch.setenv("ADMIN_API_KEY", "s3cret")
    get_settings.cache_clear()
    yield "s3cret"
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_the_metrics_endpoint_is_closed_when_a_key_is_configured(redis, admin_key):
    """An untested auth check is a decorative one."""
    app = build_app(StubBot(), generous(redis))

    assert (await get(app, "/admin/metrics")).status_code == 401
    assert (await get(app, "/admin/metrics", {"x-admin-key": "wrong"})).status_code == 401
    assert (await get(app, "/admin/metrics", {"x-admin-key": admin_key})).status_code == 200


@pytest.mark.asyncio
async def test_the_metrics_endpoint_is_open_when_no_key_is_configured(redis):
    """Fine locally, and the README says not to ship it that way."""
    get_settings.cache_clear()
    app = build_app(StubBot(), generous(redis))

    assert (await get(app, "/admin/metrics")).status_code == 200

"""The HTTP surface: request/response schemas and the chat endpoint.

An answered turn and an escalated turn are different response shapes, discriminated
by `status`, because they are different events. A client that forgets to check a
boolean flag would happily render an escalation's empty `answer` field as an answer;
a client that forgets to check `status` here gets nothing to render at all, which is
the failure we want.

The escalation payload is returned to the caller in full - the retrieved chunks and
their scores included. In production that would go to the ticketing system rather
than down the wire to a customer, but this API is also what a demo front end reads,
and the interesting thing about this bot is precisely *what it declines to answer
and why*. Hiding the evidence would hide the product.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.bot import Answered, Escalated, SupportBot
from app.config import get_settings
from app.llm import UpstreamRateLimited
from app.observability import MetricsStore, Timer, TurnMetrics
from app.ratelimit import TurnLimiter

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Optional: a new conversation is minted when it is absent. Supplying one is
    # what makes a turn a follow-up rather than a fresh question.
    conversation_id: str | None = Field(default=None, max_length=128)


class Source(BaseModel):
    doc_id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    status: str  # "answered" | "escalated"
    conversation_id: str
    confidence: float
    condensed_query: str

    # answered only
    answer: str | None = None
    citations: list[str] | None = None

    # escalated only: what a human agent needs to pick this up cold
    reason: str | None = None
    # The customer's own words, not the rewrite we searched on. An agent needs to
    # read what was actually typed: condensation is a retrieval device and it can
    # be wrong, so handing over only its output would hide the very mistake most
    # likely to have caused the escalation.
    query: str | None = None
    history: list[dict] | None = None
    retrieved: list[Source] | None = None


def get_bot(request: Request) -> SupportBot:
    return request.app.state.bot


def get_limiter(request: Request) -> TurnLimiter:
    return request.app.state.limiter


def get_metrics(request: Request) -> MetricsStore:
    return request.app.state.metrics


def client_identity(request: Request) -> str:
    """Who to charge this request to.

    An API key if one is presented, otherwise the peer address. X-Forwarded-For is
    deliberately *not* trusted: it is caller-controlled, so honouring it would let
    anyone reset their own limit by inventing a header. Behind a real proxy this
    needs the proxy's forwarded address and an explicit trusted-hop config, which
    is a deployment decision rather than a default.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    bot: SupportBot = Depends(get_bot),
    limiter: TurnLimiter = Depends(get_limiter),
    metrics: MetricsStore = Depends(get_metrics),
) -> JSONResponse:
    identity = client_identity(request)
    decision = await limiter.check(identity)

    if not decision.allowed:
        # A limit is not an error the caller can fix by retrying immediately, so
        # say when they can. Retry-After is the header clients and CDNs already
        # understand; the body repeats it for anything reading JSON.
        logger.info(
            "rate limited",
            extra={"fields": {"identity": identity, "retry_after": decision.retry_after}},
        )
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(decision.retry_after)},
            content={
                "detail": "rate limit reached",
                "retry_after_seconds": decision.retry_after,
            },
        )

    conversation_id = body.conversation_id or _new_conversation_id()

    try:
        with Timer() as timer:
            result = await bot.handle(body.message, conversation_id)
    except UpstreamRateLimited as exc:
        # Gemini refused us, not the caller. Their request was fine; we just cannot
        # serve it this second, so it is a 429 with guidance rather than a 500.
        logger.warning(
            "upstream rate limited",
            extra={"fields": {"conversation_id": conversation_id}},
        )
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
            content={
                "detail": "the assistant is briefly over its usage limit",
                "retry_after_seconds": exc.retry_after,
            },
        )

    await _record(metrics, result, conversation_id, timer.elapsed_ms)

    if isinstance(result, Answered):
        return JSONResponse(
            content=ChatResponse(
                status="answered",
                conversation_id=conversation_id,
                confidence=result.confidence,
                condensed_query=result.condensed_query,
                answer=result.text,
                citations=list(result.citations),
            ).model_dump(exclude_none=True)
        )

    return JSONResponse(content=_escalation_body(result, conversation_id).model_dump(exclude_none=True))


async def _record(
    metrics: MetricsStore,
    result: Answered | Escalated,
    conversation_id: str,
    latency_ms: float,
) -> None:
    """One log line and one set of counters per turn.

    The category comes from the best chunk even when the turn escalated: "we keep
    escalating REFUND questions" is exactly the finding the admin endpoint exists to
    surface, and it is unavailable if escalations are recorded without a category.
    """
    escalated = isinstance(result, Escalated)
    category = _category_of(result)

    logger.info(
        "turn escalated" if escalated else "turn answered",
        extra={
            "fields": {
                "conversation_id": conversation_id,
                "escalated": escalated,
                "reason": str(result.reason) if escalated else None,
                "confidence": round(result.confidence, 4),
                "category": category,
                "latency_ms": round(latency_ms, 1),
                "citations": None if escalated else list(result.citations),
            }
        },
    )

    await metrics.record(
        TurnMetrics(
            escalated=escalated,
            reason=str(result.reason) if escalated else None,
            category=category,
            confidence=result.confidence,
            latency_ms=latency_ms,
        )
    )


def _category_of(result: Answered | Escalated) -> str | None:
    if isinstance(result, Escalated):
        return result.chunks[0].candidate.category if result.chunks else None
    # An Answered cites documents; the category is the one it answered out of.
    return result.category


@router.get("/admin/metrics")
async def admin_metrics(
    request: Request,
    metrics: MetricsStore = Depends(get_metrics),
) -> JSONResponse:
    """Deflection and escalation rates, overall and by category.

    Gated on ADMIN_API_KEY when one is set. It exposes no message content - only
    counts - but "how often does this bot fail" is not a number to leave open to the
    internet by default.
    """
    expected = get_settings().admin_api_key
    if expected and request.headers.get("x-admin-key") != expected:
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    return JSONResponse(content=await metrics.snapshot())


def _escalation_body(result: Escalated, conversation_id: str) -> ChatResponse:
    return ChatResponse(
        status="escalated",
        conversation_id=conversation_id,
        confidence=result.confidence,
        condensed_query=result.condensed_query,
        reason=str(result.reason),
        query=result.query,
        history=[
            {
                "question": turn.question,
                "answer": turn.answer,
                "escalated": turn.escalated,
            }
            for turn in result.history
        ],
        # The chunks that lost, with their scores. They are the reason this
        # escalated, so they travel with it - see app/bot.py.
        retrieved=[
            Source(doc_id=chunk.doc_id, title=chunk.candidate.title, score=chunk.score)
            for chunk in result.chunks
        ],
    )


def _new_conversation_id() -> str:
    import uuid

    return uuid.uuid4().hex

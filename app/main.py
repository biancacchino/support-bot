"""FastAPI entrypoint for the support bot."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient

from app.api import router
from app.bot import SupportBot
from app.config import get_settings
from app.conversation import Condenser, ConversationStore
from app.llm import AnswerGenerator, build_gemini
from app.observability import MetricsStore, configure_logging, new_request_id, request_id_var
from app.ratelimit import RateLimiter, TurnLimiter
from app.reranker import Reranker, build_cross_encoder
from app.retrieval import Retriever, build_encoder

logger = logging.getLogger(__name__)


# At import, not in the lifespan: uvicorn logs "Started server process" and
# "Waiting for application startup" before lifespan runs, and those lines would come
# out as prose in an otherwise-JSON stream.
configure_logging(get_settings().log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.metrics = MetricsStore(app.state.redis)

    # The models load once, here, and not on the first request. Both are hundreds
    # of MB and take seconds to warm; doing it lazily would hand that latency to
    # whichever customer happened to arrive first.
    generate = build_gemini(settings)
    app.state.bot = SupportBot(
        retriever=Retriever(
            app.state.qdrant,
            build_encoder(settings),
            settings.qdrant_collection,
            settings.retrieval_top_n,
        ),
        reranker=Reranker(
            build_cross_encoder(settings), settings.rerank_top_k, settings.confidence_threshold
        ),
        generator=AnswerGenerator(generate),
        condenser=Condenser(generate),
        store=ConversationStore(app.state.redis, settings.conversation_ttl_seconds),
    )
    app.state.limiter = build_limiter(app.state.redis, settings)

    try:
        yield
    finally:
        await app.state.qdrant.close()
        await app.state.redis.aclose()


def build_limiter(redis, settings) -> TurnLimiter:
    return TurnLimiter(
        per_minute=RateLimiter(
            redis, settings.rate_limit_per_minute, window_seconds=60, namespace="client-minute"
        ),
        per_day=RateLimiter(
            redis, settings.rate_limit_per_day, window_seconds=86_400, namespace="client-day"
        ),
        # Sized in turns, not requests: one turn can cost two Gemini calls.
        upstream=RateLimiter(
            redis, settings.upstream_turns_per_minute, window_seconds=60, namespace="upstream-minute"
        ),
        upstream_day=RateLimiter(
            redis, settings.upstream_turns_per_day, window_seconds=86_400, namespace="upstream-day"
        ),
    )


app = FastAPI(title="Support Bot", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def tag_request(request, call_next):
    """Give every request an id, and hand it back in the response.

    A customer with a complaint has a timestamp and, if we put it in the response, an
    id. Without one, tracing "the bot told me refunds take 30 days" back to the turn
    that said it means grepping logs by prose.
    """
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)

    response.headers["X-Request-ID"] = request_id
    return response


async def _check_qdrant(app: FastAPI) -> tuple[bool, str]:
    try:
        await app.state.qdrant.get_collections()
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


async def _check_redis(app: FastAPI) -> tuple[bool, str]:
    try:
        await app.state.redis.ping()
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Liveness plus dependency reachability.

    Reports 503 when a dependency is unreachable so that `docker-compose up`
    failing to network the containers together surfaces here rather than at
    the first real query.

    Takes the app off the request rather than closing over the module-level
    `app`. Reading the global made this handler untestable - which is exactly why
    Codex's Phase 9 pass found it had no test - and quietly tied "is the app
    healthy" to one particular app object.
    """
    qdrant_ok, qdrant_detail = await _check_qdrant(request.app)
    redis_ok, redis_detail = await _check_redis(request.app)

    healthy = qdrant_ok and redis_ok
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "dependencies": {
            "qdrant": {"ok": qdrant_ok, "detail": qdrant_detail},
            "redis": {"ok": redis_ok, "detail": redis_detail},
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)


# Last, so that /chat, /health and /admin/metrics are matched before the catch-all:
# Starlette tries routes in registration order. The page is a demo front end for the
# API, not a second product - one file, no build step, and it reads the same JSON
# any other client would.
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")

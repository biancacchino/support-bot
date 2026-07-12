"""FastAPI entrypoint for the support bot."""

import logging
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from app.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield
    finally:
        await app.state.qdrant.close()
        await app.state.redis.aclose()


app = FastAPI(title="Support Bot", version="0.1.0", lifespan=lifespan)


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
async def health() -> JSONResponse:
    """Liveness plus dependency reachability.

    Reports 503 when a dependency is unreachable so that `docker-compose up`
    failing to network the containers together surfaces here rather than at
    the first real query.
    """
    qdrant_ok, qdrant_detail = await _check_qdrant(app)
    redis_ok, redis_detail = await _check_redis(app)

    healthy = qdrant_ok and redis_ok
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "dependencies": {
            "qdrant": {"ok": qdrant_ok, "detail": qdrant_detail},
            "redis": {"ok": redis_ok, "detail": redis_detail},
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)

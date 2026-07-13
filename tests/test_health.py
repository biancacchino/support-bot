"""The /health endpoint, and who a request is charged to.

Both were flagged by Codex during the Phase 9 coverage pass as having no test.
/health is the endpoint compose gates container readiness on, so a bug in it is a
bug in "is the app up" - which is precisely the question nobody double-checks.
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import client_identity
from app.main import _check_qdrant, _check_redis, health


class FakeQdrant:
    def __init__(self, error: str | None = None) -> None:
        self._error = error

    async def get_collections(self):
        if self._error:
            raise RuntimeError(self._error)
        return []


class FakeRedisPing:
    def __init__(self, error: str | None = None) -> None:
        self._error = error

    async def ping(self):
        if self._error:
            raise RuntimeError(self._error)
        return True


def build_app(qdrant, redis) -> FastAPI:
    app = FastAPI()
    app.state.qdrant = qdrant
    app.state.redis = redis
    app.add_api_route("/health", health, methods=["GET"])
    return app


async def get_health(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/health")


# --- health -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_is_200_when_both_dependencies_answer():
    app = build_app(FakeQdrant(), FakeRedisPing())

    response = await get_health(app)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["qdrant"]["ok"] is True
    assert body["dependencies"]["redis"]["ok"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("qdrant", "redis", "broken"),
    [
        (FakeQdrant("connection refused"), FakeRedisPing(), "qdrant"),
        (FakeQdrant(), FakeRedisPing("connection refused"), "redis"),
        (FakeQdrant("down"), FakeRedisPing("down"), "qdrant"),
    ],
)
async def test_health_is_503_when_a_dependency_is_unreachable(qdrant, redis, broken):
    """Compose gates readiness on this. Reporting 200 while Qdrant is down would
    mean the container goes healthy and the first customer discovers the outage."""
    response = await get_health(build_app(qdrant, redis))

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"][broken]["ok"] is False
    assert body["dependencies"][broken]["detail"]  # says what went wrong


@pytest.mark.asyncio
async def test_a_dependency_failure_is_reported_not_raised():
    """A health check that 500s tells you nothing about *which* dependency died."""
    ok, detail = await _check_qdrant(build_app(FakeQdrant("boom"), FakeRedisPing()))
    assert (ok, detail) == (False, "boom")

    ok, detail = await _check_redis(build_app(FakeQdrant(), FakeRedisPing("boom")))
    assert (ok, detail) == (False, "boom")


# --- who gets charged for a request -----------------------------------------


class FakeRequest:
    def __init__(self, headers: dict | None = None, client=None) -> None:
        self.headers = headers or {}
        self.client = client


class Peer:
    def __init__(self, host: str) -> None:
        self.host = host


def test_an_api_key_identifies_a_caller_before_their_address_does():
    identity = client_identity(FakeRequest({"x-api-key": "abc"}, Peer("1.2.3.4")))

    assert identity == "key:abc"


def test_without_a_key_a_caller_is_their_peer_address():
    assert client_identity(FakeRequest(client=Peer("1.2.3.4"))) == "ip:1.2.3.4"


def test_a_forwarded_for_header_cannot_change_who_you_are():
    """It is caller-controlled. Honouring it would let anyone reset their own limit
    by inventing a header, which is a rate limiter that does not limit anything."""
    identity = client_identity(
        FakeRequest({"x-forwarded-for": "9.9.9.9"}, Peer("1.2.3.4"))
    )

    assert identity == "ip:1.2.3.4"


def test_a_request_with_no_peer_still_gets_an_identity():
    """Rare (ASGI usually supplies one), but a None here would crash the endpoint
    rather than rate limit it - the limiter must not be the thing that breaks."""
    assert client_identity(FakeRequest()) == "ip:unknown"

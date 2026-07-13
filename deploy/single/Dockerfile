# The demo build: the whole stack in one container, Qdrant and Redis included.
#
# It exists because the hosts that will run a container for free will run exactly
# one, and it is deliberately not tied to any of them - it listens on $PORT, keeps
# no state on disk, and needs nothing but two secrets in the environment. Cloud Run
# is where it runs today; that is a deployment choice, not a property of this file.
#
# docker-compose.yml remains the real topology and the one local dev runs against.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    QDRANT_URL=http://localhost:6333 \
    REDIS_URL=redis://localhost:6379/0

WORKDIR /srv

# libunwind8 is not decoration: the Qdrant binary below links against
# libunwind-ptrace.so.0, which the slim image does not ship, and without it the
# server dies at exec with a shared-library error and takes the demo with it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl redis-server libunwind8 \
    && rm -rf /var/lib/apt/lists/*

# The server binary, lifted from the same image tag compose pins, so the Space and
# the compose stack run the same Qdrant.
COPY --from=qdrant/qdrant:v1.18.2 /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=qdrant/qdrant:v1.18.2 /qdrant/config /qdrant/config

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY kb ./kb
COPY web ./web
COPY scripts ./scripts
COPY pytest.ini .
COPY deploy/single/entrypoint.sh /usr/local/bin/entrypoint.sh

# Pull the embedder and the reranker at build time. A free Space is paused after a
# spell of inactivity and cold-starts on the next visit - and a recruiter is exactly
# the visitor who arrives during that cold start. Downloading a few hundred MB of
# weights then would be the whole first impression.
RUN python -c "\
from app.config import get_settings; \
from app.retrieval import build_encoder; \
from app.reranker import build_cross_encoder; \
s = get_settings(); build_encoder(s); build_cross_encoder(s)"

RUN useradd --create-home --uid 1000 appuser \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R appuser:appuser /models /srv
USER appuser

EXPOSE 8000

CMD ["/usr/local/bin/entrypoint.sh"]

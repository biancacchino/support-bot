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

# Pull the embedder and the reranker at build time. The service scales to zero and
# cold-starts on the next visit - and a recruiter is exactly the visitor who arrives
# during that cold start. Downloading a few hundred MB of weights then would be the
# whole first impression.
#
# The weights arrive in the build context, already downloaded. They are NOT fetched
# here, and that is the whole point: Cloud Build cannot reach HuggingFace's Xet
# storage. Through the bridge CDN it gets a 403 ("invalid key pair id"); through the
# native client it hangs with no timeout it will honour. Four builds died on that,
# three of them as silent 25-minute stalls, while the identical step took a minute on
# a laptop.
#
# So the laptop does it. scripts/deploy_cloudrun.sh populates .model-cache/ once and
# stages it here. The build now has no third-party network dependency at all, which is
# the property worth having: a deploy should not be able to fail because someone
# else's CDN is unreachable from someone else's builder.
#
# HF_HUB_OFFLINE makes that guarantee enforceable rather than aspirational - if the
# cache is ever incomplete, this fails loudly at build instead of silently reaching
# for the network at 3am.
COPY models /models
ENV HF_HUB_OFFLINE=1

RUN python -u -c "\
from app.config import get_settings; \
from app.retrieval import build_encoder; \
from app.reranker import build_cross_encoder; \
s = get_settings(); \
print('loading embedder from cache', flush=True); build_encoder(s); \
print('loading reranker from cache', flush=True); build_cross_encoder(s); \
print('weights present', flush=True)"

RUN useradd --create-home --uid 1000 appuser \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R appuser:appuser /models /srv
USER appuser

EXPOSE 8000

CMD ["/usr/local/bin/entrypoint.sh"]

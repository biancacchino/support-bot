# support-bot

A RAG support bot for a fictional wholesaler (Northwind).
It answers customer questions from a help-centre knowledge base and escalates to a human when it is not confident.

Python 3.12, FastAPI, Qdrant for vector search, Redis for conversation state, sentence-transformers for embedding (`all-MiniLM-L6-v2`) and reranking (`ms-marco-MiniLM-L-6-v2`), Gemini `3.1-flash-lite` for answer generation.

## Run locally

Requires a container runtime (this repo is developed on Colima) and a Gemini API key from [AI Studio](https://aistudio.google.com/apikey).

```sh
cp .env.example .env          # then add GEMINI_API_KEY
colima start --cpu 4 --memory 8 --disk 60
docker compose up -d --build
curl -s localhost:8000/health
```

Ingest the knowledge base into Qdrant, after `up` and again whenever a file under `kb/` changes:

```sh
docker compose exec app python scripts/ingest.py
```

`kb/` and `web/` are bind-mounted, so editing a document or the chat page needs no rebuild.
Ingestion is idempotent: re-running overwrites the same points and prunes chunks whose document was edited or deleted.

Then open `localhost:8000` for the chat page, a single static file served by the app itself.

## API

```sh
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "where is my order right now"}'
```

Answers and escalations are different response shapes, discriminated by `status`.

```jsonc
// answered
{"status": "answered", "conversation_id": "...", "confidence": 0.55,
 "answer": "You can track your order by...", "citations": ["tracking-your-order"]}

// escalated - no answer field
{"status": "escalated", "conversation_id": "...", "confidence": 0.03,
 "reason": "low_confidence", "history": [...],
 "retrieved": [{"doc_id": "placing-an-order", "title": "Placing an order", "score": 0.03}]}
```

Pass a `conversation_id` back to make the next turn a follow-up.
Omit it and a new conversation is minted.
Escalation is sticky: once a conversation escalates, every later turn escalates too.

Other endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | 200 only when Qdrant and Redis are both reachable |
| `GET /admin/metrics` | Deflection and escalation rates by category, gated on `X-Admin-Key` |

Logs are JSON, one object per line.
Every turn logs the request id, conversation id, latency, confidence, category, and whether it escalated.
The request id also comes back in the `X-Request-ID` response header.

## Configuration

Environment variables, set in `.env` locally (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Required. From [AI Studio](https://aistudio.google.com/apikey). |
| `ADMIN_API_KEY` | unset | Required in the `X-Admin-Key` header on `/admin/metrics`. Unset leaves the endpoint open. Generate with `openssl rand -hex 32`. |
| `CONFIDENCE_THRESHOLD` | `0.2` | Reranker score below which a turn escalates. Set by the eval, see `docs/benchmark.md`. |
| `GEMINI_RPM` / `GEMINI_RPD` | `15` / `1000` | Shared upstream budget, in turns per minute and per day. Check the real limit for your key. |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs query text. Do not use it on a public deployment. |

Rate limiting runs on two axes: per caller (10/minute, 200/day, bucketed on the peer IP) and a global upstream budget on Gemini.
Both return a `429` with a `Retry-After` header.

## Tests

```sh
# everything, including the retrieval acceptance tests (they load the real model)
docker compose run --rm -v "$PWD/tests:/srv/tests" app python -m pytest tests -q

# fast only, no model weights
docker compose run --rm -v "$PWD/tests:/srv/tests" app python -m pytest tests -q -m "not slow"
```

Scripts for exercising the pipeline by hand:

```sh
docker compose exec app python scripts/ingest.py --check          # ingest, then assert real queries retrieve the right doc
docker compose exec app python scripts/ingest.py --dry-run        # what would be ingested, without writing
docker compose exec app python scripts/check_kb_coverage.py       # corpus vs the Bitext taxonomy
docker compose exec app python scripts/ask.py --all               # 3 questions the KB answers, 3 it does not
docker compose exec app python scripts/ask.py --chat              # a two-turn conversation
docker compose exec app python scripts/ask.py --sticky            # an escalated conversation stays escalated

# the benchmark: 320 Bitext queries through retrieval, rerank and the gate. No Gemini quota.
docker compose run --rm -v "$PWD/scripts:/srv/scripts" -v "$PWD/eval:/srv/eval" \
  -v "$PWD/docs:/srv/docs" app python scripts/eval.py
```

`scripts/ask.py --all` is the quickest way to see the bot work.
`ingest.py --check` is the one that matters: an ingest can report success while retrieving the wrong document for every query.

## Deploy

The public demo runs the whole stack in one container (`deploy/single/Dockerfile`) on Cloud Run.
`docker-compose.yml` is the real topology and the one local dev runs against.

Needs a Google Cloud project with billing enabled, then `gcloud auth login` and `gcloud config set project <id>`.

```sh
GEMINI_API_KEY=... ADMIN_API_KEY=... scripts/deploy_cloudrun.sh
```

The container listens on `$PORT`, keeps nothing on disk, and needs nothing but those two secrets in the environment, so it is not tied to Cloud Run.
The service is pinned to `--max-instances 1` because conversation history lives in the container's own Redis.

Read `docs/open-questions.md` before deploying anywhere public.
Customer text reaches Gemini and Redis un-redacted, and rate limiting buckets on the peer IP, which behind a proxy is one shared bucket for every visitor.

## Repo map

- `app/` - the FastAPI application. `bot.py` (one turn, end to end), `api.py` (`POST /chat`, `GET /admin/metrics`), `retrieval.py` (stage 1 vector search), `reranker.py` (stage 2 rerank and the confidence gate), `llm.py` (grounded answer generation), `conversation.py` (Redis history and query condensation), `ingestion.py`, `ratelimit.py`, `observability.py`, `config.py`, `main.py`.
- `kb/` - the knowledge base. 25 help-centre documents, each mapped in frontmatter to the Bitext intents it answers.
- `scripts/` - `ingest.py`, `ask.py`, `eval.py`, `check_kb_coverage.py`, `sample_eval_set.py`, `deploy_cloudrun.sh`.
- `eval/` - `queries.jsonl`, the frozen 320-query eval set.
- `tests/` - pytest suite. The `slow` marker means it loads model weights.
- `web/` - `index.html`, the chat page. One file, no build step.
- `deploy/single/` - the one-container build for the public demo.
- `docs/` - see below.
- `tasks/` - build progress and running notes.

## Docs

- `docs/intent-taxonomy.md` - derived from the Bitext dataset. Ground truth for both the KB and the eval, so read it before touching either.
- `docs/benchmark.md` - generated by `scripts/eval.py`. Do not hand-edit.
- `docs/open-questions.md` - what this does and does not do about ticketing, PII and adversarial input.
- `docs/learnings.md` - why the system is built the way it is, and what the eval taught me.

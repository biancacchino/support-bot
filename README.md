# support-bot

A RAG support bot for a fictional wholesaler (Northwind), answering customer questions from a help-centre knowledge base and escalating when it is not confident.

FastAPI, Qdrant for vector search, Redis for conversation state, local sentence-transformers for embedding and reranking, Gemini for answer generation.

## Run locally

Container runtime is Colima, not Docker Desktop:

```sh
colima start --cpu 4 --memory 8 --disk 60
docker compose up -d --build
curl -s localhost:8000/health
```

`/health` returns 200 only when Qdrant and Redis are both reachable, so a networking problem surfaces there rather than at the first query.

Ingest the knowledge base into Qdrant. Do this after `up`, and again whenever a file under `kb/` changes:

```sh
docker compose exec app python scripts/ingest.py
```

`kb/` is bind-mounted, so editing a document and re-running the script is enough. No rebuild.

Ingestion is idempotent. Chunk IDs are derived from the document, so re-running overwrites the same points instead of stacking a second copy of the corpus, and chunks whose document was edited or deleted are pruned.

## Verify

```sh
# chunking logic, no model download needed
docker compose run --rm -v "$PWD/tests:/srv/tests" app python -m pytest tests -q

# the KB corpus matches the Bitext taxonomy
docker compose exec app python scripts/check_kb_coverage.py

# end to end: ingest, then put real user phrasings through vector search
# and assert the right document comes back
docker compose exec app python scripts/ingest.py --check

# what would be ingested, without writing to Qdrant
docker compose exec app python scripts/ingest.py --dry-run
```

`--check` is the one that matters. An ingest can report success while retrieving the wrong document for every query, and only `--check` catches that.

## Repo map

- `app/` - FastAPI application. `config.py` (env settings), `ingestion.py` (chunk, embed, upsert), `main.py` (entrypoint, `/health`).
- `kb/` - the knowledge base: 25 help-centre documents, each mapped in frontmatter to the Bitext intents it answers.
- `scripts/` - `ingest.py` (re-ingestion CLI), `check_kb_coverage.py` (corpus vs taxonomy).
- `docs/` - project documentation, including the derived intent taxonomy.
- `tasks/` - build progress and running notes.

## Stack

Python 3.12, FastAPI, Qdrant, Redis, sentence-transformers (`all-MiniLM-L6-v2` embedding, `ms-marco-MiniLM-L-6-v2` reranker), Gemini `2.5-flash-lite` via the `google-genai` SDK.

## Decisions

Read `docs/intent-taxonomy.md` before touching the KB or the eval. The taxonomy is derived from the Bitext dataset and is ground truth for both.

The embedding model reads at most 256 tokens and silently truncates beyond that, so `chunk_size_tokens` is capped well under it. Ingestion refuses to run if the budget is set too high rather than let the tail of a chunk be stored but never embedded.

The Qdrant image and `qdrant-client` are pinned to the same minor version. They do not tolerate drift: the client warns on every call, and the on-disk format does not survive a wide version jump.

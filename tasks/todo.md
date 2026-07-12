# Support Bot - Build Progress

Tracking the phases from the build task list.
Source docs (`prd-full.md`, `support-bot-spec.md`) were not present in the repo.
Folder layout is reconstructed from the module names the task list itself names, per Bianca's call on 2026-07-12.

## Known gaps from the missing docs

- ~~Phase 1.1: the Bitext intent mapping is not recoverable.~~
  Resolved 2026-07-12. Rather than reconstruct the lost mapping, the taxonomy was re-derived from the Bitext dataset itself (task 1.0).
  See `docs/intent-taxonomy.md`. This is now ground truth regardless of what the original plan said.
- Phase 11.1: "the earlier eval design" is not recoverable, but 1.0 unblocks it.
  The eval can now be designed against the derived taxonomy.
- Phase 12: Gemini's review pass has no PRD to check the implementation against.

## Phase 0 - Setup

- [x] 0.1 Scaffold repo structure
- [x] 0.2 `docker-compose.yml` (app + Qdrant + Redis, networked)
- [x] 0.3 `Dockerfile` for the FastAPI app
- [x] 0.4 `requirements.txt`
- [x] 0.5 `config.py` with env-based settings

Acceptance: met and verified on 2026-07-12.
`docker compose up -d --build` brings up all three containers; the app container reaches `healthy`
and `GET /health` returns HTTP 200 with both Qdrant and Redis reporting reachable.

Local container runtime is Colima (`colima start --cpu 4 --memory 8 --disk 60`), not Docker Desktop.
The Homebrew `docker` formula does not register the compose plugin, so `docker compose` needs a symlink:
`ln -sf /opt/homebrew/opt/docker-compose/bin/docker-compose ~/.docker/cli-plugins/docker-compose`

## Phase 1 - Knowledge Base & Ingestion

- [x] 1.0 Derive the intent taxonomy from the Bitext dataset (added; the task list assumed it already existed)
- [x] 1.1 Write 25 Northwind help-centre docs, each mapped to in-scope Bitext intents
- [x] 1.2 `ingestion.py` - chunking, local embedding, upsert to Qdrant with metadata
- [x] 1.3 CLI script or endpoint to trigger re-ingestion when docs change

Scope: 8 categories, 22 intents in. 3 categories, 5 intents out (`CONTACT`, `FEEDBACK`, `SUBSCRIPTION`).
The 5 out-of-scope intents are kept as near-miss negatives for the Phase 3 confidence gate and the Phase 11 eval.
Reasoning in `docs/intent-taxonomy.md`.

Verify with `python scripts/check_kb_coverage.py`, which fails on an intent typo, an out-of-scope intent,
or an in-scope intent with no document. Confirmed it fails on all three, not just that it passes.

Ingestion verified end to end on 2026-07-12: 162 chunks from 25 documents, 384-dim, and all 8 retrieval
smoke queries return their expected document (`scripts/ingest.py --check`).
Re-running ingest leaves the point count at 162 rather than doubling it, and ingesting a corpus with a
document removed pruned exactly that document's 8 points, so re-ingestion is genuinely idempotent.

Three of the 8 smoke queries land their document at rank 2 or 3 rather than 1.
That is expected at this stage and is the whole reason Phase 2 reranks: raw cosine similarity over
chunk-level vectors puts near-neighbours (delivery vs shipping, cancelling vs cancellation fees) ahead of
the right document often enough to matter. Recheck these ranks after the reranker lands.

## Fixed along the way

- `chunk_size_tokens` was 400, but `all-MiniLM-L6-v2` reads at most 256 tokens and silently truncates
  past that, so the tail of every long chunk would have been stored in Qdrant but never embedded.
  Now 224 plus a 32-token reserve for the `title > heading` prefix each chunk is embedded under, and
  ingestion hard-fails rather than truncate if the budget is raised past what the model can read.
- The Qdrant image (v1.12.1) was 6 minor versions behind the `qdrant-client` that `>=1.9,<2.0` resolved
  to (1.18), which warned on every call. Both are now pinned to 1.18.
  The old storage volume could not be read by the new server, so it was wiped and re-ingested.
  Safe: the collection is derived entirely from `kb/` and rebuilds in seconds.
- `kb/` is now bind-mounted into the app container, so 1.3 re-ingestion picks up edited docs without a
  rebuild, which is the entire point of having it.

## Deviations from the task list

- `google-generativeai` is deprecated (Google ended support in 2025) in favour of the `google-genai` SDK.
  Using `google-genai` instead, since a new project should not start on a dead SDK.
  Both reach `gemini-2.5-flash-lite`, so nothing downstream changes.
- Knowledge base corpus lives in `kb/`, not `docs/`, so it does not collide with project documentation.

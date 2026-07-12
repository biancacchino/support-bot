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
- [ ] 1.2 `ingestion.py` - chunking, local embedding, upsert to Qdrant with metadata
- [ ] 1.3 CLI script or endpoint to trigger re-ingestion when docs change

Scope: 8 categories, 22 intents in. 3 categories, 5 intents out (`CONTACT`, `FEEDBACK`, `SUBSCRIPTION`).
The 5 out-of-scope intents are kept as near-miss negatives for the Phase 3 confidence gate and the Phase 11 eval.
Reasoning in `docs/intent-taxonomy.md`.

Verify with `python scripts/check_kb_coverage.py`, which fails on an intent typo, an out-of-scope intent,
or an in-scope intent with no document. Confirmed it fails on all three, not just that it passes.

## Deviations from the task list

- `google-generativeai` is deprecated (Google ended support in 2025) in favour of the `google-genai` SDK.
  Using `google-genai` instead, since a new project should not start on a dead SDK.
  Both reach `gemini-2.5-flash-lite`, so nothing downstream changes.
- Knowledge base corpus lives in `kb/`, not `docs/`, so it does not collide with project documentation.

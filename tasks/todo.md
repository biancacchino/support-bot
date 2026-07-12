# Support Bot - Build Progress

Tracking the phases from the build task list.
Source docs (`prd-full.md`, `support-bot-spec.md`) were not present in the repo.
Folder layout is reconstructed from the module names the task list itself names, per Bianca's call on 2026-07-12.

## Known gaps from the missing docs

These are blocked or guessed until the source docs turn up.

- Phase 1.1: the Bitext intent mapping ("the earlier ground-truth plan") is not recoverable. Need the intent list before KB docs can be mapped to ground truth.
- Phase 11.1: "the earlier eval design" is not recoverable. Recall@k / MRR / out-of-domain sampling will need to be redesigned from scratch.
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

## Deviations from the task list

- `google-generativeai` is deprecated (Google ended support in 2025) in favour of the `google-genai` SDK.
  Using `google-genai` instead, since a new project should not start on a dead SDK.
  Both reach `gemini-2.5-flash-lite`, so nothing downstream changes.
- Knowledge base corpus lives in `kb/`, not `docs/`, so it does not collide with project documentation.

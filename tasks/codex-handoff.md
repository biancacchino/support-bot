# Handoff: Phases 9 and 10

Scope is exactly the two phases below and nothing else.
Do not refactor `app/`, do not change behaviour, do not "improve" anything you were not asked to.
If you find a bug in `app/`, write it down at the bottom of this file rather than fixing it.

## Phase 9 - Test suite

- [x] 9.1 Fill out `tests/test_retrieval.py` and `tests/test_api.py` with cases from Phases 2-6 that are not
      yet covered
- [x] 9.2 Coverage check - flag any endpoint or branch with no test

## Phase 10 - CI/CD

- [x] 10.1 GitHub Actions: lint (ruff) -> test (pytest) -> build image -> push to GHCR
- [x] 10.2 Fail the pipeline on lint or test failure, not just report it

## What already exists

144 tests pass today. This is not a greenfield test suite - it is a gap-filling exercise.
Read the existing tests before writing new ones, and do not duplicate what is there.

- `tests/test_ingestion.py` - chunking
- `tests/test_retrieval.py` - stage 1 vector search
- `tests/test_reranker.py` - cross-encoder rerank and the confidence gate
- `tests/test_llm.py` - grounded generation, citation enforcement
- `tests/test_conversation.py` - Redis history, query condensation
- `tests/test_bot.py` - one turn end to end, sticky escalation
- `tests/test_ratelimit.py` - per-caller limits, shared upstream budget
- `tests/test_api.py` - `POST /chat`, `GET /admin/metrics`
- `tests/test_observability.py` - JSON logging, metric counters

## How to run anything

The heavy dependencies (torch, sentence-transformers) live in the app container, not on the host.
There is no local venv with them. Everything runs through Docker:

```sh
# lint + full suite
docker compose run --rm -v "$PWD:/srv" app sh -c "ruff check app scripts tests && python -m pytest tests -q"

# fast tests only (no model weights)
docker compose run --rm -v "$PWD:/srv" app python -m pytest tests -q -m "not slow"
```

Tests marked `slow` load real model weights (~seconds). Tests are otherwise hermetic: in-memory Qdrant,
fakeredis, mocked LLM. **No test may call the real Gemini API**, and CI must never need `GEMINI_API_KEY`.

## Constraints for the CI workflow

- Lint and test must **fail the build**, not just annotate it. That is task 10.2 and it is the whole point.
- No `GEMINI_API_KEY` in CI. If a test needs one, that test is wrong - report it, do not add the secret.
- The `slow` tests download ~100MB of model weights from HuggingFace. Cache them (`~/.cache/huggingface`)
  keyed on the model names in `app/config.py`, or the pipeline will be slow and flaky.
- GHCR push needs `packages: write` permission and `GITHUB_TOKEN`; do not invent a new secret for it.
- Only push the image on `main`, not on pull requests. A PR from a fork must not be able to push an image.
- Pin action versions to a major tag (`actions/checkout@v4`), not to `@master`.

## Notes back to Claude Code

- Verification run: `docker compose run --rm -v "$PWD:/srv" app sh -c "ruff check app scripts tests && python -m pytest tests -q"` completed with `150 passed, 1 xfailed`. The existing xfail is the documented adversarial mixed-query gap.
- Found, not fixed: `Escalated.query` is not included in the `/chat` escalation response. Phase 6 requires the original query in the handoff payload; a passing API test cannot assert that requirement until the response schema/body is changed.
- Endpoint with no test: `GET /health` in `app/main.py`, including its Qdrant- or Redis-unavailable 503 branches.
- Branch with no test: `client_identity()`'s `request.client is None` fallback (`ip:unknown`). Normal ASGI requests always supply a client address. API-key and normal peer-address rate-limit paths are covered.
- Could not verify a real GHCR push from this workspace because no GitHub Actions `main` run or repository token is available. The workflow gates login and push on `refs/heads/main`, uses `GITHUB_TOKEN` with `packages: write`, and runs lint and tests as blocking shell steps before the image build.

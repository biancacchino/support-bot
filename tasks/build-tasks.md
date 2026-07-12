# Build Tasks — AI Support Ticket Triage & Response Bot
Derived from prd-full.md and support-bot-spec.md. Sequenced per the tool
division discussed: Claude Code builds core, Cursor for hands-on editing,
Codex CLI for scoped bounded tasks, Gemini CLI for final review.

Commit after each numbered task before moving to the next. Don't run two
agentic CLIs on the same uncommitted diff.

> Recovered on 2026-07-12 from the session transcript of 2026-07-12, where it
> had been pasted but never saved into the repo. Reproduced verbatim. The two
> source docs it was derived from (`prd-full.md`, `support-bot-spec.md`) are
> still missing; see `tasks/todo.md` for what that leaves unrecoverable.

---

## Phase 0 — Setup
**Tool: Claude Code CLI**

- [ ] 0.1 Scaffold repo structure per support-bot-spec.md folder layout
- [ ] 0.2 Write `docker-compose.yml` (app + Qdrant + Redis, networked)
- [ ] 0.3 Write `Dockerfile` for the FastAPI app
- [ ] 0.4 Write `requirements.txt` (fastapi, qdrant-client, redis,
      sentence-transformers, google-generativeai, cross-encoder deps, pytest)
- [ ] 0.5 `config.py` — env-based settings incl. `GEMINI_API_KEY`,
      `QDRANT_URL`, `REDIS_URL`, `CONFIDENCE_THRESHOLD`,
      `CONVERSATION_TTL_SECONDS`

**Acceptance criteria:** `docker-compose up` brings up all three containers
and the FastAPI health check endpoint responds.

---

## Phase 1 — Knowledge Base & Ingestion
**Tool: Claude Code CLI** (docs content: you write these yourself first)

- [ ] 1.1 Write 20-30 Northwind help-center-style docs (order status,
      cancellation, refund policy, subscription changes, shipping, etc.),
      each mapped to a Bitext intent per the earlier ground-truth plan
- [ ] 1.2 `ingestion.py` — chunking (~300-500 tokens, overlap), local
      embedding via sentence-transformers, upsert to Qdrant with metadata
      (source file, chunk index)
- [ ] 1.3 CLI script or endpoint to trigger re-ingestion when docs change

**Ties to PRD:** Requirements & Dependencies — "help doc corpus."
**Acceptance criteria:** running ingestion populates Qdrant with the
expected chunk count; re-running is idempotent (doesn't duplicate).

---

## Phase 2 — Retrieval (Stage 1: Vector Search)
**Tool: Claude Code CLI**

- [ ] 2.1 `retrieval.py` — embed incoming query, vector search Qdrant,
      return top ~10 candidates
- [ ] 2.2 Unit tests for retrieval against a small known doc set

**Acceptance criteria:** given a query with an obvious correct doc, that
doc appears in the top-10 candidates.

---

## Phase 3 — Reranking & Confidence
**Tool: Claude Code CLI**

- [ ] 3.1 `reranker.py` — cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks
      the top-10 candidates, returns top-k (k=4) by rerank score
- [ ] 3.2 Confidence gate: escalate if top rerank score < `CONFIDENCE_THRESHOLD`
- [ ] 3.3 Unit tests including the specific edge case this exists for:
      an off-topic-but-similar-embedding query that should now be caught
      by rerank score even if raw cosine similarity was high

**Ties to PRD:** P0 requirement — "confidence-gated escalation, scored on
reranker output, not raw similarity." This is the resolved
confidently-wrong-on-topic edge case — don't skip the specific test case.
**Acceptance criteria:** the rerank-based threshold demonstrably catches at
least one case that a similarity-only threshold would have missed (write
this as an explicit test, not just "it works").

---

## Phase 4 — Answer Generation & Citations
**Tool: Claude Code CLI**

- [ ] 4.1 `llm.py` — Gemini wrapper (`gemini-2.5-flash-lite`)
- [ ] 4.2 Prompt template that forces grounding in retrieved chunks only,
      no un-cited claims
- [ ] 4.3 Response schema includes source citation(s) on every answer —
      no exceptions, this is a P0 requirement
- [ ] 4.4 Mocked-LLM tests (don't burn real Gemini quota in CI)

**Ties to PRD:** P0 — "source citation on every answer, no exceptions."
**Acceptance criteria:** every non-escalated response includes at least one
source reference; test asserts this structurally (schema validation), not
just spot-checked.

---

## Phase 5 — Multi-Turn Conversation
**Tool: Claude Code CLI**

- [ ] 5.1 `conversation.py` — Redis-backed history keyed by `conversation_id`,
      TTL via `CONVERSATION_TTL_SECONDS`
- [ ] 5.2 Query condensation step: rewrite follow-up + history into a
      standalone query via `gemini-2.5-flash-lite` before retrieval
- [ ] 5.3 Append each turn (question + answer/escalation) to history
- [ ] 5.4 Tests: a two-turn conversation where turn 2 is only answerable
      correctly if turn 1's context was used in retrieval

**Ties to PRD:** P0 — "multi-turn conversation support."
**Acceptance criteria:** the turn-2 test in 5.4 actually fails if
condensation is disabled — prove the feature does something, don't just
test that the endpoint returns 200.

---

## Phase 6 — Escalation Handoff Payload
**Tool: Claude Code CLI**

- [ ] 6.1 Escalation response includes: original query, conversation
      history, top retrieved chunks (even below threshold), confidence score
- [ ] 6.2 Decide and implement: once a conversation escalates, does it stay
      flagged as human-owned for subsequent turns, or re-evaluate each turn?
      (Open question in PRD — pick one, document the choice in the README)

**Ties to PRD:** P0 — "escalation includes retrieved context" (agent
interview requirement).
**Acceptance criteria:** an escalated response is structurally distinct from
an answered response and carries enough for a human agent to act without
re-asking the customer anything.

---

## Phase 7 — Rate Limiting
**Tool: Claude Code CLI**

- [ ] 7.1 Redis-backed rate limiter, per-IP or per-API-key
- [ ] 7.2 Tune against Gemini Flash-Lite's actual free-tier RPM/RPD, not an
      assumed number — check current limits before hardcoding
- [ ] 7.3 Graceful 429 handling with retry-after guidance in the response

**Acceptance criteria:** exceeding the configured limit returns a clean
error, not a crash or a hung request.

---

## Phase 8 — Observability
**Tool: Claude Code CLI**

- [ ] 8.1 Structured logging: request id, conversation id, latency, rerank
      confidence score, escalated true/false
- [ ] 8.2 Basic admin endpoint/dashboard: deflection rate, escalation rate,
      false-answer rate by category (can start as a simple aggregation
      endpoint, doesn't need a UI for v1)

**Ties to PRD:** P1 — "admin dashboard (deflection/escalation/false-answer
rate by category)."

---

## Phase 9 — Test Suite
**Tool: Codex CLI** (bounded, well-defined task — good fit for a second tool)

- [ ] 9.1 Fill out `tests/test_retrieval.py`, `tests/test_api.py` with
      cases from Phases 2-6 not yet covered
- [ ] 9.2 Coverage check — flag any endpoint or branch with no test

**Handoff note:** give Codex CLI the existing code + this task list section
only, not the whole PRD — narrow scope, clear input/output.

---

## Phase 10 — CI/CD
**Tool: Codex CLI**

- [ ] 10.1 GitHub Actions: lint (ruff) → test (pytest) → build image → push
      to GHCR
- [ ] 10.2 Fail the pipeline on lint or test failure, not just report it

---

## Phase 11 — Eval Harness (Retrieval Benchmark)
**Tool: Claude Code CLI**

- [ ] 11.1 Pull sampled Bitext queries (200-300) mapped to your KB doc's
      ground-truth intents, per the earlier eval design
- [ ] 11.2 Compute Recall@k, Precision@k, MRR against retrieval+rerank
- [ ] 11.3 Include out-of-domain Bitext queries (e.g. healthcare/insurance
      intents) to measure escalation false-negative rate
- [ ] 11.4 Output a short benchmark report (the artifact you show in
      interviews — target: ≥40% deflection-equivalent recall, <2%
      false-answer rate, per PRD goals)

**Ties to PRD:** Goals & Success Metrics table — this task is what actually
produces the numbers, not a guess.

---

## Phase 12 — Full Repo Review
**Tool: Gemini CLI**

- [ ] 12.1 Full-codebase pass: does the implementation match prd-full.md
      requirements end-to-end?
- [ ] 12.2 Flag inconsistencies, dead code, or PRD requirements with no
      corresponding code
- [ ] 12.3 Check the open questions still marked unresolved in the PRD
      (ticketing system integration, PII handling, adversarial input) — are
      any accidentally silently handled or silently ignored?

**Handoff note:** give Gemini CLI the whole repo + prd-full.md — this is
the large-context review pass, not incremental building.

---

## Ongoing — Cursor
Not a phase, a mode: use Cursor throughout Phases 1-8 whenever you're
reading what Claude Code generated, want to make a small edit, or want to
understand a piece of code well enough to explain it in an interview. This
is also where bugs the automated tests miss tend to get caught by eye.

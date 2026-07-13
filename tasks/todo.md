# Support Bot - Build Progress

Tracking the phases from the build task list, now saved at `tasks/build-tasks.md`.
Source docs (`prd-full.md`, `support-bot-spec.md`) were not present in the repo.
Folder layout is reconstructed from the module names the task list itself names, per Bianca's call on 2026-07-12.

## The task list itself was recovered

2026-07-12. The 12-phase task list had only ever been pasted into a session, never committed, so it was one `/clear` away from being as lost as the PRD.
Recovered verbatim from the session transcript and committed as `tasks/build-tasks.md`. It is no longer session-only.

Recovering it corrected an error in these notes: Phase 2 is stage-1 vector search and Phase 3 is reranking *and* the confidence gate.
The notes below had those merged, and had the confidence gate under Phase 3 for the right reason but the wrong phase number.

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
That is expected at this stage and is the whole reason Phase 3 reranks: raw cosine similarity over
chunk-level vectors puts near-neighbours (delivery vs shipping, cancelling vs cancellation fees) ahead of
the right document often enough to matter. Recheck these ranks after the reranker lands.

## Phase 2 - Retrieval (stage 1: vector search)

- [x] 2.1 `retrieval.py` - embed the query, vector search Qdrant, return the top ~10 candidates
- [x] 2.2 Unit tests against a small known doc set

Acceptance: met and verified on 2026-07-12. 35 tests pass, ruff clean.

The tests come in two layers. The fast ones inject a fake encoder over a hand-built 3-chunk collection and
pin the mechanics (ordering, the limit, payload mapping, empty query, empty collection) with no model at all.
The slow ones are the acceptance criterion itself and cannot be faked: they run the real MiniLM over the
real 162-chunk corpus in an in-memory Qdrant, through `Retriever.search`, and assert the obviously-correct
document survives into the top 10.

Proved the acceptance tests are load-bearing rather than trusting a green run: capping the search at 1
result instead of 10 fails 2 of the 5 acceptance queries plus 2 mechanics tests.
The 2 that fail are exactly the queries whose document does not rank 1 on cosine similarity, which is the
near-miss problem Phase 3 exists to fix.

The acceptance tests assert recall, not rank, and that is deliberate. Pinning rank here would pin a number
we already know is wrong for 3 of 8 smoke queries. Recall is all stage 1 owes; rank is Phase 3's job.

`scripts/ingest.py --check` now runs through `app.retrieval` rather than its own copy of the vector search.
Its premise was always "the same search the bot will use", and once the bot had a real retrieval module, a
hand-rolled copy would have been checking the wrong thing.

Stage 1 returns raw cosine similarity and nothing thresholds on it. Documented in `retrieval.py` and the
README, because thresholding on it is the exact mistake the Phase 3 confidence gate exists to prevent.

## Phase 3 - Reranking & confidence

- [x] 3.1 `reranker.py` - cross-encoder reranks the top 10, returns the top k (k=4) by rerank score
- [x] 3.2 Confidence gate: escalate when the top rerank score is below `CONFIDENCE_THRESHOLD`
- [x] 3.3 Tests, including the off-topic-but-similar-embedding case the gate exists for

Acceptance: met and verified on 2026-07-12. 65 tests pass, 1 xfail (below), ruff clean.

The required test is `test_rerank_catches_what_similarity_alone_would_answer`, and the comparison is made
fair by construction rather than by picking a flattering threshold. A similarity-only gate does not get to
choose a convenient number: it has to stay lenient enough to answer every genuine query, so the strictest
one it can possibly use is the *lowest* top-1 cosine any in-scope query produces (0.358). "I want to file a
complaint about your service" scores 0.416 on cosine, clearing even that bar, so a similarity gate answers
it - out of `tracking-your-refund`. The reranker scores it 0.000 and escalates.

Proved the gate is load-bearing by scoring it on cosine similarity instead of the reranker: 4 off-topic
queries get answered (complaint -> refund docs, "CEO's salary" -> ordering docs, bitcoin, leave-a-review)
and the acceptance test fails. Reverted.

### Two things the measurements changed

Reranking the *bare chunk text* was wrong. The cross-encoder has to read the chunk under its
`title > heading` prefix, the same context ingestion embeds it under, or it judges a passage whose subject
is missing and marks genuine matches down for it: "where is my order right now" went 0.346 -> 0.552, and
the impostor "can i pay for my order in bitcoin" went 0.476 -> 0.084. `Candidate.passage` now carries it.

`CONFIDENCE_THRESHOLD` of 0.5 was a guess and it was too strict: it escalated 2 of the 8 genuine queries,
which is a 25% hit on the deflection target, silently. Now 0.35, measured - worst genuine query 0.455, best
impostor 0.084. Tuned on 17 queries, which is not many. Phase 11 re-tunes on 200-300.

### Known gaps, deliberately left visible

Rerank ranks the right document 1st for 6 of 8 genuine queries, and puts it in the top-4 for 8 of 8.
The two misses are near-neighbours (card-rejected leads with `accepted-payment-methods` over
`payment-declined-or-failed`; cancelling-cost leads with `cancelling-an-order` over `cancellation-fees`),
and in both the right document is 2nd and still goes to the LLM. The tests assert top-k membership, not
rank 1, because top-k is what the design promises: all 4 chunks are handed to generation and any can be
cited. Phase 11 measures rank properly as MRR, over 200-300 queries rather than 8.

`test_adversarial_mixed_query_escalates` is a strict xfail, not a deleted test.
"Can I track my order if I paid with a stolen card" is half in scope: the reranker answers the tracking half
at 0.75 confidence out of the payments doc, and ignores the fraud. It should escalate to a human.
Phase 12 asks specifically about adversarial input, and a quietly dropped test case is how that question
gets answered wrongly.

## Phase 4 - Answer generation & citations

- [x] 4.1 `llm.py` - Gemini wrapper
- [x] 4.2 Prompt that forces grounding in the retrieved chunks only, no un-cited claims
- [x] 4.3 Response schema carries source citation(s) on every answer
- [x] 4.4 Mocked-LLM tests, so CI never spends Gemini quota

Acceptance: met and verified on 2026-07-12. 81 tests pass, 1 xfail, ruff clean, and the pipeline runs
end to end against real Gemini (`scripts/ask.py --all`): 3 answerable questions come back with citations,
3 unanswerable ones escalate.

Citations are enforced in code, not asked for in the prompt. The model only ever sees the reranked chunks,
and what it returns is validated before it is served: an answer that cites nothing, or that cites a document
it was never given, raises `UngroundedAnswer` and escalates. A fabricated citation is the worst failure this
system can produce, because the citation is the part a customer trusts - it turns a guess into something that
looks sourced. `test_every_served_answer_cites_a_real_source` asserts the invariant over every shape of reply
(valid, uncited, fabricated-source, refusal, non-JSON): either it raises, or it returns an answer whose
citations are non-empty and drawn from the sources actually supplied. There is no third outcome.

### The spec's model no longer exists

`gemini-2.5-flash-lite` is closed to new API keys. It still appears in `models.list()`, which is what makes
this confusing, but calling it returns 404 "no longer available to new users", so the spec's choice is not
buildable as written. Now on `gemini-3.1-flash-lite`, the current model at the same tier, verified working
against Bianca's key. Pinned to an explicit version rather than the `gemini-flash-lite-latest` alias: the
alias moves under you, and a benchmark number in Phase 11 is only worth showing if the model that produced
it can be named.

### A stale .env silently broke the confidence gate

Found because mounting the whole repo into the test container exposed `.env` for the first time, and a
Phase 3 test failed on a change that had nothing to do with Phase 3.

`.env` still pinned `CONFIDENCE_THRESHOLD=0.5` from before Phase 3 measured it at 0.35. The env file wins
over the code default, correctly, so the *running app* was gating at 0.5 and escalating 2 of the 8 genuine
queries - the exact 25% deflection loss Phase 3 thought it had fixed. The measured value now lives in
`config.py` alone, and it is commented out of `.env.example` so it cannot go stale in two places again.

Worse than the stale value: the test suite's verdict depended on an untracked local file. `tests/conftest.py`
now builds `Settings(_env_file=None)`, so the suite asserts against the defaults the repo ships and whatever
is in someone's `.env` stays their business. Verified the fix by putting `CONFIDENCE_THRESHOLD=0.99` in
`.env` and confirming the suite still passes, where before it would have escalated everything.


## Phase 5 - Multi-turn conversation

- [x] 5.1 `conversation.py` - Redis-backed history keyed by `conversation_id`, TTL from settings
- [x] 5.2 Query condensation: rewrite follow-up + history into a standalone query before retrieval
- [x] 5.3 Append each turn (question + answer, or the escalation) to history
- [x] 5.4 Tests: a two-turn conversation where turn 2 is only answerable if turn 1 was used

Acceptance: met and verified on 2026-07-12. 97 tests pass, 1 xfail, ruff clean, and the two-turn
conversation runs end to end against real Redis and real Gemini (`scripts/ask.py --chat`).

The task list said to prove the turn-2 test fails when condensation is disabled, rather than settle for a
test that only proves the endpoint returns 200. Did that: made `Condenser.condense` return the query
unchanged, and `test_turn_two_needs_turn_one` fails with
"condensed to 'how long will it take' and still did not reach the refund docs - got registration-problems".
Reverted. The test runs through the real `Condenser` and the real history rather than comparing two
hardcoded strings, which is what makes it sensitive to the feature actually being wired in.

### Condensation is a safety feature, not a quality one

This is the thing worth saying out loud in an interview. Retrieval is stateless: it embeds the string it is
handed. So the follow-up "how long will it take", asked after "i want a refund for a damaged item", retrieves
chunks about *account registration*, scores 0.373, clears the 0.35 confidence gate, and confidently answers
how long registration takes. The customer asked about a refund.

Condensed against the history, the same follow-up becomes "How long will it take to receive a refund for a
damaged item?", retrieves the refund documents, and scores 0.995.

So condensation is not there to make follow-ups nicer. It is there because a vague follow-up otherwise
becomes a *confident wrong answer* - the exact failure the Phase 3 gate exists to prevent, arriving through a
door the gate cannot see. The gate scores how well the query matches a document; it has no way of knowing the
query was the wrong query.

### Smaller decisions

History is a Redis list (`RPUSH` + `LRANGE`), not a JSON blob rewritten on every turn: two turns arriving at
once should both survive, and a read-modify-write would lose one.

The TTL is reset on every turn, so it measures silence rather than age. An hour into a live conversation is
not the moment to forget it. An expiring key is also the cheapest privacy story available - nothing has to
remember to delete anything.

Escalated turns are kept in the history and shown to the condenser as "(escalated to a human agent)".
The customer still said the thing, and the next turn's pronouns may point at it.

A failed condensation (bad JSON, empty rewrite) logs a warning and falls back to the raw follow-up. That is
degraded, not broken - it is exactly the behaviour we had before this feature - and failing the customer's
question outright because a rewrite would not parse is the worse trade.

Tests use `fakeredis`, so they exercise the real client API (RPUSH ordering, TTL semantics) without needing
the compose stack, which is the same bargain the in-memory Qdrant already makes.

## Phase 6 - Escalation handoff payload

- [x] 6.1 Escalation carries the original query, the conversation history, the top retrieved chunks
      (including the ones below threshold) and the confidence score
- [x] 6.2 Decided: escalation is **sticky and permanent**. Documented in the README, per the task list.

Acceptance: met and verified on 2026-07-12. 108 tests pass, 1 xfail, ruff clean, and both behaviours are
visible end to end (`scripts/ask.py --sticky`): turn 1 escalates on low confidence, and turn 2
("where is my order right now") escalates as `human_owned` at confidence **0.469** - comfortably above the
0.35 gate, so the bot demonstrably could have answered it and deliberately did not.

### 6.2, the PRD's open question

Bianca chose sticky-and-permanent from the three options. Once a conversation goes to a human it stays with
the human, even for a turn the bot is confident about.

The reasoning that decided it: re-gating each turn lets the bot start answering again while an agent is
mid-reply, and a customer getting two voices in one thread is a worse failure than a human spending ten
seconds on "thanks!".

The cost is real and was accepted rather than hidden: trivial follow-ups after an escalation do burn agent
time, and there is no route back to the bot inside the same conversation - it takes a new `conversation_id`.
If that cost ever bites, the fix is a handback (an agent explicitly releasing the conversation), not
re-gating every turn.

### Answered and Escalated are different types

Not one type with an `escalated=True` flag. An `Answered` cannot reach an agent's queue and an `Escalated`
cannot reach a customer, and the type system enforces what a boolean would merely record. That is also what
task 6.1 means by "structurally distinct".

The handoff carries the chunks that *lost* - the ones below the confidence threshold. They are the evidence,
and withholding them would hide the reason the turn escalated: an agent who can see the bot nearly matched
the refund policy knows something quite different from one who sees that it matched nothing at all.

A human-owned turn still runs retrieval, even though the bot is not allowed to answer it. The agent should
not have to search the knowledge base by hand. It skips the LLM call, though - there is no answer to write,
so there is no reason to spend the quota finding that out.

### `app/bot.py` now owns the pipeline

`scripts/ask.py` had grown its own copy of condense -> retrieve -> rerank -> gate -> answer, which would have
drifted from whatever the Phase 7 endpoint did. `SupportBot.handle()` is now the single path, and the CLI,
the coming API endpoint and the Phase 11 eval all call it.

## Phase 7 - Rate limiting

- [x] 7.0 `POST /chat` (added; see below - the task list never numbered it)
- [x] 7.1 Redis-backed rate limiter, per API key or per IP
- [x] 7.2 Tuned against Gemini's actual limits rather than an assumed number
- [x] 7.3 Graceful 429 with retry-after guidance

Acceptance: met and verified on 2026-07-12. 128 tests pass, 1 xfail, ruff clean, and the limit fires over
real HTTP against the running container: requests return 200 until the budget is spent, then a clean
`429` with `Retry-After: 42`, never a crash or a hang.

### The task list forgot the endpoint

There is no numbered task anywhere in the 12 phases that builds the chat endpoint, but Phases 5, 6 and 7 all
assume it - you cannot rate-limit an API that does not exist. Built `POST /chat` here as 7.0 and flagged it
rather than quietly folding it into 7.1. It is very likely the missing `support-bot-spec.md` had it.

### A per-IP limit does not protect the Gemini quota

The thing worth saying in an interview. These protect different things and conflating them is the actual
failure mode:

- The **per-caller** limit (10/min, 200/day) is about fairness and abuse.
- The **global upstream budget** is about not spending a quota we do not have. The Gemini free tier is
  *project-wide*, so a hundred callers each comfortably inside their own limit still exhaust it between them.
  Without the global limiter, the next request reaches Gemini, gets a 429 from Google, and turns into a 500
  for a customer who did nothing wrong. `test_the_shared_upstream_budget_stops_polite_callers_from_exhausting_gemini`
  is that scenario, written down.

The upstream budget is sized in **turns, not requests**, because one turn can cost two Gemini calls: condense
the follow-up, then write the answer. Sizing a 15 RPM budget as 15 turns would overspend it 2x on any
conversation past the first turn. 15 // 2 = 7 turns/minute, and that is exactly what bit in the live test.

### 7.2, honestly

The task list says to check the real limits rather than hardcode an assumed number. Checked: **Google no
longer publishes a fixed free-tier RPM/RPD per model.** The docs say limits depend on the account and are
shown live in AI Studio, and third-party trackers contradict each other (1,000 vs 1,500 RPD). 15 RPM is the
one figure they agree on for flash-lite.

So the honest implementation is not a hardcoded number at all: `GEMINI_RPM` / `GEMINI_RPD` are settings with
conservative defaults (15 / 1000), documented as needing to be set from the real value for the key in use.
Bianca should confirm hers at https://aistudio.google.com/rate-limit.

### Two 429s, for two different reasons

Being over *our* limit and Gemini being over *its* limit are different events, and both must be a clean 429
with `Retry-After` rather than a 500. The second one is the sneaky one: our budget exists to prevent it, but
the quota is shared with anything else using the same key, so it can still happen. `UpstreamRateLimited`
carries it out of `llm.py` and the endpoint turns it into "the assistant is briefly over its usage limit",
not a stack trace.

A rate-limited turn never reaches the bot: the refusal has to be cheaper than the work it prevents, which is
the entire point of having one.

The limiter fails **open**, not closed. A counter that lost its TTL (a crash between INCR and EXPIRE) would
otherwise count up forever and ban a caller permanently. It restores the expiry instead. Letting one extra
request through is a far better failure than a silent permanent ban.

`X-Forwarded-For` is deliberately not trusted: it is caller-controlled, so honouring it would let anyone
reset their own limit by inventing a header. Behind a real proxy this needs the proxy's address and an
explicit trusted-hop config, which is a deployment decision rather than a default.

## Phase 8 - Observability

- [x] 8.1 Structured logging: request id, conversation id, latency, confidence, escalated true/false
- [x] 8.2 Admin endpoint: deflection rate, escalation rate, and rates by category

Acceptance: met and verified on 2026-07-12. 144 tests pass, 1 xfail, ruff clean, and both are real against
the running container: driving 3 turns produced `deflection_rate: 0.6667`, `mean_latency_ms: 898.2`, and one
`low_confidence` escalation, and **17/17 log lines parse as JSON**.

### False-answer rate is null, on purpose, and says why

The PRD asks for false-answer rate by category. It is **not computable from live traffic**. A false answer is
one the bot gave confidently and *wrongly*, and nothing in a request tells us it was wrong - that needs ground
truth or a human saying so. The endpoint returns `null` plus a note pointing at Phase 11, which has labelled
queries and is where the number actually comes from.

Reporting a plausible number here, or silently redefining false-answer rate as something cheaper to measure
(escalation rate, say), would be worse than reporting nothing, because someone would put it in a slide and
believe it.

Same reasoning, smaller stakes: rates are `null` rather than `0.0` when no traffic has arrived. A 0% deflection
rate is an emergency and no traffic is a Tuesday, and a dashboard that renders them identically will have
someone debugging a bot that is fine.

### Escalated turns record their category

Otherwise the single most useful thing the endpoint could tell you - "we keep escalating REFUND questions" - is
exactly the thing it cannot. The category comes from the best chunk even when the turn escalated.

Rate-limited requests are *not* counted as turns. They never reached the bot, so counting them would inflate
the escalation rate with traffic the bot never saw, and make a capacity problem look like a product problem.

### Two defects the end-to-end run caught

Both invisible to the test suite, both found by actually reading the container's log stream:

- **Uvicorn's own access lines were plain text** in an otherwise-JSON stream, so an aggregator could parse
  neither half. Its loggers now propagate through the JSON formatter. `configure_logging` also had to move to
  import time: uvicorn logs "Started server process" before the lifespan runs, and those lines came out as
  prose.
- **sentence-transformers printed a tqdm progress bar on every single query.** `show_progress_bar=False` was set
  in ingestion but not in the query encoder, so every retrieval dumped `Batches: 100%|...` into the logs.

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
- The model is `gemini-3.1-flash-lite`, not the `gemini-2.5-flash-lite` the task list named.
  That model is closed to new API keys and returns 404 on every call, so the spec is not buildable as
  written. Same tier, current, and pinned to an explicit version rather than a `-latest` alias.
- Knowledge base corpus lives in `kb/`, not `docs/`, so it does not collide with project documentation.
- `POST /chat` was built in Phase 7 as task 7.0. No numbered task in the task list ever creates it, yet
  Phases 5-7 all depend on it existing. Presumably it lived in the missing `support-bot-spec.md`.

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

Then open `localhost:8000` for the chat page.
It is one static file (`web/index.html`), served by the app itself, and it is a demo front end for the API rather than a second product.
It renders an escalation as its own shape - no answer, and instead the documents that lost and the scores they lost with - because what this bot declines to answer is the interesting half of it.

## Deploy the demo

The public demo runs on Cloud Run, and `scripts/deploy_cloudrun.sh` builds and ships it in one command.
Once: a Google Cloud project with billing enabled, then `gcloud auth login` and `gcloud config set project <id>`.
The free tier covers a demo comfortably, but the project needs a card on file to exist at all.

```sh
GEMINI_API_KEY=... ADMIN_API_KEY=... scripts/deploy_cloudrun.sh
```

The demo runs the whole stack in **one** container (`deploy/single/Dockerfile`), with Qdrant and Redis inside it - the Qdrant binary is copied out of the same image tag compose pins, so the demo and the compose stack run the same server.
`docker-compose.yml` remains the real topology and the one local dev runs against, and the two are worth testing separately: a bug that lives in the gap between them (a missing shared library, say) is invisible to `docker compose up`.

One container because the hosts that will run one for free will run exactly one.
It is not tied to Cloud Run: it listens on `$PORT`, keeps nothing on disk, and needs nothing but two secrets in the environment.
Cloud Run is where it runs today because its free tier scales to zero, which means a demo nobody is looking at costs nothing.
(It was going to be a Hugging Face Space, until July 2026, when HF quietly made Docker Spaces on free hardware a PRO feature. Static Spaces are still free, and cannot run any of this.)

Two things about the container are deliberate.
The embedding and reranker weights are baked in at **build** time, because the service scales to zero and cold-starts on the next visit, and the visitor who arrives during that cold start is exactly the person the demo exists for.
The collection is rebuilt from `kb/` at **start** time, because there is no persistent disk - it costs a few seconds on a corpus this size and it means there is no stale-index failure mode.

Two secrets, passed to the deploy script and set as environment variables on the service. Neither is in the repo, and the script cannot ship them by accident: it builds from `git archive HEAD`, which is tracked files only.

- `GEMINI_API_KEY`, from [AI Studio](https://aistudio.google.com/apikey).
- `ADMIN_API_KEY`, which nobody issues: it is a secret you invent, checked against the `X-Admin-Key` header, and its only property is being unguessable. Generate one with `openssl rand -hex 32`. `/admin/metrics` is **open when it is unset**, which is fine locally and is not fine on the internet: it exposes no message content, but "how often does this bot fail" is not a number to hand out.

Locally the same two live in `.env` (gitignored, see `.env.example`).

Three known caveats of a public deployment, all of which follow from things `docs/open-questions.md` already says out loud.
Rate limiting buckets on the peer IP, and behind Cloud Run's front end every request arrives from the proxy, so all visitors share one bucket - the shared upstream Gemini budget is what actually protects the quota, and it is sized for that.
Conversation history lives in the container's own Redis, which is why the service is pinned to `--max-instances 1`: a second instance would answer follow-ups with no memory of the conversation.
And customer text reaches Gemini and Redis un-redacted, so the page says so and asks people not to paste anything real.
Do not set `LOG_LEVEL=DEBUG` there: DEBUG logs the query text.

## The API

```sh
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "where is my order right now"}'
```

An answer and an escalation are different response shapes, discriminated by `status`, because they are different events.
A client that forgets to check a boolean flag would happily render an escalation's empty answer field; a client that forgets to check `status` here has nothing to render at all, which is the failure we want.

```jsonc
// answered
{"status": "answered", "conversation_id": "...", "confidence": 0.55,
 "answer": "You can track your order by...", "citations": ["tracking-your-order"]}

// escalated - no answer field exists to render
{"status": "escalated", "conversation_id": "...", "confidence": 0.03,
 "reason": "low_confidence", "history": [...],
 "retrieved": [{"doc_id": "placing-an-order", "title": "Placing an order", "score": 0.03}]}
```

Pass a `conversation_id` back to make the next turn a follow-up. Omit it and a new conversation is minted.

The escalation carries the retrieved chunks and their scores - the ones that scored *below* the gate. In production that payload belongs in the ticketing system rather than in a customer's browser, but it is returned here because the interesting thing about this bot is what it declines to answer and why, and hiding the evidence would hide the product.

## Observability

Logs are JSON, one object per line, all of them - uvicorn's access lines are routed through the same formatter, because a stream that is half JSON and half prose parses as neither.

Every turn logs one line carrying the request id, conversation id, latency, confidence, category, and whether it escalated:

```json
{"ts": "2026-07-13T01:48:37+0000", "level": "INFO", "request_id": "5d52f722bbb4",
 "message": "turn escalated", "conversation_id": "d5b77cdf...", "escalated": true,
 "reason": "low_confidence", "confidence": 0.0276, "category": "ORDER", "latency_ms": 379.1}
```

The request id also comes back in the `X-Request-ID` response header, so a customer complaint ("it told me refunds take 30 days") traces to the exact turn, and from there to the exact chunks that were retrieved.

```sh
curl -s localhost:8000/admin/metrics    # deflection + escalation rates, by category
```

Set `ADMIN_API_KEY` to require an `X-Admin-Key` header on that endpoint. It exposes no message content, only counts - but "how often does this bot fail" is not a number to leave open to the internet.

**On false-answer rate.** The PRD asks for it and the endpoint returns `null`, on purpose. A false answer is one the bot gave confidently and *wrongly*, and nothing in a request says it was wrong - that needs ground truth or a human saying so. It is produced by the Phase 11 eval against labelled queries, not by live traffic. Reporting a plausible-looking number here, or quietly redefining it as something cheaper to measure, would be worse than reporting nothing: someone would put it in a slide.

Rates are `null` rather than `0` when nothing has happened yet, for the same reason. A 0% deflection rate is an emergency; no traffic is a Tuesday.

## Rate limiting

Two axes, protecting two different things. Conflating them is how a free tier gets exhausted by callers who were each individually well behaved.

- **Per caller** (the peer IP): 10/minute, 200/day. This is about fairness and abuse.
- **A global upstream budget**, per minute *and* per day: this is about not spending a Gemini quota we do not have. The free tier is *project-wide*, so a hundred callers each politely under their own limit will still exhaust it between them - and the hundred-and-first gets a 500 from an upstream 429 nobody was watching for.

The upstream budget is sized in **turns, not requests**, because one turn can cost two Gemini calls (condensing the follow-up, then writing the answer). Sizing a 15 RPM budget as 15 turns would overspend it by 2x on any conversation past its first turn. So the default works out at 7 turns/minute and 500 turns/day.

Both axes return a clean `429` with a `Retry-After` header and the same figure in the body, and a limited turn never reaches the bot - the refusal has to be cheaper than the work it prevents.

**A caller is their IP address, and nothing else.**
`X-Forwarded-For` is not trusted, and neither is `X-API-Key`: the code used to bucket the limit on whatever `X-API-Key` string arrived, and nothing anywhere validated that string, so any caller could hit their limit and then mint a fresh allowance by typing a different one.
Phase 12 found it, and a key nobody checks is not an identity.
An API key can come back the day something issues and verifies them, bucketing on the identity the key *resolves to* rather than on the key itself.

Note for deploying this behind a proxy: every request will arrive from the proxy's address, so all callers share one bucket.
That needs the forwarded address plus an explicit trusted-hop config, which is a deployment decision rather than a default.

`GEMINI_RPM` and `GEMINI_RPD` default to 15 and 1000, which is the conservative reading of a number Google no longer publishes per model: the docs say limits depend on the account and are shown live in [AI Studio](https://aistudio.google.com/rate-limit), and third-party trackers disagree (1,000 vs 1,500 RPD). **Check the real number for your key and set it** rather than trusting the default.

## Verify

```sh
# the whole suite: chunking, retrieval mechanics, and the retrieval acceptance
# tests (those load the real model and embed the corpus - the `slow` marker)
docker compose run --rm -v "$PWD/tests:/srv/tests" app python -m pytest tests -q

# just the fast ones, no model weights
docker compose run --rm -v "$PWD/tests:/srv/tests" app python -m pytest tests -q -m "not slow"

# the KB corpus matches the Bitext taxonomy
docker compose exec app python scripts/check_kb_coverage.py

# end to end: ingest, then put real user phrasings through vector search
# and assert the right document comes back
docker compose exec app python scripts/ingest.py --check

# what would be ingested, without writing to Qdrant
docker compose exec app python scripts/ingest.py --dry-run

# the whole pipeline as a customer meets it: retrieve, rerank, gate, answer or
# escalate. Makes a real Gemini call, so it needs GEMINI_API_KEY in .env
docker compose exec app python scripts/ask.py "where is my order right now"
docker compose exec app python scripts/ask.py --all

# a two-turn conversation, where turn 2 ("how long will it take") only makes
# sense because turn 1 happened
docker compose exec app python scripts/ask.py --chat

# an escalated conversation stays escalated: turn 2 is a question the bot
# answers happily on its own, and does not answer here
docker compose exec app python scripts/ask.py --sticky

# the benchmark: 320 Bitext queries through retrieval, rerank and the gate.
# Costs no Gemini quota - it measures everything decided before the LLM is reached
docker compose run --rm -v "$PWD/scripts:/srv/scripts" -v "$PWD/eval:/srv/eval" \
  -v "$PWD/docs:/srv/docs" app python scripts/eval.py
```

`scripts/ask.py --all` is the demo. Three questions the KB answers and three it does not; the first three come back with citations, the last three escalate.

`--chat` is the multi-turn demo, and it shows the condensed query it actually retrieved on.

`--check` is the one that matters. An ingest can report success while retrieving the wrong document for every query, and only `--check` catches that.

## Repo map

- `app/` - FastAPI application. `bot.py` (one turn, end to end - the pipeline everything else calls), `api.py` (`POST /chat`, `GET /admin/metrics`), `ratelimit.py` (per-caller limits + the shared upstream budget), `observability.py` (JSON logging + the metric counters), `config.py` (env settings), `ingestion.py` (chunk, embed, upsert), `retrieval.py` (stage 1 vector search), `reranker.py` (stage 2 cross-encoder rerank + confidence gate), `llm.py` (grounded answer generation with citations), `conversation.py` (Redis history + query condensation), `main.py` (entrypoint, `/health`).
- `kb/` - the knowledge base: 25 help-centre documents, each mapped in frontmatter to the Bitext intents it answers.
- `scripts/` - `ingest.py` (re-ingestion CLI), `check_kb_coverage.py` (corpus vs taxonomy), `ask.py` (the pipeline as a customer meets it), `sample_eval_set.py` (draw the eval set from Bitext), `eval.py` (the benchmark).
- `eval/` - `queries.jsonl`, the frozen 320-query eval set. Committed on purpose: a benchmark that resamples every run measures something different every run.
- `docs/` - project documentation: the derived intent taxonomy, `benchmark.md` (generated - re-run `scripts/eval.py`, do not hand-edit), and `open-questions.md` (what this does and does not do about ticketing, PII and adversarial input - read it before deploying anywhere public).
- `web/` - `index.html`, the chat page. One file, no build step, served by the app at `/`.
- `deploy/single/` - the whole stack in one container, for the public demo. Not the real topology; compose is.
- `tasks/` - build progress and running notes.

## Stack

Python 3.12, FastAPI, Qdrant, Redis, sentence-transformers (`all-MiniLM-L6-v2` embedding, `ms-marco-MiniLM-L-6-v2` reranker), Gemini `3.1-flash-lite` via the `google-genai` SDK.

The spec asked for `gemini-2.5-flash-lite`, which Google has since closed to new API keys - it still shows up in `models.list()` but calling it returns a 404, so the spec's choice is not buildable as written. `3.1-flash-lite` is the current model at the same tier. It is pinned to an explicit version rather than the `-latest` alias, because an alias that moves under you makes Phase 11's benchmark numbers unreproducible.

## Decisions

Read `docs/intent-taxonomy.md` before touching the KB or the eval. The taxonomy is derived from the Bitext dataset and is ground truth for both.

The embedding model reads at most 256 tokens and silently truncates beyond that, so `chunk_size_tokens` is capped well under it. Ingestion refuses to run if the budget is set too high rather than let the tail of a chunk be stored but never embedded.

Retrieval is two stages, and only the second one is allowed to judge.
Stage 1 (`retrieval.py`) casts a wide net and returns raw cosine similarity, which is a similarity signal and not a confidence signal - an off-topic query can sit at high similarity to a document it has nothing to do with.
Nothing thresholds on it. The confidence gate is scored on the cross-encoder in stage 2 (`reranker.py`), which is the only thing in the pipeline that reads the question and the passage together.

This is measurable, not theoretical. Gate on cosine similarity instead and the bot answers "I want to file a complaint about your service" out of the refund docs, and "what is the CEO's salary" out of the ordering docs. Gate on the reranker and both escalate. `tests/test_reranker.py` asserts exactly that, against a similarity threshold chosen to be as strict as it possibly can be while still answering every genuine query.

`CONFIDENCE_THRESHOLD` is 0.2, set by the eval over 320 real Bitext queries (`docs/benchmark.md`).
It hits both PRD targets with margin on each: **42.3% deflection** against a 40% target, **1.3% false answers** against a 2% cap.
Not the 0.1 that maximises deflection at 45.9% - that sits at 1.9% against a 2.0% cap, which is not a margin, it is a coincidence.

**The threshold was never the lever, and this is the thing worth knowing about the system.**
On the original corpus no threshold hit both targets at once: it was 25% deflection at 1.9% false answers, or 43% deflection at 5%.
The sweep only moves along one axis, buying deflection with false answers or the reverse.

What moved the whole curve was fixing what the knowledge base *says*.
The eval named three intents failing on coverage rather than ranking, and the gaps turned out to be vocabulary: the delivery-times doc had headings like "Cut-off times" and "Working days" and never answered "how long until my parcel arrives"; the delivery-options doc said "delivery" where every customer says "shipping"; the refund policy never answered "in which cases can I ask for a refund" as a question.
Chunks are embedded under their `title > heading`, so the heading vocabulary is half the retrieval signal.

Rewriting those three documents took deflection from 25.0% to 42.3% *and* false answers from 1.9% to 1.3%.
That is not a trade, and no amount of tuning could have produced it.

The honest caveat, from the same eval: **the cross-encoder is a good ranker and a poorly calibrated confidence signal.**
Sigmoid-squashing a raw logit does not make it a probability, and the threshold is being read as one.
Calibrating it properly (Platt scaling, on the labelled eval set that now exists) is the next thing worth doing.

**Escalation is sticky and permanent.** Once a conversation goes to a human it stays with the human: every later turn escalates too, even one the bot is confident it could answer.

This was an open question in the PRD, and the alternative - re-running the confidence gate on every turn - means the bot can start answering again while an agent is mid-reply.
A customer getting two voices in one thread is a worse failure than a human spending ten seconds on "thanks!".

The cost is real and accepted: trivial follow-ups after an escalation do burn agent time, and there is no way back to the bot inside the same conversation - it takes a new `conversation_id`.
If that cost ever bites, the fix is a handback (an agent explicitly releasing the conversation), not re-gating each turn.

An escalation is a different *type* from an answer, not an answer with a flag set.
It carries what the customer typed, what the query was condensed to, the conversation so far, the confidence score, and the retrieved chunks *including the ones that scored below the threshold* - those are the evidence, and they are the reason it escalated.
An agent who can see the bot nearly matched the refund policy knows something quite different from one who sees it matched nothing at all.

Follow-ups are condensed into a standalone question before retrieval, and this is not a nicety.
Retrieval is stateless - it embeds the string it is handed - so "how long will it take" retrieves chunks about account registration, clears the confidence gate at 0.373, and confidently answers a question the customer never asked.
Rewriting it against the history ("how long will it take to receive a refund for a damaged item?") sends it to the refund documents at 0.995.
Condensation is what stops a vague follow-up becoming a confident wrong answer, and `tests/test_conversation.py` fails if it is removed.

Every answer carries a citation, and that is enforced in code rather than requested in the prompt.
The model is shown only the reranked chunks, and what it returns is checked before it is served: an answer that cites nothing, or that cites a document it was never given, is refused and escalates.
A fabricated citation is the worst failure available to this system - the citation is the part a customer trusts, so an invented one launders a guess into something that looks sourced - and `tests/test_llm.py` asserts the invariant structurally over every shape of model reply, not on a happy-path example.

The Qdrant image and `qdrant-client` are pinned to the same minor version. They do not tolerate drift: the client warns on every call, and the on-disk format does not survive a wide version jump.

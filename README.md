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

## Rate limiting

Two axes, protecting two different things. Conflating them is how a free tier gets exhausted by callers who were each individually well behaved.

- **Per caller** (API key if present, otherwise peer IP): 10/minute, 200/day. This is about fairness and abuse.
- **A global upstream budget**: this is about not spending a Gemini quota we do not have. The free tier is *project-wide*, so a hundred callers each politely under their own limit will still exhaust it between them - and the hundred-and-first gets a 500 from an upstream 429 nobody was watching for.

The upstream budget is sized in **turns, not requests**, because one turn can cost two Gemini calls (condensing the follow-up, then writing the answer). Sizing a 15 RPM budget as 15 turns would overspend it by 2x on any conversation past its first turn. So the default works out at 7 turns/minute.

Both limits return a clean `429` with a `Retry-After` header and the same figure in the body, and a limited turn never reaches the bot - the refusal has to be cheaper than the work it prevents.

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
```

`scripts/ask.py --all` is the demo. Three questions the KB answers and three it does not; the first three come back with citations, the last three escalate.

`--chat` is the multi-turn demo, and it shows the condensed query it actually retrieved on.

`--check` is the one that matters. An ingest can report success while retrieving the wrong document for every query, and only `--check` catches that.

## Repo map

- `app/` - FastAPI application. `bot.py` (one turn, end to end - the pipeline everything else calls), `api.py` (`POST /chat`), `ratelimit.py` (per-caller limits + the shared upstream budget), `config.py` (env settings), `ingestion.py` (chunk, embed, upsert), `retrieval.py` (stage 1 vector search), `reranker.py` (stage 2 cross-encoder rerank + confidence gate), `llm.py` (grounded answer generation with citations), `conversation.py` (Redis history + query condensation), `main.py` (entrypoint, `/health`).
- `kb/` - the knowledge base: 25 help-centre documents, each mapped in frontmatter to the Bitext intents it answers.
- `scripts/` - `ingest.py` (re-ingestion CLI), `check_kb_coverage.py` (corpus vs taxonomy).
- `docs/` - project documentation, including the derived intent taxonomy.
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

`CONFIDENCE_THRESHOLD` is 0.35, and the number is measured. The worst genuine query scores 0.455 and the best impostor 0.084, so 0.35 sits in the gap with room either side. It is tuned on 17 queries, which is not many - Phase 11 re-tunes it on 200-300.

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

# Learnings

Notes on why this bot is built the way it is.
Most of these are things I got wrong first, or things the eval told me that I did not expect.

## The knowledge base moved the numbers, not the threshold

This is the one worth knowing.

`CONFIDENCE_THRESHOLD` is 0.2, set by the eval over 320 real Bitext queries (`benchmark.md`).
It hits both PRD targets with margin on each: 42.3% deflection against a 40% target, 1.3% false answers against a 2% cap.
Not the 0.1 that maximises deflection at 45.9%, because that sits at 1.9% against a 2.0% cap, which is not a margin, it is a coincidence.

But on the original corpus, no threshold hit both targets at once.
It was 25% deflection at 1.9% false answers, or 43% deflection at 5%.
The sweep only moves along one axis, buying deflection with false answers or the reverse.

What moved the whole curve was fixing what the knowledge base *says*.
The eval named three intents failing on coverage rather than ranking, and the gaps turned out to be vocabulary.
The delivery-times doc had headings like "Cut-off times" and "Working days" and never answered "how long until my parcel arrives".
The delivery-options doc said "delivery" where every customer says "shipping".
The refund policy never answered "in which cases can I ask for a refund" as a question.
Chunks are embedded under their `title > heading`, so the heading vocabulary is half the retrieval signal.

Rewriting those three documents took deflection from 25.0% to 42.3% *and* false answers from 1.9% to 1.3%.
That is not a trade, and no amount of tuning could have produced it.

The honest caveat, from the same eval: the cross-encoder is a good ranker and a poorly calibrated confidence signal.
Sigmoid-squashing a raw logit does not make it a probability, and the threshold is being read as one.
Calibrating it properly (Platt scaling, on the labelled eval set that now exists) is the next thing worth doing.

## The two worst intents are the eval's problem, not the KB's

The by-intent table names `place_order` and `check_cancellation_fee` as the weakest two, and the obvious next move is to rewrite them the way the three delivery and refund docs were rewritten.
Tracing every one of their twenty queries through the pipeline says not to.

`check_cancellation_fee` is the clear case.
The one query phrased in the KB's own terms, "i try to check the cancellation fee", scores 0.99 with all four chunks from `cancellation-fees`.
The other nine ask about "early exit fees", "early termination fees" and "withdrawal penalties", which is subscription and contract-exit language from Bitext's telecom origin.
Northwind is a wholesale food supplier, and its cancellation fee is a restocking charge on an order that has started picking, not an early-termination penalty on a contract.
The cross-encoder scores those nine near zero even when it has retrieved `cancellation-fees` at rank 1, and they escalate.
That is the right answer.
Adding "early termination" vocabulary to the document to lift the scores would make the bot confidently answer questions Northwind cannot answer, which is a worse failure than sending them to a human.

`place_order` is retrieved correctly for eight of ten, but every document scores near zero because the queries are informal and misspelled, like "can uhelp me buying an artikcle" and "where can i eanr a few of ur item".
Low scores escalate, and nothing here clears the 0.2 gate, so no document edit changes the deflection number.
An additive edit could nudge the ranking, but a rank change on ten queries is inside the noise, and half the miss is genuinely garbled input a human should see anyway.

The lesson is that a low MRR on ten queries is not automatically a coverage gap.
The delivery and refund intents were fixable because retrieval was missing the document; these two are not, because retrieval finds the document and the gate is right to distrust the match.

## One negative in a hundred clears the gate, and it is a near-collision

The eval leaks one out-of-domain query: "how do I reject offers?", a Bitext insurance settlement-rejection intent.
It scores 0.35 against `cancelling-an-order`, because that document says "Tell the driver you are rejecting the consignment", and "reject offers" lands close to "reject a delivery".
It is inside the target and left as a known near-collision rather than chased.
Tightening the gate enough to catch it would cost deflection on the genuine cancel-order queries that sit just above it, and one semantic collision at 0.35 is a cheaper thing to write down than to design around.

## Only the reranker is allowed to judge

Retrieval is two stages.
Stage 1 (`retrieval.py`) casts a wide net and returns raw cosine similarity, which is a similarity signal and not a confidence signal.
An off-topic query can sit at high similarity to a document it has nothing to do with.
Nothing thresholds on it.
The confidence gate is scored on the cross-encoder in stage 2 (`reranker.py`), which is the only thing in the pipeline that reads the question and the passage together.

This is measurable, not theoretical.
Gate on cosine similarity instead and the bot answers "I want to file a complaint about your service" out of the refund docs, and "what is the CEO's salary" out of the ordering docs.
Gate on the reranker and both escalate.
`tests/test_reranker.py` asserts exactly that, against a similarity threshold chosen to be as strict as it possibly can be while still answering every genuine query.

## Escalation is a type, not a flag

An answer and an escalation are different response shapes, discriminated by `status`, because they are different events.
A client that forgets to check a boolean flag would happily render an escalation's empty answer field.
A client that forgets to check `status` here has nothing to render at all, which is the failure I want.

The escalation carries what the customer typed, what the query was condensed to, the conversation so far, the confidence score, and the retrieved chunks *including the ones that scored below the gate*.
Those are the evidence, and they are the reason it escalated.
An agent who can see the bot nearly matched the refund policy knows something quite different from one who sees it matched nothing at all.

In production that payload belongs in the ticketing system rather than in a customer's browser.
It is returned here because the interesting thing about this bot is what it declines to answer and why, and hiding the evidence would hide the product.
The chat page renders an escalation in its own shape for the same reason: no answer, and instead the documents that lost and the scores they lost with.

## Escalation is sticky, and that cost is accepted

Once a conversation goes to a human it stays with the human.
Every later turn escalates too, even one the bot is confident it could answer.

This was an open question in the PRD.
The alternative, re-running the confidence gate on every turn, means the bot can start answering again while an agent is mid-reply.
A customer getting two voices in one thread is a worse failure than a human spending ten seconds on "thanks!".

The cost is real.
Trivial follow-ups after an escalation do burn agent time, and there is no way back to the bot inside the same conversation, it takes a new `conversation_id`.
If that ever bites, the fix is a handback (an agent explicitly releasing the conversation), not re-gating each turn.

## Condensation is what stops a vague follow-up becoming a confident wrong answer

Follow-ups are condensed into a standalone question before retrieval, and this is not a nicety.

Retrieval is stateless.
It embeds the string it is handed, so "how long will it take" retrieves chunks about account registration, clears the confidence gate at 0.373, and confidently answers a question the customer never asked.
Rewriting it against the history ("how long will it take to receive a refund for a damaged item?") sends it to the refund documents at 0.995.

`tests/test_conversation.py` fails if it is removed.

## Citations are enforced in code, not requested in the prompt

The model is shown only the reranked chunks, and what it returns is checked before it is served.
An answer that cites nothing, or that cites a document it was never given, is refused and escalates.

A fabricated citation is the worst failure available to this system.
The citation is the part a customer trusts, so an invented one launders a guess into something that looks sourced.
`tests/test_llm.py` asserts the invariant structurally over every shape of model reply, not on a happy-path example.

## Rate limiting is two axes, because it protects two things

Conflating them is how a free tier gets exhausted by callers who were each individually well behaved.

Per caller (the peer IP) is 10/minute and 200/day, and that is about fairness and abuse.
The global upstream budget, per minute and per day, is about not spending a Gemini quota I do not have.
The free tier is *project-wide*, so a hundred callers each politely under their own limit will still exhaust it between them, and the hundred-and-first gets a 500 from an upstream 429 nobody was watching for.

The upstream budget is sized in turns, not requests, because one turn can cost two Gemini calls: condensing the follow-up, then writing the answer.
Sizing a 15 RPM budget as 15 turns would overspend it by 2x on any conversation past its first turn.
So the default works out at 7 turns/minute and 500 turns/day.

A limited turn never reaches the bot.
The refusal has to be cheaper than the work it prevents.

### A caller is their IP address, and nothing else

`X-Forwarded-For` is not trusted, and neither is `X-API-Key`.
The code used to bucket the limit on whatever `X-API-Key` string arrived, and nothing anywhere validated that string, so any caller could hit their limit and then mint a fresh allowance by typing a different one.
Phase 12 found it.
A key nobody checks is not an identity.

An API key can come back the day something issues and verifies them, bucketing on the identity the key *resolves to* rather than on the key itself.

Behind a proxy, every request arrives from the proxy's address, so all callers share one bucket.
Fixing that needs the forwarded address plus an explicit trusted-hop config, which is a deployment decision rather than a default.
On the public demo I did not fix it: the shared upstream Gemini budget is what actually protects the quota there, and it is sized for that.

`GEMINI_RPM` and `GEMINI_RPD` default to 15 and 1000, which is the conservative reading of a number Google no longer publishes per model.
The docs say limits depend on the account and are shown live in [AI Studio](https://aistudio.google.com/rate-limit), and third-party trackers disagree (1,000 vs 1,500 RPD).
Check the real number for your key rather than trusting the default.

## Reporting nothing beats reporting a plausible number

The PRD asks for a false-answer rate and `/admin/metrics` returns `null`, on purpose.

A false answer is one the bot gave confidently and *wrongly*, and nothing in a request says it was wrong.
That needs ground truth, or a human saying so.
It is produced by the eval against labelled queries, not by live traffic.
Reporting a plausible-looking number here, or quietly redefining it as something cheaper to measure, would be worse than reporting nothing, because someone would put it in a slide.

Rates are `null` rather than `0` when nothing has happened yet, for the same reason.
A 0% deflection rate is an emergency.
No traffic is a Tuesday.

## Logs that are half JSON and half prose parse as neither

All logs are JSON, including uvicorn's access lines, which are routed through the same formatter.

The request id comes back in the `X-Request-ID` header, so a customer complaint ("it told me refunds take 30 days") traces to the exact turn, and from there to the exact chunks that were retrieved.

`/admin/metrics` exposes no message content, only counts, but "how often does this bot fail" is not a number to leave open to the internet.
It is open when `ADMIN_API_KEY` is unset, which is fine locally and is not fine on the internet.

## The demo container

The demo runs the whole stack in one container, with Qdrant and Redis inside it.
One container because the hosts that will run one for free will run exactly one.
The Qdrant binary is copied out of the same image tag compose pins, so the demo and the compose stack run the same server.

`docker-compose.yml` remains the real topology, and the two are worth testing separately.
A bug that lives in the gap between them, a missing shared library say, is invisible to `docker compose up`.

Two things about the container are deliberate.
The embedding and reranker weights are baked in at **build** time, because the service scales to zero and cold-starts on the next visit, and the visitor who arrives during that cold start is exactly the person the demo exists for.
The collection is rebuilt from `kb/` at **start** time, because there is no persistent disk.
It costs a few seconds on a corpus this size, and it means there is no stale-index failure mode.

Cloud Run is where it runs today because its free tier scales to zero, which means a demo nobody is looking at costs nothing.
It was going to be a Hugging Face Space, until July 2026, when HF made Docker Spaces on free hardware a PRO feature.
Static Spaces are still free, and cannot run any of this.

The deploy script builds from `git archive HEAD`, which is tracked files only, so it cannot ship a secret by accident.

## Smaller things that cost me time

The embedding model reads at most 256 tokens and silently truncates beyond that, so `chunk_size_tokens` is capped well under it.
Ingestion refuses to run if the budget is set too high rather than let the tail of a chunk be stored but never embedded.

`/health` returns 200 only when Qdrant and Redis are both reachable, so a networking problem surfaces there rather than at the first query.

`eval/queries.jsonl` is committed on purpose.
A benchmark that resamples every run measures something different every run.

The Qdrant image and `qdrant-client` are pinned to the same minor version.
They do not tolerate drift: the client warns on every call, and the on-disk format does not survive a wide version jump.

The spec asked for `gemini-2.5-flash-lite`, which Google has since closed to new API keys.
It still shows up in `models.list()` but calling it returns a 404, so the spec's choice is not buildable as written.
`3.1-flash-lite` is the current model at the same tier, and it is pinned to an explicit version rather than the `-latest` alias, because an alias that moves under you makes the benchmark numbers unreproducible.

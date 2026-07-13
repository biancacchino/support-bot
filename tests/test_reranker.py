"""Reranker and confidence-gate tests.

The one that matters is test_rerank_catches_what_similarity_alone_would_answer.
The whole two-stage design exists for the confidently-wrong-on-topic case, and a
suite that never demonstrates the reranker catching something cosine similarity
would have waved through has not tested the feature - it has tested that the
code runs.
"""

import math

import pytest

from app.reranker import Ranked, Reranker, RerankResult, sigmoid
from app.retrieval import Candidate
from tests.conftest import IN_SCOPE_QUERIES, OFF_TOPIC_QUERIES

THRESHOLD = 0.35


def candidate(doc_id: str, text: str = "body text", score: float = 0.5) -> Candidate:
    return Candidate(
        doc_id=doc_id,
        title=doc_id.replace("-", " ").capitalize(),
        category="ORDER",
        intents=("track_order",),
        heading="A heading",
        chunk_index=0,
        text=text,
        score=score,
    )


def fake_cross_encoder(logits_by_doc: dict[str, float]):
    """A cross-encoder whose verdict we choose, keyed by the doc in the passage."""

    def cross_encode(pairs):
        out = []
        for _query, passage in pairs:
            for doc_id, logit in logits_by_doc.items():
                if doc_id.replace("-", " ").capitalize() in passage:
                    out.append(logit)
                    break
            else:
                raise AssertionError(f"unexpected passage: {passage!r}")
        return out

    return cross_encode


def make_reranker(logits_by_doc, top_k=4, threshold=THRESHOLD) -> Reranker:
    return Reranker(fake_cross_encoder(logits_by_doc), top_k, threshold)


# --- mechanics --------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_reorders_by_cross_encoder_not_by_retrieval_score():
    """The point of stage 2: stage 1's order is a suggestion, not a verdict."""
    # Retrieval liked "wrong" most (0.9); the cross-encoder disagrees.
    candidates = [candidate("wrong", score=0.9), candidate("right", score=0.4)]
    reranker = make_reranker({"wrong": -4.0, "right": 4.0})

    result = await reranker.rerank("a query", candidates)

    assert [r.doc_id for r in result.ranked] == ["right", "wrong"]
    assert result.top.doc_id == "right"


@pytest.mark.asyncio
async def test_rerank_keeps_only_top_k():
    candidates = [candidate(f"doc-{i}") for i in range(10)]
    logits = {f"doc-{i}": float(i) for i in range(10)}

    result = await make_reranker(logits, top_k=4).rerank("a query", candidates)

    assert [r.doc_id for r in result.ranked] == ["doc-9", "doc-8", "doc-7", "doc-6"]


@pytest.mark.asyncio
async def test_rerank_reads_the_passage_with_its_title_and_heading():
    """The cross-encoder must see the context the bare chunk text leaves out."""
    seen = []

    def spy(pairs):
        seen.extend(pairs)
        return [1.0] * len(pairs)

    await Reranker(spy, 4, THRESHOLD).rerank("a query", [candidate("tracking-your-order")])

    query, passage = seen[0]
    assert query == "a query"
    assert "Tracking your order" in passage
    assert "A heading" in passage
    assert "body text" in passage


@pytest.mark.asyncio
async def test_confidence_is_the_best_chunk_not_the_average():
    """Three plausible-but-weak chunks must not drown out the one that answers."""
    candidates = [candidate("weak-1"), candidate("weak-2"), candidate("weak-3"), candidate("strong")]
    logits = {"weak-1": -2.0, "weak-2": -2.0, "weak-3": -2.0, "strong": 3.0}

    result = await make_reranker(logits).rerank("a query", candidates)

    assert result.confidence == pytest.approx(sigmoid(3.0))
    assert result.escalate is False


# --- the gate ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalates_when_the_best_candidate_is_weak():
    result = await make_reranker({"doc": -1.0}).rerank("a query", [candidate("doc")])

    assert result.escalate is True
    assert result.confidence < THRESHOLD
    # Escalating still hands over what it found. Phase 6 needs these.
    assert result.ranked != []


@pytest.mark.asyncio
async def test_answers_when_the_best_candidate_is_strong():
    result = await make_reranker({"doc": 2.0}).rerank("a query", [candidate("doc")])

    assert result.escalate is False
    assert result.confidence > THRESHOLD


@pytest.mark.asyncio
async def test_threshold_is_a_floor_not_a_ceiling():
    """Exactly at the threshold is confident enough. Below it is not."""
    at = math.log(THRESHOLD / (1 - THRESHOLD))  # logit whose sigmoid is THRESHOLD

    result = await make_reranker({"doc": at}).rerank("a query", [candidate("doc")])

    assert result.confidence == pytest.approx(THRESHOLD)
    assert result.escalate is False


@pytest.mark.asyncio
async def test_nothing_retrieved_escalates():
    """No answer is not a confident answer. Do not invite the LLM to invent one."""
    result = await make_reranker({}).rerank("a query", [])

    assert result == RerankResult(ranked=[], confidence=0.0, escalate=True)
    assert result.top is None


def test_ranked_carries_the_whole_candidate_through():
    """Phase 4 cites the source and Phase 6 hands the chunks to a human."""
    ranked = Ranked(candidate=candidate("refund-policy", text="Refunds take 5 days."), score=0.9)

    assert ranked.doc_id == "refund-policy"
    assert ranked.candidate.text == "Refunds take 5 days."


# --- the case this whole stage exists for -----------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_rerank_catches_what_similarity_alone_would_answer(real_retriever, real_reranker):
    """Phase 3 acceptance: the rerank gate catches a query a similarity gate answers.

    The comparison is made fair by construction. A similarity-only gate does not
    get to pick a convenient threshold: it has to be lenient enough to still
    answer every genuine query, so the most aggressive one it can possibly use
    is the *lowest* top-1 cosine similarity any in-scope query produces. That is
    the strongest similarity-only gate that exists.

    Then we show an off-topic query that clears even that bar on cosine
    similarity - so this gate answers it, confidently and wrongly, citing a
    document that has nothing to do with the question - and that the rerank gate
    escalates.
    """
    in_scope_similarities = []
    for query, _ in IN_SCOPE_QUERIES:
        hits = await real_retriever.search(query)
        in_scope_similarities.append(hits[0].score)
    strongest_similarity_gate = min(in_scope_similarities)

    impostor = "i want to file a complaint about your service"
    candidates = await real_retriever.search(impostor)
    cosine = candidates[0].score

    # A similarity-only gate would have answered this.
    assert cosine >= strongest_similarity_gate, (
        f"{impostor!r} scores {cosine:.3f} on cosine, below the {strongest_similarity_gate:.3f} "
        "floor, so a similarity gate would have caught it and this test proves nothing"
    )

    # The reranker, having actually read the question against the passage, does not.
    result = await real_reranker.rerank(impostor, candidates)

    assert result.escalate is True, (
        f"{impostor!r} was answered with {result.top.doc_id} at {result.confidence:.3f} - "
        "the confidence gate is not doing the one job it exists for"
    )
    assert result.confidence < cosine, (
        "the reranker is no more sceptical of this query than raw similarity was"
    )


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("query", OFF_TOPIC_QUERIES)
async def test_off_topic_queries_escalate(real_retriever, real_reranker, query):
    """Nothing the KB cannot answer should get an answer anyway."""
    result = await real_reranker.rerank(query, await real_retriever.search(query))

    assert result.escalate is True, (
        f"answered {query!r} with {result.top.doc_id} at confidence {result.confidence:.3f}"
    )


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="known gap: a half-in-scope adversarial query. The bot answers the "
    "half it knows and ignores the fraud. Left failing on purpose - Phase 12 "
    "asks explicitly about adversarial input, and a silently deleted test case "
    "would be how that question gets answered wrongly.",
)
async def test_adversarial_mixed_query_escalates(real_retriever, real_reranker):
    """Half in scope, half not, and the in-scope half is enough to convince it.

    "can i track my order if i paid with a stolen card" is a tracking question,
    which the KB answers, wrapped around a fraud admission, which it does not.
    The reranker scores it 0.75 and answers it out of the payments doc. It should
    escalate: a human needs to see this one, and the confident half of the answer
    is what makes that failure dangerous rather than merely unhelpful.
    """
    query = "can i track my order if i paid with a stolen card"

    result = await real_reranker.rerank(query, await real_retriever.search(query))

    assert result.escalate is True


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected_doc"), IN_SCOPE_QUERIES)
async def test_genuine_queries_are_answered_from_the_right_document(
    real_retriever, real_reranker, query, expected_doc
):
    """The other half of the gate, and the half a paranoid threshold breaks.

    A gate that escalates everything catches every impostor and deflects nothing,
    so this asserts the genuine queries survive it - each escalation here is a
    deflection the product does not get.

    The document assertion is *membership in the top k*, not rank 1, because the
    top k is what the design actually promises: all four chunks go to the LLM and
    any of them can be cited, so an answer is grounded and correctly sourced as
    long as the right document is among them. It is 8/8 on that.

    Rank 1 is 6/8, and the two misses are honest near-misses - "my card got
    rejected at checkout" leads with accepted-payment-methods over
    payment-declined-or-failed, and "do you charge me for cancelling" leads with
    cancelling-an-order over cancellation-fees. In both the right document is
    second and still cited. Asserting rank 1 here would pin a number the design
    does not depend on; Phase 11 measures rank properly, as MRR, over 200-300
    queries instead of 8.
    """
    result = await real_reranker.rerank(query, await real_retriever.search(query))

    assert result.escalate is False, (
        f"escalated {query!r} at confidence {result.confidence:.3f} - that is a lost deflection"
    )
    assert expected_doc in {r.doc_id for r in result.ranked}, (
        f"{expected_doc!r} is not among the chunks handed to the LLM for {query!r}; "
        f"got {[r.doc_id for r in result.ranked]} - nothing can cite it now"
    )

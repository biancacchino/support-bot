"""Answer-generation tests, with the LLM mocked.

No real Gemini call happens here: CI must not depend on a quota, a network, or a
model's mood. What is being tested is not "does Gemini write good prose" - it is
the contract around it, which is ours and is enforceable: the model is shown only
the retrieved chunks, and nothing it returns reaches a customer unless it cites a
source we actually gave it.

The citation rule is P0, so it is asserted structurally - every path that yields a
GeneratedAnswer yields one with a real, non-empty citation - rather than checked
by eye on a happy-path example.
"""

import json

import pytest

from app.llm import (
    AnswerGenerator,
    GeneratedAnswer,
    UngroundedAnswer,
    build_prompt,
)
from app.reranker import Ranked
from app.retrieval import Candidate


def ranked(doc_id: str, text: str = "Refunds are issued within 5 working days.") -> Ranked:
    return Ranked(
        candidate=Candidate(
            doc_id=doc_id,
            title=doc_id.replace("-", " ").capitalize(),
            category="REFUND",
            intents=("get_refund",),
            heading="Timing",
            chunk_index=0,
            text=text,
            score=0.6,
        ),
        score=0.9,
    )


def responder(payload):
    """An LLM that returns exactly what we tell it to."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)

    async def generate(_prompt: str) -> str:
        return raw

    return generate


def spy_responder(payload):
    """As above, but records the prompt it was given."""
    seen: list[str] = []
    inner = responder(payload)

    async def generate(prompt: str) -> str:
        seen.append(prompt)
        return await inner(prompt)

    return generate, seen


# --- the prompt -------------------------------------------------------------


def test_prompt_carries_the_sources_and_their_ids():
    prompt = build_prompt(
        "when do i get my money back",
        [ranked("refund-policy", "Refunds take 5 days."), ranked("tracking-your-refund", "Track it here.")],
    )

    assert "when do i get my money back" in prompt
    assert "source-id: refund-policy" in prompt
    assert "source-id: tracking-your-refund" in prompt
    assert "Refunds take 5 days." in prompt
    assert "Track it here." in prompt


def test_prompt_shows_the_model_nothing_but_the_retrieved_chunks():
    """Grounding is enforced by what the model is given, not only by asking it."""
    prompt = build_prompt("when do i get my money back", [ranked("refund-policy", "Refunds take 5 days.")])

    assert "tracking-your-refund" not in prompt
    assert "cancellation-fees" not in prompt


# --- what gets served -------------------------------------------------------


@pytest.mark.asyncio
async def test_grounded_answer_is_returned_with_its_citations():
    generate = responder({"answer": "Refunds take 5 working days.", "citations": ["refund-policy"]})

    answer = await AnswerGenerator(generate).answer("when do i get my money back", [ranked("refund-policy")])

    assert answer == GeneratedAnswer(text="Refunds take 5 working days.", citations=("refund-policy",))


@pytest.mark.asyncio
async def test_duplicate_citations_are_collapsed():
    generate = responder({"answer": "Five days.", "citations": ["refund-policy", "refund-policy"]})

    answer = await AnswerGenerator(generate).answer("q", [ranked("refund-policy")])

    assert answer.citations == ("refund-policy",)


# --- what does not get served ----------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_citing_nothing_is_refused():
    """P0: no uncited claims. An answer with no source does not reach a customer."""
    generate = responder({"answer": "Refunds take about a week, I think.", "citations": []})

    with pytest.raises(UngroundedAnswer, match="without citing"):
        await AnswerGenerator(generate).answer("q", [ranked("refund-policy")])


@pytest.mark.asyncio
async def test_a_fabricated_citation_is_refused():
    """The dangerous failure: a real-sounding answer attributed to a page we never showed it.

    This is worse than no citation at all. The citation is the part a customer
    trusts, so an invented one launders a guess into something that looks sourced.
    """
    generate = responder({"answer": "Refunds take 30 days.", "citations": ["refund-policy-v2"]})

    with pytest.raises(UngroundedAnswer, match="cited sources it was not given"):
        await AnswerGenerator(generate).answer("q", [ranked("refund-policy")])


@pytest.mark.asyncio
async def test_a_declined_answer_is_refused_rather_than_served_empty():
    """The model saying "not in the sources" is a correct outcome - it escalates."""
    generate = responder({"answer": "", "citations": []})

    with pytest.raises(UngroundedAnswer, match="declined"):
        await AnswerGenerator(generate).answer("q", [ranked("refund-policy")])


@pytest.mark.asyncio
async def test_a_non_json_reply_is_refused():
    generate = responder("Sure! Refunds usually take about a week.")

    with pytest.raises(UngroundedAnswer, match="did not return JSON"):
        await AnswerGenerator(generate).answer("q", [ranked("refund-policy")])


@pytest.mark.asyncio
async def test_no_sources_means_the_model_is_never_asked():
    """Nothing to ground in is not a prompt to write carefully - it is no prompt at all."""
    generate, seen = spy_responder({"answer": "anything", "citations": ["refund-policy"]})

    with pytest.raises(UngroundedAnswer, match="no sources"):
        await AnswerGenerator(generate).answer("q", [])

    assert seen == []


# --- the P0 rule, structurally ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "Five days.", "citations": ["refund-policy"]},
        {"answer": "Five days.", "citations": ["refund-policy", "tracking-your-refund"]},
        {"answer": "Five days.", "citations": []},
        {"answer": "Five days.", "citations": ["made-up-doc"]},
        {"answer": "", "citations": ["refund-policy"]},
        {"answer": "Five days.", "citations": ["refund-policy", "made-up-doc"]},
        "not json at all",
    ],
)
async def test_every_served_answer_cites_a_real_source(payload):
    """The invariant, over every shape of reply: served implies cited and real.

    Either the call raises, or it returns an answer whose citations are non-empty
    and drawn from the sources actually supplied. There is no third outcome, and
    that is what "citation on every answer, no exceptions" means in code.
    """
    sources = [ranked("refund-policy"), ranked("tracking-your-refund")]
    supplied = {r.candidate.doc_id for r in sources}

    try:
        answer = await AnswerGenerator(responder(payload)).answer("q", sources)
    except UngroundedAnswer:
        return  # refused, which is always an acceptable outcome

    assert answer.text
    assert answer.citations
    assert set(answer.citations) <= supplied

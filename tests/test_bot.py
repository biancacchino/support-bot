"""One turn of the bot, end to end, with the models stubbed.

The gate, the store and the citation rules are real here - only the two model
calls (cross-encoder, LLM) are stubbed, because those are the slow parts and their
behaviour is pinned in their own suites. What is being tested is the decision the
bot makes with what they return, and what it hands to a human when it declines.

The two outcomes are different types on purpose. An Answered cannot reach an
agent's queue and an Escalated cannot reach a customer, and `isinstance` says so
more loudly than an `escalated=True` flag ever could.
"""

import pytest
from fakeredis.aioredis import FakeRedis

from app.bot import Answered, Escalated, EscalationReason, SupportBot
from app.conversation import Condenser, ConversationStore, Turn
from app.llm import AnswerGenerator
from app.reranker import Reranker
from app.retrieval import Candidate

TTL = 3600
THRESHOLD = 0.35

ANSWER_JSON = '{"answer": "Refunds take five working days.", "citations": ["refund-policy"]}'
UNCITED_JSON = '{"answer": "Refunds take about a week, I reckon.", "citations": []}'
CONDENSE_JSON = '{"query": "how long does my refund take"}'


def candidate(doc_id: str) -> Candidate:
    return Candidate(
        doc_id=doc_id,
        title=doc_id.replace("-", " ").capitalize(),
        category="REFUND",
        intents=("get_refund",),
        heading="Timing",
        chunk_index=0,
        text="Refunds are issued within 5 working days.",
        score=0.5,
    )


class StubRetriever:
    """Returns a fixed candidate set, and remembers what it was asked."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates
        self.queries: list[str] = []

    async def search(self, query: str, limit: int | None = None) -> list[Candidate]:
        self.queries.append(query)
        return self._candidates


def cross_encoder(logit: float):
    def encode(pairs):
        return [logit] * len(pairs)

    return encode


def responder(raw: str, calls: list | None = None):
    async def generate(prompt: str) -> str:
        if calls is not None:
            calls.append(prompt)
        return raw

    return generate


@pytest.fixture
async def redis():
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def make_bot(redis, *, logit: float, answer_json: str = ANSWER_JSON, llm_calls=None) -> SupportBot:
    """A bot whose reranker is as confident as `logit` says, and no more."""
    store = ConversationStore(redis, ttl_seconds=TTL)
    return SupportBot(
        retriever=StubRetriever([candidate("refund-policy"), candidate("tracking-your-refund")]),
        reranker=Reranker(cross_encoder(logit), top_k=4, threshold=THRESHOLD),
        generator=AnswerGenerator(responder(answer_json, llm_calls)),
        condenser=Condenser(responder(CONDENSE_JSON)),
        store=store,
    )


CONFIDENT = 3.0  # sigmoid ~0.95, well over the gate
DOUBTFUL = -3.0  # sigmoid ~0.05, well under it


# --- answering --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_confident_turn_is_answered_with_citations(redis):
    result = await make_bot(redis, logit=CONFIDENT).handle("how long for a refund", "c1")

    assert isinstance(result, Answered)
    assert result.text == "Refunds take five working days."
    assert result.citations == ("refund-policy",)
    assert result.confidence > THRESHOLD


@pytest.mark.asyncio
async def test_an_answered_turn_is_remembered(redis):
    bot = make_bot(redis, logit=CONFIDENT)
    store = ConversationStore(redis, ttl_seconds=TTL)

    await bot.handle("how long for a refund", "c1")

    assert await store.history("c1") == [
        Turn(
            question="how long for a refund",
            answer="Refunds take five working days.",
            escalated=False,
            citations=("refund-policy",),
        )
    ]


# --- escalating -------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unconfident_turn_escalates(redis):
    result = await make_bot(redis, logit=DOUBTFUL).handle("what is your ceo paid", "c1")

    assert isinstance(result, Escalated)
    assert result.reason is EscalationReason.LOW_CONFIDENCE
    assert result.confidence < THRESHOLD


@pytest.mark.asyncio
async def test_a_confident_but_uncited_answer_escalates(redis):
    """The gate believed the chunks; the model would not cite them. Hand it over."""
    result = await make_bot(redis, logit=CONFIDENT, answer_json=UNCITED_JSON).handle("q", "c1")

    assert isinstance(result, Escalated)
    assert result.reason is EscalationReason.UNGROUNDED


@pytest.mark.asyncio
async def test_the_handoff_carries_what_an_agent_needs_to_act(redis):
    """Phase 6: an agent should not have to ask the customer anything again."""
    bot = make_bot(redis, logit=DOUBTFUL)
    store = ConversationStore(redis, ttl_seconds=TTL)
    await store.append("c1", Turn(question="i want a refund", answer="Sure.", escalated=False))

    result = await bot.handle("how long will it take", "c1")

    assert isinstance(result, Escalated)
    assert result.query == "how long will it take"  # what they actually typed
    assert result.condensed_query == "how long does my refund take"  # what we searched on
    assert [t.question for t in result.history] == ["i want a refund"]  # what came before
    assert result.confidence < THRESHOLD  # how sure the bot was


@pytest.mark.asyncio
async def test_the_handoff_carries_the_chunks_that_lost(redis):
    """The below-threshold chunks are the evidence, so they travel with the escalation.

    An agent who can see the bot nearly matched the refund policy knows something
    quite different from one who sees that it matched nothing at all. Withholding
    the losing chunks hides the reason this escalated.
    """
    result = await make_bot(redis, logit=DOUBTFUL).handle("q", "c1")

    assert isinstance(result, Escalated)
    assert [chunk.doc_id for chunk in result.chunks] == ["refund-policy", "tracking-your-refund"]
    assert all(chunk.score < THRESHOLD for chunk in result.chunks)


@pytest.mark.asyncio
async def test_the_history_handed_over_does_not_repeat_this_turn(redis):
    """`query` is this turn. Showing it again inside `history` would just be noise."""
    result = await make_bot(redis, logit=DOUBTFUL).handle("only question", "c1")

    assert isinstance(result, Escalated)
    assert result.history == []


# --- sticky escalation ------------------------------------------------------


@pytest.mark.asyncio
async def test_once_escalated_the_conversation_stays_with_the_human(redis):
    """The decision from 6.2: escalation is sticky and permanent.

    A confident turn arriving after an escalation is still escalated. The bot
    jumping back in while an agent is mid-reply gives the customer two voices in
    one thread, which is worse than a human spending ten seconds on an easy
    follow-up.
    """
    store = ConversationStore(redis, ttl_seconds=TTL)
    await make_bot(redis, logit=DOUBTFUL).handle("i want to complain", "c1")
    assert await store.is_escalated("c1")

    # A turn the bot would happily have answered, had it not lost the conversation.
    llm_calls: list[str] = []
    result = await make_bot(redis, logit=CONFIDENT, llm_calls=llm_calls).handle(
        "where is my order", "c1"
    )

    assert isinstance(result, Escalated)
    assert result.reason is EscalationReason.HUMAN_OWNED
    assert result.confidence > THRESHOLD  # it was confident, and answered anyway: no
    assert llm_calls == []  # and it did not spend a Gemini call finding that out


@pytest.mark.asyncio
async def test_a_human_owned_turn_still_hands_over_the_context(redis):
    """The bot may not answer, but the agent should not have to search by hand."""
    await make_bot(redis, logit=DOUBTFUL).handle("i want to complain", "c1")

    result = await make_bot(redis, logit=CONFIDENT).handle("where is my order", "c1")

    assert isinstance(result, Escalated)
    assert result.chunks != []
    assert [t.question for t in result.history] == ["i want to complain"]


@pytest.mark.asyncio
async def test_escalation_does_not_leak_between_conversations(redis):
    await make_bot(redis, logit=DOUBTFUL).handle("i want to complain", "c1")

    result = await make_bot(redis, logit=CONFIDENT).handle("how long for a refund", "c2")

    assert isinstance(result, Answered)


@pytest.mark.asyncio
async def test_an_escalated_turn_is_remembered_as_escalated(redis):
    store = ConversationStore(redis, ttl_seconds=TTL)

    await make_bot(redis, logit=DOUBTFUL).handle("i want to complain", "c1")

    assert await store.history("c1") == [
        Turn(question="i want to complain", answer="", escalated=True, citations=())
    ]

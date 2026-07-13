"""Conversation history and query condensation.

The acceptance test is test_turn_two_needs_turn_one, and it is written to fail if
condensation is removed. That is the whole point of it: a two-turn test that only
checks the endpoint returned 200 would pass against a bot that ignores history
entirely, and would tell us nothing.

Redis is faked, not mocked - fakeredis runs the real client against a real
implementation of the commands, so RPUSH ordering and TTL semantics are exercised
rather than asserted about.
"""

import pytest
from fakeredis.aioredis import FakeRedis

from app.conversation import (
    HISTORY_WINDOW,
    Condenser,
    ConversationStore,
    Turn,
    format_history,
)

TTL = 3600

REFUND_DOCS = {"requesting-a-refund", "refund-policy", "tracking-your-refund", "returns-and-damaged-goods"}


@pytest.fixture
async def store():
    redis = FakeRedis(decode_responses=True)
    yield ConversationStore(redis, ttl_seconds=TTL)
    await redis.aclose()


def turn(question: str, answer: str = "An answer.", escalated: bool = False, citations=("refund-policy",)) -> Turn:
    return Turn(question=question, answer=answer, escalated=escalated, citations=citations)


def responder(raw: str):
    async def generate(_prompt: str) -> str:
        return raw

    return generate


# --- history ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_turns_come_back_in_the_order_they_happened(store):
    await store.append("c1", turn("first"))
    await store.append("c1", turn("second"))

    assert [t.question for t in await store.history("c1")] == ["first", "second"]


@pytest.mark.asyncio
async def test_a_turn_survives_the_round_trip_intact(store):
    await store.append("c1", turn("q", answer="a", escalated=True, citations=("refund-policy", "cancellation-fees")))

    assert (await store.history("c1"))[0] == Turn(
        question="q", answer="a", escalated=True, citations=("refund-policy", "cancellation-fees")
    )


@pytest.mark.asyncio
async def test_conversations_do_not_leak_into_each_other(store):
    await store.append("c1", turn("mine"))
    await store.append("c2", turn("yours"))

    assert [t.question for t in await store.history("c1")] == ["mine"]
    assert [t.question for t in await store.history("c2")] == ["yours"]


@pytest.mark.asyncio
async def test_an_unknown_conversation_has_no_history(store):
    assert await store.history("never-seen") == []


@pytest.mark.asyncio
async def test_history_is_windowed_to_the_recent_turns(store):
    for i in range(HISTORY_WINDOW + 4):
        await store.append("c1", turn(f"q{i}"))

    history = await store.history("c1")

    assert len(history) == HISTORY_WINDOW
    assert history[-1].question == f"q{HISTORY_WINDOW + 3}"  # the most recent survives
    assert history[0].question == "q4"  # the oldest is dropped


@pytest.mark.asyncio
async def test_history_expires(store):
    await store.append("c1", turn("q"))

    ttl = await store._redis.ttl("conversation:c1")

    assert 0 < ttl <= TTL


@pytest.mark.asyncio
async def test_a_new_turn_pushes_the_expiry_out(store):
    """TTL measures silence, not age. An hour into a live chat is no time to forget it."""
    await store.append("c1", turn("q1"))
    await store._redis.expire("conversation:c1", 5)  # nearly expired

    await store.append("c1", turn("q2"))

    assert await store._redis.ttl("conversation:c1") > 5


def test_escalated_turns_are_shown_as_escalated_in_the_history():
    """The next turn's condensation has to know a human took over."""
    formatted = format_history([turn("i want to complain", answer="", escalated=True)])

    assert "i want to complain" in formatted
    assert "escalated" in formatted


# --- condensation -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_first_turn_is_not_condensed():
    """Nothing to resolve against, so do not spend a round trip resolving it."""
    called = []

    async def generate(prompt: str) -> str:
        called.append(prompt)
        return '{"query": "should not be used"}'

    condensed = await Condenser(generate).condense("where is my order", history=[])

    assert condensed == "where is my order"
    assert called == []


@pytest.mark.asyncio
async def test_a_follow_up_is_rewritten_to_stand_alone():
    generate = responder('{"query": "How long does a refund for a damaged item take?"}')

    condensed = await Condenser(generate).condense(
        "how long will it take", history=[turn("i want a refund for a damaged item")]
    )

    assert condensed == "How long does a refund for a damaged item take?"


@pytest.mark.asyncio
async def test_the_condenser_is_shown_the_history_and_the_follow_up():
    seen = []

    async def generate(prompt: str) -> str:
        seen.append(prompt)
        return '{"query": "x"}'

    await Condenser(generate).condense(
        "how long will it take", history=[turn("i want a refund for a damaged item")]
    )

    assert "i want a refund for a damaged item" in seen[0]
    assert "how long will it take" in seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not json", '{"query": ""}', "{}", '["wrong shape"]'])
async def test_a_failed_condensation_falls_back_to_the_raw_follow_up(raw):
    """Degraded, not broken: this is exactly the behaviour we had before the feature.

    The customer's question still gets retrieved on, just without its context.
    Failing the turn outright because a rewrite did not parse would be a worse
    trade than answering it slightly worse.
    """
    condensed = await Condenser(responder(raw)).condense("how long will it take", history=[turn("q")])

    assert condensed == "how long will it take"


# --- the acceptance criterion -----------------------------------------------


TURN_1 = "i want to cancel my order"
TURN_2 = "how much will that cost me"

# The documents that genuinely answer turn 2 once it is read in context.
CANCEL_DOCS = {"cancellation-fees", "cancelling-an-order"}

# What Gemini actually returns for this follow-up, observed against the real API
# and pinned here. The LLM is stubbed - CI does not call Gemini - but the rewrite
# it is stubbed with is a real one, not a convenient one.
CONDENSED_TURN_2 = "How much will it cost me to cancel my order?"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_turn_two_needs_turn_one(store, real_retriever, real_reranker):
    """Phase 5 acceptance: turn 2 is only answerable because turn 1 happened.

    Written so that it *fails* if condensation is removed, which is the bar the
    task list sets. It runs through the real Condenser and the real conversation
    history, against the real retriever and reranker - so a Condenser that stopped
    rewriting, or a history that stopped being read, breaks this test rather than
    quietly degrading the product.

    "How much will that cost me" is about a cancellation fee only if you have read
    turn 1. On its own it retrieves the *delivery price list* - a document about a
    different subject entirely - at 0.063. Condensed against the history it reaches
    `cancellation-fees` at 0.976.

    The guard below is the load-bearing part. It asserts the raw follow-up does
    *not* already find the right document, because a control that passes by accident
    proves nothing about the feature. It has caught a real regression once: adding a
    "how long a refund takes" section to the KB made the previous scenario's raw
    follow-up ("how long will it take") retrieve the refund docs unaided, which
    quietly turned this test into one that would pass with condensation removed.
    """
    # Turn 1 happened, and is in the history.
    await store.append("c1", turn(TURN_1, answer="You can cancel from Orders while it is still Placed."))

    # Turn 2, condensed against that history exactly as the app does it.
    condenser = Condenser(responder(f'{{"query": "{CONDENSED_TURN_2}"}}'))
    retrieval_query = await condenser.condense(TURN_2, await store.history("c1"))

    condensed_result = await real_reranker.rerank(
        retrieval_query, await real_retriever.search(retrieval_query)
    )

    # Turn 2 as the customer typed it, with the history ignored. This is the
    # control, and it is what the whole feature is measured against.
    raw_result = await real_reranker.rerank(TURN_2, await real_retriever.search(TURN_2))

    assert raw_result.top.doc_id not in CANCEL_DOCS, (
        "the raw follow-up already retrieves the right document, so this test cannot "
        "prove condensation does anything - pick a follow-up that genuinely depends on turn 1"
    )

    assert condensed_result.escalate is False
    assert condensed_result.top.doc_id in CANCEL_DOCS, (
        f"condensed to {retrieval_query!r} and still did not reach the cancellation docs - "
        f"got {condensed_result.top.doc_id}. If condensation was disabled, this is how it shows up"
    )
    assert condensed_result.confidence > raw_result.confidence

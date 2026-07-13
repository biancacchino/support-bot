"""One turn of the bot: condense, retrieve, rerank, gate, answer or hand off.

This is the whole pipeline in one place, so the CLI, the API endpoint and the eval
harness all run the same code path rather than three drifting copies of it.

A turn ends in exactly one of two things, and they are different types rather than
one type with an `escalated` flag: an Answered carries prose and citations, an
Escalated carries everything a human agent needs to pick the conversation up cold.
Nothing that reaches a customer can be an Escalated, and nothing that reaches an
agent's queue can be an Answered, so the type system enforces what a boolean would
merely record.

Escalation is sticky and permanent. Once a conversation goes to a human it stays
with the human: every later turn escalates too, even one the bot could answer
confidently. The alternative - re-gating each turn - lets the bot start answering
again while an agent is mid-reply, and a customer getting two voices in one thread
is a worse failure than a human spending ten seconds on "thanks!". The cost is
real and accepted: trivial follow-ups after an escalation do burn agent time, and
the only way back to the bot is a new conversation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from app.conversation import Condenser, ConversationStore, Turn
from app.llm import AnswerGenerator, UngroundedAnswer
from app.reranker import Ranked, Reranker
from app.retrieval import Retriever

logger = logging.getLogger(__name__)


class EscalationReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"  # the reranker did not believe any chunk
    UNGROUNDED = "ungrounded"  # confident chunks, but the model would not cite them
    HUMAN_OWNED = "human_owned"  # an earlier turn escalated; a human has this now


@dataclass(frozen=True)
class Answered:
    """What a customer sees. Always cited - see llm.py."""

    text: str
    citations: tuple[str, ...]
    confidence: float
    condensed_query: str
    # The KB category this was answered out of, for the metrics breakdown. Taken
    # from the best chunk rather than from the citations, because the citations are
    # doc_ids and the category is what "escalation rate by category" is grouped on.
    category: str | None = None


@dataclass(frozen=True)
class Escalated:
    """What an agent picks up.

    Deliberately enough to act on without going back to the customer: what they
    asked, what they had already asked, what the bot found, and how sure it was.

    `chunks` are the reranked candidates *including the ones below the confidence
    threshold*. They are the reason this escalated, so withholding them would hide
    the evidence: an agent who can see the bot nearly matched a refund doc knows
    something different from one who sees it matched nothing at all.
    """

    reason: EscalationReason
    query: str
    condensed_query: str
    confidence: float
    history: list[Turn]
    chunks: list[Ranked]


class SupportBot:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: AnswerGenerator,
        condenser: Condenser,
        store: ConversationStore,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._condenser = condenser
        self._store = store

    async def handle(self, query: str, conversation_id: str) -> Answered | Escalated:
        history = await self._store.history(conversation_id)
        human_owned = await self._store.is_escalated(conversation_id)

        condensed = await self._condenser.condense(query, history)
        candidates = await self._retriever.search(condensed)
        result = await self._reranker.rerank(condensed, candidates)

        # Retrieval still runs for a human-owned conversation, and the chunks still
        # go in the payload. The bot is not allowed to answer, but the agent should
        # not have to search the knowledge base by hand either.
        if human_owned:
            return await self._escalate(
                EscalationReason.HUMAN_OWNED, query, condensed, result, history, conversation_id
            )

        if result.escalate:
            return await self._escalate(
                EscalationReason.LOW_CONFIDENCE, query, condensed, result, history, conversation_id
            )

        try:
            answer = await self._generator.answer(condensed, result.ranked)
        except UngroundedAnswer as exc:
            # The gate was confident and the model still would not ground an
            # answer. Handing over beats serving something uncited.
            logger.info("ungrounded answer for %r, escalating: %s", condensed, exc)
            return await self._escalate(
                EscalationReason.UNGROUNDED, query, condensed, result, history, conversation_id
            )

        await self._store.append(
            conversation_id,
            Turn(
                question=query,
                answer=answer.text,
                escalated=False,
                citations=answer.citations,
            ),
        )
        return Answered(
            text=answer.text,
            citations=answer.citations,
            confidence=result.confidence,
            condensed_query=condensed,
            category=result.top.candidate.category if result.top else None,
        )

    async def _escalate(
        self,
        reason: EscalationReason,
        query: str,
        condensed: str,
        result,
        history: list[Turn],
        conversation_id: str,
    ) -> Escalated:
        # The history the agent is handed is the one from *before* this turn, plus
        # this turn recorded separately as `query`. Appending first would show them
        # the same question twice.
        await self._store.append(
            conversation_id, Turn(question=query, answer="", escalated=True)
        )
        await self._store.mark_escalated(conversation_id)

        logger.info(
            "escalating conversation %s (%s, confidence %.3f)",
            conversation_id,
            reason,
            result.confidence,
        )
        return Escalated(
            reason=reason,
            query=query,
            condensed_query=condensed,
            confidence=result.confidence,
            history=history,
            chunks=result.ranked,
        )

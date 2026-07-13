"""Multi-turn conversation: history in Redis, and the condensation step.

The condensation step is the whole feature. Retrieval is stateless - it embeds
the string it is handed - so a follow-up like "how long does that take?" retrieves
nothing useful, because "that" is not in the sentence. The pronoun is in the
previous turn, and the vector search cannot see it.

So before retrieval, the follow-up and the history are rewritten into one
standalone question ("how long does a refund for a damaged item take?"), and
*that* is what gets embedded. Everything downstream stays stateless and none of
it needs to know a conversation exists.

History lives in Redis under a TTL. A support conversation is not worth keeping
forever, and an expiring key is the cheapest possible privacy story: nothing has
to remember to delete it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from redis.asyncio import Redis

from app.llm import GenerateFn

logger = logging.getLogger(__name__)

# How many past turns the condenser is shown. Support follow-ups refer to the
# recent thread, not to something twenty turns ago, and an unbounded history is
# an unbounded prompt.
HISTORY_WINDOW = 6

CONDENSE_PROMPT = """\
Rewrite the customer's latest message into a single standalone question that can
be understood with no other context.

Resolve every pronoun and reference ("that", "it", "the second one") against the
conversation so far. Keep the customer's own wording wherever you can - do not
answer the question, do not add detail they did not give, and do not invent
specifics like order numbers or dates.

If the latest message already stands on its own, return it unchanged.

Reply with JSON in exactly this shape:
{"query": "<the standalone question>"}

CONVERSATION SO FAR

%(history)s

LATEST MESSAGE

%(latest)s"""


@dataclass(frozen=True)
class Turn:
    """One exchange. Kept whether it was answered or escalated.

    An escalated turn is still part of the conversation: the customer said it,
    and the next turn's condensation has to be able to see it.
    """

    question: str
    answer: str
    escalated: bool
    citations: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> Turn:
        payload: dict[str, Any] = json.loads(raw)
        return cls(
            question=payload["question"],
            answer=payload["answer"],
            escalated=payload["escalated"],
            citations=tuple(payload.get("citations", ())),
        )


class ConversationStore:
    """Append-only turn history per conversation, expiring on a TTL."""

    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"conversation:{conversation_id}"

    @staticmethod
    def _escalated_key(conversation_id: str) -> str:
        return f"conversation:{conversation_id}:escalated"

    async def append(self, conversation_id: str, turn: Turn) -> None:
        """Add a turn and push the expiry out.

        RPUSH rather than read-modify-write: two turns arriving at once should
        both survive, and a JSON blob rewritten by both would lose one.

        The TTL is reset on every turn, so it measures silence rather than age -
        an hour into a live conversation is not the moment to forget it.
        """
        key = self._key(conversation_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, turn.to_json())
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def history(self, conversation_id: str, limit: int = HISTORY_WINDOW) -> list[Turn]:
        """The last `limit` turns, oldest first. Unknown conversation: no turns."""
        raw = await self._redis.lrange(self._key(conversation_id), -limit, -1)
        return [Turn.from_json(item) for item in raw]

    async def mark_escalated(self, conversation_id: str) -> None:
        """Hand the conversation to a human, permanently.

        Escalation is sticky: from here on the bot does not answer in this
        conversation, even a turn it would be confident about. See app/bot.py for
        why, and for what that costs.

        Shares the history's TTL, so the flag cannot outlive the turns it refers
        to and leave a conversation escalated with nothing to hand over.
        """
        key = self._escalated_key(conversation_id)
        pipe = self._redis.pipeline()
        pipe.set(key, "1")
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def is_escalated(self, conversation_id: str) -> bool:
        return await self._redis.exists(self._escalated_key(conversation_id)) == 1


def format_history(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"Customer: {turn.question}")
        # What the assistant actually said. An escalation is reported as one,
        # because "we passed you to a human" is context for what comes next.
        lines.append(f"Assistant: {turn.answer}" if not turn.escalated else "Assistant: (escalated to a human agent)")
    return "\n".join(lines)


class Condenser:
    """Rewrites a follow-up plus history into a standalone query."""

    def __init__(self, generate: GenerateFn) -> None:
        self._generate = generate

    async def condense(self, query: str, history: list[Turn]) -> str:
        """The query to actually retrieve on.

        First turn: the query as given - there is nothing to resolve it against,
        and a round trip to the LLM would buy nothing.
        """
        if not history:
            return query

        prompt = CONDENSE_PROMPT % {"history": format_history(history), "latest": query}
        raw = await self._generate(prompt)

        condensed = self._parse(raw)
        if condensed is None:
            # Retrieving on the raw follow-up is degraded, not broken: it is
            # exactly the behaviour we would have had without this feature. Fall
            # back loudly rather than failing the customer's question outright.
            logger.warning("condensation failed, retrieving on the raw follow-up: %r", raw[:200])
            return query

        logger.debug("condensed %r -> %r", query, condensed)
        return condensed

    @staticmethod
    def _parse(raw: str) -> str | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        condensed = (payload.get("query") or "").strip() if isinstance(payload, dict) else ""
        return condensed or None

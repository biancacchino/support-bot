"""Ask the bot a question, end to end, against the real stack.

    python scripts/ask.py "where is my order"
    python scripts/ask.py --all      # a fixed set: things it should answer, things it should not
    python scripts/ask.py --chat     # a two-turn conversation, where turn 2 needs turn 1

The full pipeline: condense, retrieve, rerank, gate, and either answer with
citations or escalate. This is the first point where the whole thing is visible as
a customer would experience it, and it makes a real Gemini call, so it needs
GEMINI_API_KEY. --chat also needs Redis, which compose already runs.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.bot import Answered, SupportBot  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.conversation import Condenser, ConversationStore  # noqa: E402
from app.llm import AnswerGenerator, build_gemini  # noqa: E402
from app.reranker import Reranker, build_cross_encoder  # noqa: E402
from app.retrieval import Retriever, build_encoder  # noqa: E402

DEMO_QUERIES = [
    # answerable from the KB
    "where is my order right now",
    "do you charge me for cancelling",
    "my card got rejected at checkout",
    # not answerable: the gate should escalate rather than improvise
    "i want to file a complaint about your service",
    "what is the ceo of northwind's salary",
    "i need to renew my health insurance policy",
]

# Turn 2 is unanswerable on its own: "it" is only defined by turn 1.
CHAT_TURNS = [
    "i want a refund for a damaged item",
    "how long will it take",
]

# Turn 1 escalates. Turn 2 is a question the bot answers happily on its own - and
# does not answer here, because a human owns this conversation now.
STICKY_TURNS = [
    "i want to file a complaint about your service",
    "where is my order right now",
]


async def ask(bot: SupportBot, query: str, conversation_id: str) -> None:
    """One turn, printed the way an operator would want to read it."""
    print(f"\n{'=' * 78}\nQ  {query}")

    result = await bot.handle(query, conversation_id)

    if result.condensed_query != query:
        print(f"   (condensed to: {result.condensed_query!r})")

    if isinstance(result, Answered):
        print(f"-> ANSWERED   confidence {result.confidence:.3f}")
        print(f"\n   {result.text}")
        print(f"\n   sources: {', '.join(result.citations)}")
        return

    # The handoff payload, which is the whole point of an escalation: everything
    # an agent needs to pick this up without re-asking the customer anything.
    print(f"-> ESCALATED  {result.reason}  confidence {result.confidence:.3f}")
    if result.history:
        print(f"\n   conversation so far ({len(result.history)} turn(s)):")
        for turn in result.history:
            said = turn.answer if not turn.escalated else "(escalated)"
            print(f"     customer: {turn.question}")
            print(f"     bot:      {said[:70]}")
    print("\n   what the bot found, and how sure it was (below the gate, or it would have answered):")
    for chunk in result.chunks:
        print(f"     {chunk.score:.3f}  {chunk.doc_id}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="the customer's question")
    parser.add_argument("--all", action="store_true", help="run the demo set")
    parser.add_argument("--chat", action="store_true", help="run the two-turn conversation")
    parser.add_argument(
        "--sticky", action="store_true", help="show that an escalated conversation stays escalated"
    )
    args = parser.parse_args()

    if not any((args.query, args.all, args.chat, args.sticky)):
        parser.error("give a query, or --all, or --chat, or --sticky")

    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    generate = build_gemini(settings)

    bot = SupportBot(
        retriever=Retriever(
            qdrant, build_encoder(settings), settings.qdrant_collection, settings.retrieval_top_n
        ),
        reranker=Reranker(
            build_cross_encoder(settings), settings.rerank_top_k, settings.confidence_threshold
        ),
        generator=AnswerGenerator(generate),
        condenser=Condenser(generate),
        store=ConversationStore(redis, settings.conversation_ttl_seconds),
    )

    try:
        if args.chat or args.sticky:
            # A fresh id each run, so the demo never reads a previous run's history.
            conversation_id = f"demo-{uuid.uuid4().hex[:8]}"
            print(f"conversation {conversation_id}")
            for query in CHAT_TURNS if args.chat else STICKY_TURNS:
                await ask(bot, query, conversation_id)
        else:
            # One-shot questions are still conversations, just very short ones. A
            # fresh id each time keeps them from inheriting each other's history -
            # and, now that escalation is sticky, from inheriting each other's
            # escalations.
            for query in DEMO_QUERIES if args.all else [args.query]:
                await ask(bot, query, f"demo-{uuid.uuid4().hex[:8]}")
    finally:
        await qdrant.close()
        await redis.aclose()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

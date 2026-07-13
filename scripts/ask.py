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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.conversation import Condenser, ConversationStore, Turn  # noqa: E402
from app.llm import AnswerGenerator, UngroundedAnswer, build_gemini  # noqa: E402
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


async def ask(pipeline, query: str, conversation_id: str | None = None) -> None:
    """One turn, end to end. With a conversation_id, history is used and updated."""
    print(f"\n{'=' * 78}\nQ  {query}")

    history = await pipeline.store.history(conversation_id) if conversation_id else []
    retrieval_query = await pipeline.condenser.condense(query, history)

    if retrieval_query != query:
        print(f"   (condensed to: {retrieval_query!r})")

    candidates = await pipeline.retriever.search(retrieval_query)
    result = await pipeline.reranker.rerank(retrieval_query, candidates)

    turn = None
    if result.escalate:
        top = result.top.doc_id if result.top else "nothing"
        print(
            f"-> ESCALATED  confidence {result.confidence:.3f} < "
            f"{pipeline.threshold} (best guess was {top})"
        )
        turn = Turn(question=query, answer="", escalated=True)
    else:
        try:
            answer = await pipeline.generator.answer(retrieval_query, result.ranked)
        except UngroundedAnswer as exc:
            # The gate was confident and the model still would not ground an
            # answer. Escalating rather than serving something uncited is the point.
            print(f"-> ESCALATED  confidence {result.confidence:.3f}, but ungrounded: {exc}")
            turn = Turn(question=query, answer="", escalated=True)
        else:
            print(f"-> ANSWERED   confidence {result.confidence:.3f}")
            print(f"\n   {answer.text}")
            print(f"\n   sources: {', '.join(answer.citations)}")
            turn = Turn(
                question=query,
                answer=answer.text,
                escalated=False,
                citations=answer.citations,
            )

    if conversation_id:
        await pipeline.store.append(conversation_id, turn)


@dataclass
class Pipeline:
    retriever: Retriever
    reranker: Reranker
    generator: AnswerGenerator
    condenser: Condenser
    store: ConversationStore
    threshold: float


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="the customer's question")
    parser.add_argument("--all", action="store_true", help="run the demo set")
    parser.add_argument("--chat", action="store_true", help="run the two-turn conversation")
    args = parser.parse_args()

    if not args.query and not args.all and not args.chat:
        parser.error("give a query, or --all, or --chat")

    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    generate = build_gemini(settings)

    pipeline = Pipeline(
        retriever=Retriever(
            qdrant, build_encoder(settings), settings.qdrant_collection, settings.retrieval_top_n
        ),
        reranker=Reranker(
            build_cross_encoder(settings), settings.rerank_top_k, settings.confidence_threshold
        ),
        generator=AnswerGenerator(generate),
        condenser=Condenser(generate),
        store=ConversationStore(redis, settings.conversation_ttl_seconds),
        threshold=settings.confidence_threshold,
    )

    try:
        if args.chat:
            # A fresh id each run, so the demo never reads a previous run's history.
            conversation_id = f"demo-{uuid.uuid4().hex[:8]}"
            print(f"conversation {conversation_id}")
            for query in CHAT_TURNS:
                await ask(pipeline, query, conversation_id=conversation_id)
        else:
            for query in DEMO_QUERIES if args.all else [args.query]:
                await ask(pipeline, query)
    finally:
        await qdrant.close()
        await redis.aclose()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

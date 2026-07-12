"""Ask the bot a question, end to end, against the real stack.

    python scripts/ask.py "where is my order"
    python scripts/ask.py --all      # a fixed set: things it should answer, things it should not

The full pipeline: retrieve, rerank, gate, and either answer with citations or
escalate. This is the first point where the whole thing is visible as a customer
would experience it, and it makes a real Gemini call, so it needs GEMINI_API_KEY.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.config import get_settings  # noqa: E402
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


async def ask(query: str, retriever, reranker, generator, threshold: float) -> None:
    print(f"\n{'=' * 78}\nQ  {query}")

    candidates = await retriever.search(query)
    result = await reranker.rerank(query, candidates)

    if result.escalate:
        top = result.top.doc_id if result.top else "nothing"
        print(f"-> ESCALATED  confidence {result.confidence:.3f} < {threshold} (best guess was {top})")
        return

    try:
        answer = await generator.answer(query, result.ranked)
    except UngroundedAnswer as exc:
        # The gate was confident and the model still would not ground an answer.
        # Escalating here rather than serving something uncited is the point.
        print(f"-> ESCALATED  confidence {result.confidence:.3f}, but ungrounded: {exc}")
        return

    print(f"-> ANSWERED   confidence {result.confidence:.3f}")
    print(f"\n   {answer.text}")
    print(f"\n   sources: {', '.join(answer.citations)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="the customer's question")
    parser.add_argument("--all", action="store_true", help="run the demo set")
    args = parser.parse_args()

    if not args.query and not args.all:
        parser.error("give a query, or --all")

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    retriever = Retriever(
        client, build_encoder(settings), settings.qdrant_collection, settings.retrieval_top_n
    )
    reranker = Reranker(
        build_cross_encoder(settings), settings.rerank_top_k, settings.confidence_threshold
    )
    generator = AnswerGenerator(build_gemini(settings))

    try:
        for query in DEMO_QUERIES if args.all else [args.query]:
            await ask(query, retriever, reranker, generator, settings.confidence_threshold)
    finally:
        await client.close()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

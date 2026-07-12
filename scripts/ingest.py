"""Ingest the KB corpus into Qdrant. Run this whenever the docs change.

    python scripts/ingest.py              # chunk, embed, upsert
    python scripts/ingest.py --dry-run    # chunk only, print the plan, touch nothing
    python scripts/ingest.py --check      # ingest, then prove retrieval works

--check is the honest test. It puts real user phrasings through the same vector
search the bot will use and asserts the right document comes back. Ingestion
that "succeeds" while retrieving the wrong document is the failure mode worth
catching, and a green upsert log will not catch it.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.ingestion import ingest, load_chunks, load_embedder  # noqa: E402
from app.retrieval import Retriever, build_encoder  # noqa: E402

# Phrased the way a user would, not the way the documents are. Each is a
# question the corpus should answer, paired with the document that answers it.
SMOKE_QUERIES: list[tuple[str, str]] = [
    ("where is my order right now", "tracking-your-order"),
    ("i forgot my password and cant log in", "resetting-your-password"),
    ("my card got rejected at checkout", "payment-declined-or-failed"),
    ("how long until my refund shows up", "tracking-your-refund"),
    ("i need to send this back, it arrived broken", "returns-and-damaged-goods"),
    ("can i change the address after ordering", "changing-your-shipping-address"),
    ("do you charge me for cancelling", "cancellation-fees"),
    ("where do i download a copy of my invoice", "downloading-an-invoice"),
]

TOP_K = 3


def dry_run(settings) -> int:
    model = load_embedder(settings)

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.tokenize(text))

    chunks = load_chunks(
        Path(settings.kb_dir),
        count_tokens,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )

    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk.doc_id] = by_doc.get(chunk.doc_id, 0) + 1

    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id:<34} {count:>3} chunks")

    sizes = [count_tokens(chunk.text) for chunk in chunks]
    print(f"\n{len(chunks)} chunks from {len(by_doc)} documents")
    print(f"tokens per chunk: min {min(sizes)}, mean {sum(sizes) // len(sizes)}, max {max(sizes)}")
    print(f"model input limit: {model.max_seq_length}")
    return 0


async def _check(settings) -> int:
    """Query the ingested collection through the real retriever, and grade it.

    Deliberately goes through app.retrieval rather than re-issuing its own
    vector search: a check that passes against a hand-rolled copy of retrieval
    tells you nothing about the retrieval the bot actually runs.
    """
    client = AsyncQdrantClient(url=settings.qdrant_url)
    retriever = Retriever(
        client,
        build_encoder(settings),
        settings.qdrant_collection,
        settings.retrieval_top_n,
    )

    print(f"\nRetrieval check ({len(SMOKE_QUERIES)} queries, expecting a hit in top {TOP_K}):\n")
    failures = 0

    try:
        for query, expected in SMOKE_QUERIES:
            hits = await retriever.search(query, limit=TOP_K)
            ranked = [hit.doc_id for hit in hits]

            if expected in ranked:
                rank = ranked.index(expected) + 1
                print(f"  PASS  rank {rank}  {hits[rank - 1].score:.3f}  {query}")
            else:
                failures += 1
                print(f"  FAIL            {query}")
                print(f"          expected {expected}, got {', '.join(ranked) or 'nothing'}")
    finally:
        await client.close()

    print()
    if failures:
        print(f"{failures}/{len(SMOKE_QUERIES)} queries did not retrieve their document.")
        return 1
    print(f"All {len(SMOKE_QUERIES)} queries retrieved their document.")
    return 0


def check(settings) -> int:
    return asyncio.run(_check(settings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="chunk only, write nothing")
    parser.add_argument("--check", action="store_true", help="run retrieval smoke queries after ingest")
    args = parser.parse_args()

    settings = get_settings()

    if args.dry_run:
        return dry_run(settings)

    stats = ingest(settings)
    print(
        f"ingested {stats['chunks']} chunks from {stats['documents']} documents "
        f"into '{stats['collection']}' ({stats['dimensions']}-dim)"
    )
    if stats["deleted"]:
        print(f"removed {stats['deleted']} stale points from documents that changed or were deleted")

    if args.check:
        return check(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())

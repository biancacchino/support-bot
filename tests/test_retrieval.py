"""Retrieval tests.

Two layers, deliberately:

- The fast ones inject a fake encoder over a hand-built collection. They pin the
  mechanics - ordering, the limit, payload mapping, the empty cases - without a
  model, so a broken payload key fails in milliseconds.
- `test_obvious_query_retrieves_its_document` is the acceptance criterion
  itself, and a fake encoder cannot stand in for it: it runs the real embedding
  model over the real corpus and asserts the right document survives into the
  top 10 of 160-odd chunks. Marked slow; it loads model weights.

Both use an in-memory Qdrant, so neither needs the compose stack running.
"""

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.retrieval import Candidate, Retriever, build_encoder

VECTOR_SIZE = 4
COLLECTION = "test_kb"

# Three chunks, one per axis, so "nearest" is obvious by construction.
FIXTURE_POINTS = [
    {
        "vector": [1.0, 0.0, 0.0, 0.0],
        "payload": {
            "doc_id": "tracking-your-order",
            "title": "Tracking your order",
            "category": "ORDER",
            "intents": ["track_order"],
            "heading": "Where to find your tracking link",
            "chunk_index": 0,
            "text": "Your tracking link appears in the shipping confirmation email.",
        },
    },
    {
        "vector": [0.0, 1.0, 0.0, 0.0],
        "payload": {
            "doc_id": "refund-policy",
            "title": "Refund policy",
            "category": "REFUND",
            "intents": ["check_refund_policy"],
            "heading": "When you are eligible",
            "chunk_index": 0,
            "text": "Refunds are available within 30 days of delivery.",
        },
    },
    {
        "vector": [0.0, 0.0, 1.0, 0.0],
        "payload": {
            "doc_id": "resetting-your-password",
            "title": "Resetting your password",
            "category": "ACCOUNT",
            "intents": ["recover_password"],
            "heading": "Requesting a reset link",
            "chunk_index": 0,
            "text": "Use the forgotten password link on the sign-in page.",
        },
    },
]

# What the fake encoder "understands". Each query lands exactly on one axis.
FAKE_VECTORS = {
    "where is my order": [1.0, 0.0, 0.0, 0.0],
    "can i get a refund": [0.0, 1.0, 0.0, 0.0],
    "i forgot my password": [0.0, 0.0, 1.0, 0.0],
    "something else entirely": [0.0, 0.0, 0.0, 1.0],
}


def fake_encoder(query: str) -> list[float]:
    return FAKE_VECTORS[query]


@pytest.fixture
async def client():
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE, distance=models.Distance.COSINE
        ),
    )
    yield client
    await client.close()


@pytest.fixture
async def populated(client):
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(id=i, vector=p["vector"], payload=p["payload"])
            for i, p in enumerate(FIXTURE_POINTS)
        ],
        wait=True,
    )
    return client


def make_retriever(client, top_n: int = 10) -> Retriever:
    return Retriever(client, fake_encoder, COLLECTION, top_n)


# --- mechanics --------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_nearest_first(populated):
    hits = await make_retriever(populated).search("can i get a refund")

    assert hits[0].doc_id == "refund-policy"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.asyncio
async def test_search_maps_the_whole_payload(populated):
    top = (await make_retriever(populated).search("where is my order"))[0]

    assert top == Candidate(
        doc_id="tracking-your-order",
        title="Tracking your order",
        category="ORDER",
        intents=("track_order",),
        heading="Where to find your tracking link",
        chunk_index=0,
        text="Your tracking link appears in the shipping confirmation email.",
        score=top.score,
    )
    assert top.source == "tracking-your-order"


@pytest.mark.asyncio
async def test_search_caps_at_top_n(populated):
    assert len(await make_retriever(populated, top_n=2).search("where is my order")) == 2


@pytest.mark.asyncio
async def test_search_limit_overrides_top_n(populated):
    hits = await make_retriever(populated, top_n=10).search("where is my order", limit=1)

    assert len(hits) == 1


@pytest.mark.asyncio
async def test_search_returns_everything_it_has_when_short(populated):
    """Fewer chunks than top_n is not an error - the reranker just gets less."""
    hits = await make_retriever(populated, top_n=10).search("where is my order")

    assert len(hits) == len(FIXTURE_POINTS)


@pytest.mark.asyncio
async def test_unrelated_query_still_returns_candidates(populated):
    """Stage 1 recalls, it does not judge.

    A query orthogonal to every chunk still comes back with candidates, because
    filtering these out is the reranker's job in Phase 3 and thresholding here
    on cosine similarity is the exact mistake the confidence gate exists to
    avoid.
    """
    hits = await make_retriever(populated).search("something else entirely")

    assert hits != []


@pytest.mark.asyncio
async def test_empty_collection_returns_no_candidates(client):
    assert await make_retriever(client).search("where is my order") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_is_rejected(populated, query):
    with pytest.raises(ValueError, match="empty query"):
        await make_retriever(populated).search(query)


# --- the acceptance criterion ----------------------------------------------


# The user's words, not the corpus's. Each is a question the KB answers, paired
# with the document that answers it.
OBVIOUS_QUERIES = [
    ("where is my order right now", "tracking-your-order"),
    ("i forgot my password and cant log in", "resetting-your-password"),
    ("my card got rejected at checkout", "payment-declined-or-failed"),
    ("i need to send this back, it arrived broken", "returns-and-damaged-goods"),
    ("do you charge me for cancelling", "cancellation-fees"),
]


@pytest.fixture(scope="module")
def encoded_corpus():
    """The real KB, chunked and embedded once with the real model.

    Module-scoped because loading MiniLM and encoding 160-odd chunks costs
    seconds and every query below can share one copy of the result.
    """
    pytest.importorskip("sentence_transformers")

    from pathlib import Path

    from app.ingestion import load_chunks, load_embedder

    settings = Settings(kb_dir=str(Path(__file__).resolve().parent.parent / "kb"))
    model = load_embedder(settings)

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.tokenize(text))

    chunks = load_chunks(
        Path(settings.kb_dir),
        count_tokens,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    vectors = model.encode(
        [chunk.embed_text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return settings, chunks, vectors, build_encoder(settings)


@pytest.fixture
async def real_retriever(encoded_corpus):
    """A Retriever over the whole corpus, in an in-memory Qdrant."""
    settings, chunks, vectors, encoder = encoded_corpus

    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE
        ),
    )
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=chunk.point_id, vector=vector.tolist(), payload=chunk.payload()
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
        wait=True,
    )

    yield Retriever(client, encoder, COLLECTION, settings.retrieval_top_n), len(chunks)
    await client.close()


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected_doc"), OBVIOUS_QUERIES)
async def test_obvious_query_retrieves_its_document(real_retriever, query, expected_doc):
    """Phase 2 acceptance: the obviously-correct doc is in the top 10.

    Rank within those 10 is deliberately not asserted. Raw cosine similarity
    puts near-neighbours (delivery vs shipping, cancelling vs cancellation fees)
    ahead of the right document often enough that pinning rank here would be
    pinning a number we already know is wrong. Getting it to rank 1 is what the
    Phase 3 reranker is for; recall is all stage 1 owes.
    """
    retriever, total_chunks = real_retriever

    hits = await retriever.search(query)
    retrieved = [hit.doc_id for hit in hits]

    assert expected_doc in retrieved, (
        f"{expected_doc!r} missing from the top {len(hits)} of {total_chunks} "
        f"chunks; got {retrieved}"
    )

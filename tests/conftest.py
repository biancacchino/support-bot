"""Shared fixtures.

The expensive things - loading MiniLM, loading the cross-encoder, embedding the
162-chunk corpus - happen once per session. Everything downstream of them is
cheap, so each test still gets a clean in-memory Qdrant of its own.

Nothing here talks to the compose stack. These run against an in-memory Qdrant,
so the suite does not need a live server, only the model weights.
"""

from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.reranker import Reranker
from app.retrieval import Retriever

COLLECTION = "test_kb"

KB_DIR = Path(__file__).resolve().parent.parent / "kb"

# Real questions the KB genuinely answers. The gate must not escalate these:
# every one it escalates is a deflection the product does not get.
IN_SCOPE_QUERIES = [
    ("where is my order right now", "tracking-your-order"),
    ("i forgot my password and cant log in", "resetting-your-password"),
    ("my card got rejected at checkout", "payment-declined-or-failed"),
    ("how long until my refund shows up", "tracking-your-refund"),
    ("i need to send this back, it arrived broken", "returns-and-damaged-goods"),
    ("can i change the address after ordering", "changing-your-shipping-address"),
    ("do you charge me for cancelling", "cancellation-fees"),
    ("where do i download a copy of my invoice", "downloading-an-invoice"),
]

# Questions the KB cannot answer, phrased in the KB's own vocabulary so they
# land near real documents in embedding space. The gate must escalate all of
# these. The first four are the out-of-scope Bitext intents (CONTACT, FEEDBACK,
# SUBSCRIPTION); the last two are out-of-domain entirely.
OFF_TOPIC_QUERIES = [
    "i want to speak to a human agent right now",
    "i want to file a complaint about your service",
    "sign me up for your newsletter",
    "how do i leave a review of the product i bought",
    "can i pay for my order in bitcoin",
    "does my order come with gift wrapping",
    "what is the ceo of northwind's salary",
    "i need to renew my health insurance policy",
    "how do i book an appointment with my doctor",
]


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings from the code defaults, with the developer's .env shut out.

    `_env_file=None` is load-bearing. pydantic-settings lets an .env override a
    code default, which is right for the app and wrong for the suite: it means a
    stale value in an untracked local file changes what the tests conclude. It
    already did. A .env still pinning the old CONFIDENCE_THRESHOLD=0.5 made the
    gate escalate a genuine query and failed a Phase 3 test that had nothing to
    do with the change being made.

    Tests assert against the defaults the repo ships. Whatever is in someone's
    .env is their business and not the suite's.
    """
    return Settings(_env_file=None, kb_dir=str(KB_DIR))


@pytest.fixture(scope="session")
def corpus(settings):
    """The real KB, chunked and embedded once with the real model."""
    pytest.importorskip("sentence_transformers")

    from app.ingestion import load_chunks, load_embedder
    from app.retrieval import build_encoder

    model = load_embedder(settings)

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.tokenize(text))

    chunks = load_chunks(
        KB_DIR,
        count_tokens,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    vectors = model.encode(
        [chunk.embed_text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return chunks, vectors, build_encoder(settings)


@pytest.fixture(scope="session")
def cross_encode(settings):
    """The real cross-encoder, loaded once."""
    pytest.importorskip("sentence_transformers")

    from app.reranker import build_cross_encoder

    return build_cross_encoder(settings)


@pytest.fixture
async def real_retriever(settings, corpus):
    """A Retriever over the whole corpus, in an in-memory Qdrant."""
    chunks, vectors, encoder = corpus

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

    yield Retriever(client, encoder, COLLECTION, settings.retrieval_top_n)
    await client.close()


@pytest.fixture
def real_reranker(settings, cross_encode) -> Reranker:
    return Reranker(cross_encode, settings.rerank_top_k, settings.confidence_threshold)

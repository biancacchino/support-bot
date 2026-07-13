"""Stage 1 retrieval: embed the query, vector search Qdrant, return candidates.

This is deliberately only the recall stage. It casts a wide net (top ~10) and
does not decide anything: the score it returns is raw cosine similarity, which
is a similarity signal and *not* a confidence signal. An off-topic query can sit
at high cosine similarity to a document it has nothing to do with, which is
exactly why the confidence gate is scored on the cross-encoder in Phase 3 rather
than here. Nothing downstream should threshold on `Candidate.score`.

The encoder is injected rather than constructed here so tests can pin ordering
and payload mapping with a fake, and so the model is loaded once at startup
instead of once per query.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from anyio import to_thread
from qdrant_client import AsyncQdrantClient

from app.config import Settings

logger = logging.getLogger(__name__)

# Takes a query string, returns a normalized embedding.
Encoder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class Candidate:
    """One retrieved chunk, with the metadata a reranker or a human needs."""

    doc_id: str
    title: str
    category: str
    intents: tuple[str, ...]
    heading: str
    chunk_index: int
    text: str
    score: float

    @classmethod
    def from_point(cls, point) -> Candidate:
        payload = point.payload or {}
        return cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            category=payload["category"],
            intents=tuple(payload.get("intents", ())),
            heading=payload["heading"],
            chunk_index=payload["chunk_index"],
            text=payload["text"],
            score=point.score,
        )

    @property
    def passage(self) -> str:
        """The chunk as the reranker should read it, with its context restored.

        Same shape as the text ingestion embeds, and for the same reason: a
        chunk in isolation is often ambiguous ("That link works for 60 days"
        does not say what link). Handing the cross-encoder the bare text makes
        it judge a passage whose subject is missing, and it marks genuine
        matches down for it.
        """
        return f"{self.title} > {self.heading}\n\n{self.text}"


class Retriever:
    """Embeds a query and pulls the nearest chunks out of Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        encoder: Encoder,
        collection: str,
        top_n: int,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._collection = collection
        self._top_n = top_n

    async def search(self, query: str, limit: int | None = None) -> list[Candidate]:
        """Return the `limit` (default `top_n`) nearest chunks, best first."""
        query = query.strip()
        if not query:
            raise ValueError("cannot retrieve against an empty query")

        limit = self._top_n if limit is None else limit

        # encode() is CPU-bound and holds the GIL for tens of milliseconds. On
        # the event loop that stalls every other in-flight request, so it goes
        # to a worker thread.
        vector = await to_thread.run_sync(self._encoder, query)

        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )

        candidates = [Candidate.from_point(point) for point in response.points]
        logger.debug(
            "retrieved %d candidates for %r (top score %.3f)",
            len(candidates),
            query,
            candidates[0].score if candidates else float("nan"),
        )
        return candidates


def build_encoder(settings: Settings) -> Encoder:
    """Load the embedding model and return a callable that encodes one query.

    Must match ingestion: the same model, and normalized vectors, or the cosine
    distances the collection was built with stop meaning anything.
    """
    from app.ingestion import load_embedder

    model = load_embedder(settings)

    def encode(query: str) -> Sequence[float]:
        # show_progress_bar=False or sentence-transformers prints a tqdm bar to the
        # log stream on every single query, which is noise in a terminal and garbage
        # in a log aggregator.
        return model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    return encode

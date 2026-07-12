"""Stage 2: rerank the candidates, then decide whether we are confident enough.

Stage 1 recalls broadly on cosine similarity between a query vector and a chunk
vector, which never lets the two look at each other. A cross-encoder reads the
query and the chunk *together*, so it can tell "how do I cancel my order" from
"what does cancelling cost" - a distinction the bi-encoder collapses.

That is why the confidence gate is scored here and not on stage 1. Cosine
similarity is a similarity signal: an off-topic query can sit at high similarity
to a document it has nothing to do with, and a similarity-only gate answers it
confidently and wrongly. The rerank score is the only thing in the pipeline that
has actually compared the question to the passage.

Cross-encoder outputs are unbounded logits. They are squashed through a sigmoid
so the gate can be a plain 0-1 probability, which is what CONFIDENCE_THRESHOLD
means.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from anyio import to_thread

from app.config import Settings
from app.retrieval import Candidate

logger = logging.getLogger(__name__)

# Takes (query, passage) pairs, returns one relevance logit per pair.
CrossEncoderFn = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


def sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


@dataclass(frozen=True)
class Ranked:
    """A candidate the cross-encoder has actually read, with its verdict."""

    candidate: Candidate
    score: float  # 0-1. Relevance of this chunk to this query.

    @property
    def doc_id(self) -> str:
        return self.candidate.doc_id


@dataclass(frozen=True)
class RerankResult:
    """What stage 2 hands to answer generation, or to a human."""

    ranked: list[Ranked]
    confidence: float
    escalate: bool

    @property
    def top(self) -> Ranked | None:
        return self.ranked[0] if self.ranked else None


class Reranker:
    """Reorders stage-1 candidates and gates on how good the best one is."""

    def __init__(
        self,
        cross_encode: CrossEncoderFn,
        top_k: int,
        threshold: float,
    ) -> None:
        self._cross_encode = cross_encode
        self._top_k = top_k
        self._threshold = threshold

    async def rerank(self, query: str, candidates: Sequence[Candidate]) -> RerankResult:
        """Score every candidate against the query, keep the best `top_k`."""
        if not candidates:
            # Nothing retrieved is not a confident answer, it is no answer. An
            # empty KB or a failed search escalates rather than inviting the LLM
            # to fill the silence.
            return RerankResult(ranked=[], confidence=0.0, escalate=True)

        pairs = [(query, candidate.passage) for candidate in candidates]

        # Same reasoning as the embedder: this is CPU-bound and would otherwise
        # block the event loop for every other request in flight.
        logits = await to_thread.run_sync(self._cross_encode, pairs)

        ranked = sorted(
            (
                Ranked(candidate=candidate, score=sigmoid(float(logit)))
                for candidate, logit in zip(candidates, logits, strict=True)
            ),
            key=lambda r: r.score,
            reverse=True,
        )[: self._top_k]

        # Confidence is the best chunk's score, not a mean over the top k.
        # Averaging would let three weak-but-plausible chunks drown out the one
        # chunk that actually answers the question.
        confidence = ranked[0].score
        escalate = confidence < self._threshold

        logger.debug(
            "reranked %d candidates for %r: top %s at %.3f, escalate=%s",
            len(candidates),
            query,
            ranked[0].doc_id,
            confidence,
            escalate,
        )
        return RerankResult(ranked=ranked, confidence=confidence, escalate=escalate)


def build_cross_encoder(settings: Settings) -> CrossEncoderFn:
    """Load the cross-encoder and return a callable that scores query/passage pairs."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.reranker_model)

    def cross_encode(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return model.predict(list(pairs), show_progress_bar=False).tolist()

    return cross_encode

"""Chunk the KB corpus, embed it locally, and upsert it into Qdrant.

Chunking is a pure function over a token-counting callable rather than over the
embedding model itself. That keeps the part with all the edge cases (long
paragraphs, overlap, section boundaries) testable without downloading a few
hundred MB of model weights.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models

from app.config import Settings

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

# Fixed namespace so a chunk's point ID is a pure function of (doc_id, index).
# Re-ingesting the same corpus overwrites the same points instead of growing a
# second copy of the collection alongside the first.
CHUNK_NAMESPACE = uuid.UUID("b8f0a0e2-1c9a-4b1e-9a3a-6c0d4f2e7a11")

REQUIRED_FRONTMATTER = ("title", "doc_id", "category", "intents")

# Tokens held back from the chunk budget for the "title > heading" prefix that
# every chunk is embedded under. See Chunk.embed_text.
PREFIX_TOKEN_RESERVE = 32

HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    category: str
    intents: tuple[str, ...]
    heading: str
    chunk_index: int
    text: str

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(CHUNK_NAMESPACE, f"{self.doc_id}:{self.chunk_index}"))

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        A chunk read in isolation is often ambiguous: "That link works for 60
        days, then expires" does not say what link, or what it tracks. Embedding
        the chunk under its document and section titles gives the vector the
        subject the prose leaves implicit.
        """
        return f"{self.title} > {self.heading}\n\n{self.text}"

    def payload(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "intents": list(self.intents),
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


def parse_doc(path: Path) -> tuple[dict, str]:
    """Split a KB file into its frontmatter mapping and its markdown body."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")

    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter) or {}

    missing = [f for f in REQUIRED_FRONTMATTER if not meta.get(f)]
    if missing:
        raise ValueError(f"{path.name}: missing frontmatter field(s): {', '.join(missing)}")
    if meta["doc_id"] != path.stem:
        raise ValueError(f"{path.name}: doc_id '{meta['doc_id']}' does not match filename")

    return meta, body.strip()


def split_sections(body: str, title: str) -> list[tuple[str, str]]:
    """Split a body into (heading, text) pairs on its markdown headings.

    Headings are the strongest topic boundary the corpus gives us, so chunks
    never straddle two of them. Text before the first heading is attributed to
    the document title.
    """
    sections: list[tuple[str, str]] = []
    heading = title
    buffer: list[str] = []

    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)

    if "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))

    return sections


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Last resort: split on whitespace for a unit no boundary can break up."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []

    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) > max_tokens:
            if len(current) == 1:
                pieces.append(current.pop())  # single word over budget; take it
            else:
                current.pop()
                pieces.append(" ".join(current))
                current = [word]

    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_oversized(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Break a paragraph that will not fit, on sentence boundaries where it can."""
    pieces: list[str] = []
    current: list[str] = []

    for sentence in SENTENCE_RE.split(text):
        if count_tokens(sentence) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_hard_split(sentence, count_tokens, max_tokens))
            continue

        current.append(sentence)
        if count_tokens(" ".join(current)) > max_tokens:
            current.pop()
            pieces.append(" ".join(current))
            current = [sentence]

    if current:
        pieces.append(" ".join(current))
    return pieces


def _trailing_overlap(
    units: list[str], count_tokens: TokenCounter, overlap_tokens: int
) -> list[str]:
    """The tail of the previous chunk, to be repeated at the head of the next."""
    if overlap_tokens <= 0:
        return []

    carried: list[str] = []
    for unit in reversed(units):
        candidate = [unit, *carried]
        if count_tokens("\n\n".join(candidate)) > overlap_tokens:
            break
        carried = candidate
    return carried


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Pack a section into chunks of at most `max_tokens`, overlapping slightly.

    Paragraphs are kept whole wherever they fit. The overlap exists so that an
    answer spanning a paragraph break is still fully present in at least one
    chunk rather than being split across two that each retrieve on their own.
    """
    units: list[str] = []
    for paragraph in PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_oversized(paragraph, count_tokens, max_tokens))

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        if current and count_tokens("\n\n".join([*current, unit])) > max_tokens:
            chunks.append("\n\n".join(current))
            current = [*_trailing_overlap(current, count_tokens, overlap_tokens), unit]
            # The carried overlap plus the new unit can itself overflow; shed
            # overlap oldest-first until it fits. The unit always survives.
            while len(current) > 1 and count_tokens("\n\n".join(current)) > max_tokens:
                current.pop(0)
        else:
            current.append(unit)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def load_chunks(
    kb_dir: Path,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Read every KB document and chunk it. Pure: no model, no network."""
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise ValueError(f"no markdown documents found in {kb_dir}")

    chunks: list[Chunk] = []
    for path in paths:
        meta, body = parse_doc(path)
        index = 0
        for heading, section in split_sections(body, meta["title"]):
            for piece in chunk_text(section, count_tokens, max_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        doc_id=meta["doc_id"],
                        title=meta["title"],
                        category=meta["category"],
                        intents=tuple(meta["intents"]),
                        heading=heading,
                        chunk_index=index,
                        text=piece,
                    )
                )
                index += 1
    return chunks


def load_embedder(settings: Settings):
    """Load the local sentence-transformers model.

    Imported lazily: it pulls in torch, which is slow enough that the chunking
    tests would notice.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    if client.collection_exists(collection):
        existing = client.get_collection(collection).config.params.vectors.size
        if existing != dim:
            raise ValueError(
                f"collection '{collection}' has {existing}-dim vectors but the "
                f"embedding model produces {dim}. Delete the collection and re-ingest."
            )
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Filtering by category or intent is how the eval joins queries to expected
    # documents, and how retrieval gets scoped later. Unindexed payload filters
    # in Qdrant fall back to a full scan.
    for field in ("doc_id", "category", "intents"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def prune_stale(client: QdrantClient, collection: str, live: Iterable[Chunk]) -> int:
    """Delete points that the current corpus no longer produces.

    Upserting alone is not enough: a deleted document, or a document that lost a
    section, leaves orphaned points behind that still answer queries.
    """
    live_ids = {chunk.point_id for chunk in live}
    stale: list[str] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        stale.extend(str(p.id) for p in points if str(p.id) not in live_ids)
        if offset is None:
            break

    if stale:
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=stale),
        )
    return len(stale)


def ingest(settings: Settings, client: QdrantClient | None = None) -> dict:
    """Chunk, embed and upsert the whole corpus. Safe to re-run."""
    model = load_embedder(settings)
    model_limit = model.max_seq_length
    budget = settings.chunk_size_tokens

    if budget + PREFIX_TOKEN_RESERVE > model_limit:
        raise ValueError(
            f"chunk_size_tokens={budget} plus a {PREFIX_TOKEN_RESERVE}-token prefix "
            f"exceeds the {model_limit}-token input limit of {settings.embedding_model}. "
            "The encoder would silently truncate every oversized chunk, so the tail of "
            f"it would be stored but never embedded. Lower it to "
            f"{model_limit - PREFIX_TOKEN_RESERVE} or less."
        )

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.tokenize(text))

    chunks = load_chunks(Path(settings.kb_dir), count_tokens, budget, settings.chunk_overlap_tokens)

    # The reserve is a guess about title+heading length. Verify it rather than
    # trust it, because being wrong here costs silent truncation.
    for chunk in chunks:
        length = count_tokens(chunk.embed_text)
        if length > model_limit:
            raise ValueError(
                f"{chunk.doc_id} chunk {chunk.chunk_index} is {length} tokens once "
                f"prefixed, over the model's {model_limit}. Raise PREFIX_TOKEN_RESERVE."
            )

    client = client or QdrantClient(url=settings.qdrant_url)
    vectors = model.encode(
        [chunk.embed_text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ensure_collection(client, settings.qdrant_collection, len(vectors[0]))
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=chunk.point_id,
                vector=vector.tolist(),
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
        wait=True,
    )
    deleted = prune_stale(client, settings.qdrant_collection, chunks)

    docs = {chunk.doc_id for chunk in chunks}
    logger.info(
        "ingested %d chunks from %d documents (%d stale points removed)",
        len(chunks),
        len(docs),
        deleted,
    )
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "deleted": deleted,
        "collection": settings.qdrant_collection,
        "dimensions": len(vectors[0]),
    }

"""Chunking tests.

These use a whitespace token counter, not the real tokenizer. The point is to
pin the chunking logic (boundaries, overlap, oversized paragraphs), and a fake
counter makes the expected numbers legible. The real tokenizer's agreement with
the model's input limit is enforced in ingest() and exercised by
`scripts/ingest.py --check`.
"""

import pytest

from app.ingestion import (
    Chunk,
    chunk_text,
    load_chunks,
    parse_doc,
    split_sections,
)


def count_words(text: str) -> int:
    return len(text.split())


def write_doc(tmp_path, name: str, frontmatter: str, body: str):
    path = tmp_path / f"{name}.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    return path


# --- frontmatter ------------------------------------------------------------


def test_parse_doc_reads_frontmatter_and_body(tmp_path):
    path = write_doc(
        tmp_path,
        "tracking-your-order",
        "title: Tracking your order\ndoc_id: tracking-your-order\n"
        "category: ORDER\nintents: [track_order]",
        "# Tracking your order\n\nBody text.",
    )
    meta, body = parse_doc(path)

    assert meta["doc_id"] == "tracking-your-order"
    assert meta["intents"] == ["track_order"]
    assert body.startswith("# Tracking your order")


def test_parse_doc_rejects_missing_frontmatter(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("# No frontmatter here", encoding="utf-8")

    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        parse_doc(path)


def test_parse_doc_rejects_missing_field(tmp_path):
    path = write_doc(tmp_path, "x", "title: X\ndoc_id: x\ncategory: ORDER", "Body.")

    with pytest.raises(ValueError, match="intents"):
        parse_doc(path)


def test_parse_doc_rejects_doc_id_that_does_not_match_filename(tmp_path):
    # The eval joins on doc_id. A mismatch silently scores every query for the
    # document as a miss, so it has to fail here instead.
    path = write_doc(
        tmp_path,
        "tracking-your-order",
        "title: T\ndoc_id: tracking-orders\ncategory: ORDER\nintents: [track_order]",
        "Body.",
    )
    with pytest.raises(ValueError, match="does not match filename"):
        parse_doc(path)


# --- sections ---------------------------------------------------------------


def test_split_sections_breaks_on_headings():
    body = "# Doc\n\nIntro line.\n\n## First\n\nOne.\n\n## Second\n\nTwo."
    sections = split_sections(body, "Doc")

    assert sections == [("Doc", "Intro line."), ("First", "One."), ("Second", "Two.")]


def test_split_sections_attributes_leading_text_to_the_title():
    sections = split_sections("Text before any heading.", "My title")
    assert sections == [("My title", "Text before any heading.")]


def test_split_sections_drops_empty_sections():
    body = "## Empty\n\n## Full\n\nContent."
    assert split_sections(body, "Doc") == [("Full", "Content.")]


# --- chunking ---------------------------------------------------------------


def test_chunks_stay_within_budget():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, count_words, max_tokens=50, overlap_tokens=10)

    assert chunks
    assert all(count_words(chunk) <= 50 for chunk in chunks)


def test_short_section_stays_one_chunk():
    chunks = chunk_text("Just a short line.", count_words, 50, 10)
    assert chunks == ["Just a short line."]


def test_paragraphs_are_kept_whole_when_they_fit():
    text = "alpha one two\n\nbeta three four\n\ngamma five six"
    chunks = chunk_text(text, count_words, max_tokens=100, overlap_tokens=0)

    assert chunks == [text]


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    text = "aaa\n\nbbb\n\nccc\n\nddd"
    chunks = chunk_text(text, count_words, max_tokens=2, overlap_tokens=1)

    assert len(chunks) > 1
    # Every chunk after the first opens with the last unit of the one before it.
    for previous, following in zip(chunks, chunks[1:]):
        assert following.split("\n\n")[0] == previous.split("\n\n")[-1]


def test_no_overlap_when_disabled():
    text = "aaa\n\nbbb\n\nccc\n\nddd"
    chunks = chunk_text(text, count_words, max_tokens=2, overlap_tokens=0)

    assert chunks == ["aaa\n\nbbb", "ccc\n\nddd"]


def test_oversized_paragraph_is_split_on_sentences():
    text = "One two three. Four five six. Seven eight nine."
    chunks = chunk_text(text, count_words, max_tokens=6, overlap_tokens=0)

    assert all(count_words(chunk) <= 6 for chunk in chunks)
    assert "".join(chunks).count("three") == 1


def test_sentence_longer_than_the_budget_is_hard_split():
    # No sentence boundary to use, so it falls back to whitespace. Nothing may
    # exceed the budget, because the encoder would truncate whatever does.
    text = "word " * 40
    chunks = chunk_text(text.strip(), count_words, max_tokens=10, overlap_tokens=0)

    assert all(count_words(chunk) <= 10 for chunk in chunks)
    assert sum(count_words(chunk) for chunk in chunks) == 40


def test_content_survives_chunking():
    text = "\n\n".join(f"unique{i} " + "filler " * 20 for i in range(8))
    chunks = chunk_text(text, count_words, max_tokens=40, overlap_tokens=8)

    joined = " ".join(chunks)
    for i in range(8):
        assert f"unique{i}" in joined


# --- chunk identity ---------------------------------------------------------


def test_point_id_is_stable_across_runs():
    def make():
        return Chunk("doc", "T", "ORDER", ("track_order",), "H", 3, "text").point_id

    assert make() == make()


def test_point_id_differs_per_chunk_and_per_doc():
    a = Chunk("doc-a", "T", "ORDER", (), "H", 0, "x").point_id
    b = Chunk("doc-a", "T", "ORDER", (), "H", 1, "x").point_id
    c = Chunk("doc-b", "T", "ORDER", (), "H", 0, "x").point_id

    assert len({a, b, c}) == 3


def test_embed_text_carries_the_document_and_section_titles():
    chunk = Chunk("d", "Tracking your order", "ORDER", (), "If tracking stalls", 0, "Body.")

    assert chunk.embed_text == "Tracking your order > If tracking stalls\n\nBody."


# --- corpus -----------------------------------------------------------------


def test_load_chunks_indexes_each_document_from_zero(tmp_path):
    for name in ("a-doc", "b-doc"):
        write_doc(
            tmp_path,
            name,
            f"title: {name}\ndoc_id: {name}\ncategory: ORDER\nintents: [track_order]",
            "## One\n\nFirst.\n\n## Two\n\nSecond.",
        )

    chunks = load_chunks(tmp_path, count_words, max_tokens=50, overlap_tokens=0)

    for doc_id in ("a-doc", "b-doc"):
        indexes = [c.chunk_index for c in chunks if c.doc_id == doc_id]
        assert indexes == list(range(len(indexes)))


def test_load_chunks_carries_metadata_onto_every_chunk(tmp_path):
    write_doc(
        tmp_path,
        "refund-policy",
        "title: Refund policy\ndoc_id: refund-policy\ncategory: REFUND\n"
        "intents: [check_refund_policy, get_refund]",
        "## A\n\nFirst.\n\n## B\n\nSecond.",
    )

    chunks = load_chunks(tmp_path, count_words, 50, 0)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.category == "REFUND"
        assert chunk.intents == ("check_refund_policy", "get_refund")
        assert chunk.payload()["intents"] == ["check_refund_policy", "get_refund"]


def test_load_chunks_rejects_an_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no markdown documents"):
        load_chunks(tmp_path, count_words, 50, 0)

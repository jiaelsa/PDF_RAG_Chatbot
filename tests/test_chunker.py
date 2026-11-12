import pytest

from rag.chunker import chunk_pages, split_text
from rag.models import PageText


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i} talks about retrieval." for i in range(60))
    chunks = split_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert "Sentence number 0" in chunks[0]
    assert "Sentence number 59" in chunks[-1]


def test_overlap_shared_between_neighbours():
    text = " ".join(f"Fact {i} is important." for i in range(50))
    chunks = split_text(text, chunk_size=200, overlap=60)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1]
        assert last_sentence.split()[0] in b  # some tail of `a` reappears in `b`


def test_long_token_is_hard_split():
    chunks = split_text("x" * 2500, chunk_size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) == 2500


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_chunk_pages_keeps_page_numbers_and_unique_ids():
    pages = [PageText(1, "Alpha " * 300), PageText(5, "Beta sentence here. " * 100)]
    chunks = chunk_pages(pages, "d" * 64, "doc.pdf", 500, 100)
    assert {c.page for c in chunks} == {1, 5}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.doc_name == "doc.pdf" for c in chunks)

import pytest

from rag.ingestion import ingest_pdf_bytes
from rag.pdf_loader import NoTextError, PDFLoadError, load_pdf_bytes
from tests.helpers import SAMPLE_PAGES, make_pdf


def test_loader_extracts_pages_in_order():
    pages = load_pdf_bytes(make_pdf(SAMPLE_PAGES))
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Retrieval-Augmented Generation" in pages[0].text
    assert "ChromaDB" in pages[1].text


def test_loader_rejects_garbage():
    with pytest.raises(PDFLoadError):
        load_pdf_bytes(b"this is not a pdf")


def test_loader_blank_pdf_raises_no_text():
    with pytest.raises(NoTextError):
        load_pdf_bytes(make_pdf([" ", " "]))


def test_end_to_end_ingest_and_search(store, embedder):
    data = make_pdf(SAMPLE_PAGES)
    res = ingest_pdf_bytes(data, "rag_intro.pdf", store, embedder, chunk_size=400, chunk_overlap=80)
    assert not res.skipped and res.pages == 3 and res.chunks >= 3
    assert store.count() == res.chunks

    hits = store.query(embedder.embed_query("What does cosine similarity measure for embeddings?"), top_k=3)
    assert hits and hits[0].chunk.page == 2
    assert hits[0].chunk.doc_name == "rag_intro.pdf"
    assert hits[0].score >= hits[-1].score


def test_duplicate_upload_is_skipped(store, embedder):
    data = make_pdf(SAMPLE_PAGES)
    ingest_pdf_bytes(data, "a.pdf", store, embedder)
    n = store.count()
    again = ingest_pdf_bytes(data, "a.pdf", store, embedder)
    assert again.skipped and store.count() == n


def test_list_filter_and_delete(store, embedder):
    r1 = ingest_pdf_bytes(make_pdf(SAMPLE_PAGES), "one.pdf", store, embedder)
    r2 = ingest_pdf_bytes(make_pdf(["Completely different topic about astronomy and planets."]), "two.pdf", store, embedder)
    docs = {d.doc_name: d for d in store.list_documents()}
    assert set(docs) == {"one.pdf", "two.pdf"} and docs["one.pdf"].pages == 3

    hits = store.query(embedder.embed_query("retrieval"), top_k=5, doc_ids=[r2.doc_id])
    assert {h.chunk.doc_name for h in hits} == {"two.pdf"}

    store.delete_document(r1.doc_id)
    assert {d.doc_name for d in store.list_documents()} == {"two.pdf"}


def test_empty_store_query(store, embedder):
    assert store.query(embedder.embed_query("anything")) == []

"""Launches the Streamlit script headlessly with a fake embedder / no LLM key."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from rag.ingestion import ingest_pdf_bytes  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402
from tests.helpers import SAMPLE_PAGES, FakeEmbedder, make_pdf  # noqa: E402


def test_app_answers_with_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr("rag.embedder.get_embedder", lambda *a, **k: FakeEmbedder())

    store = VectorStore(str(tmp_path / "c"), "pdf_chunks")
    ingest_pdf_bytes(make_pdf(SAMPLE_PAGES), "rag_intro.pdf", store, FakeEmbedder())

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("What is cosine similarity?").run()
    assert not at.exception
    assert len(at.chat_message) >= 2
    assert any("Sources" in e.label for e in at.expander)

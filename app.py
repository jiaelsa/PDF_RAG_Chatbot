"""Streamlit front-end:  streamlit run app.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

# Streamlit Cloud / HF Spaces: copy secrets into environment variables so that the
# same code works locally (.env) and when deployed (secrets).
try:
    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from rag.config import get_settings  # noqa: E402
from rag.embedder import get_embedder  # noqa: E402
from rag.ingestion import ingest_pdf_bytes  # noqa: E402
from rag.llm import LLMClient  # noqa: E402
from rag.pdf_loader import PDFLoadError  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

st.set_page_config(page_title="PDF Q&A Bot (RAG)", page_icon="📄", layout="wide")


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Opening vector index…")
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, s.collection_name)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)…")
def get_embed():
    return get_embedder()


@st.cache_resource
def get_llm() -> LLMClient:
    return LLMClient()


store = get_store()
embedder = get_embed()
llm = get_llm()
pipeline = RAGPipeline(Retriever(store, embedder), llm)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Documents")
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("Index uploaded PDFs", type="primary", disabled=not files, use_container_width=True):
        bar = st.progress(0.0)
        for f in files:
            label = st.empty()
            try:
                res = ingest_pdf_bytes(
                    f.getvalue(),
                    f.name,
                    store,
                    embedder,
                    progress=lambda stage, frac, n=f.name: (
                        label.caption(f"{n}: {stage}"),
                        bar.progress(min(frac, 1.0)),
                    ),
                )
                if res.skipped:
                    st.info(f"{f.name}: already indexed")
                else:
                    st.success(f"{f.name}: {res.pages} pages → {res.chunks} chunks")
            except PDFLoadError as exc:
                st.error(f"{f.name}: {exc}")
            except Exception as exc:  # keep the app alive on unexpected failures
                st.error(f"{f.name}: unexpected error – {exc}")
            label.empty()
        bar.empty()
        st.session_state.uploader_key += 1  # clears the uploader
        st.rerun()

    docs = store.list_documents()
    st.subheader(f"Indexed ({len(docs)})")
    if not docs:
        st.caption("Nothing indexed yet. Upload a PDF to begin.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d.doc_name}**  \n<small>{d.pages} pages · {d.chunks} chunks</small>", unsafe_allow_html=True)
        if c2.button("🗑", key=f"del_{d.doc_id}", help="Remove from index"):
            store.delete_document(d.doc_id)
            st.rerun()

    st.divider()
    st.subheader("Settings")
    names = [d.doc_name for d in docs]
    scope = st.multiselect("Search only in", names, placeholder="All documents")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
    if llm.configured:
        st.caption(f"🟢 LLM: `{llm.settings.model}`")
    else:
        st.caption("🟡 No LLM key set – answers fall back to the best passage. See README.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- main
st.title("📄 PDF Document Q&A Bot")
st.caption("Retrieval-Augmented Generation: answers come only from your documents, with page-level sources.")


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            badge = "✅ cited" if s.cited else "retrieved"
            st.markdown(f"**[{s.index}] {s.doc_name} — page {s.page}** · similarity {s.score:.2f} · {badge}")
            st.caption(s.snippet)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if not docs:
    st.info("👈 Upload and index at least one PDF, then ask a question.")

question = st.chat_input("Ask a question about your documents…", disabled=not docs)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                ids = [d.doc_id for d in docs if d.doc_name in scope] or None
                ans = pipeline.ask(question, top_k=top_k, doc_ids=ids)
                text, sources = ans.text, ans.sources
            except Exception as exc:
                text, sources = f"⚠️ Something went wrong: {exc}", []
        st.markdown(text)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})

"""Wires the real components together (used by the CLI scripts and the app)."""
from __future__ import annotations

from .config import get_settings
from .embedder import get_embedder
from .llm import LLMClient
from .pipeline import RAGPipeline
from .retriever import Retriever
from .vector_store import VectorStore


def build_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, s.collection_name)


def build_pipeline(store: VectorStore | None = None) -> RAGPipeline:
    store = store or build_store()
    retriever = Retriever(store, get_embedder())
    return RAGPipeline(retriever, LLMClient())

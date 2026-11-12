"""Semantic retrieval: embed the question, search the index, rank and filter."""
from __future__ import annotations

from typing import Sequence

from .embedder import Embedder
from .models import RetrievedChunk
from .vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        min_score: float = 0.20,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best first, dropping passages whose
        cosine similarity is below ``min_score`` (they would only add noise and
        invite hallucination)."""
        question = question.strip()
        if not question:
            return []
        k = top_k or self.top_k
        hits = self.store.query(self.embedder.embed_query(question), top_k=k, doc_ids=doc_ids)
        hits = [h for h in hits if h.score >= self.min_score]
        return sorted(hits, key=lambda h: h.score, reverse=True)

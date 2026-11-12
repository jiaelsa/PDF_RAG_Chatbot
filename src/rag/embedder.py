"""Text -> dense vector embeddings.

Two interchangeable backends (select with ``EMBEDDING_BACKEND``):

* ``sentence-transformers`` (default): Hugging Face model, best quality.
* ``onnx``: the same MiniLM model through ONNX Runtime (shipped with ChromaDB).
  No PyTorch needed, so it installs faster and uses far less RAM -- handy for
  small free hosting tiers.

All vectors are L2-normalised, so cosine similarity == dot product.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

from .config import Settings, get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OnnxEmbedder:
    """all-MiniLM-L6-v2 via ONNX Runtime (downloads ~80 MB on first use)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self.model_name = "onnx/all-MiniLM-L6-v2"
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_normalise([float(x) for x in v]) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=4)
def _build(backend: str, model_name: str) -> Embedder:
    if backend == "onnx":
        return OnnxEmbedder()
    if backend in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Return a process-wide cached embedder (model is loaded only once)."""
    s = settings or get_settings()
    return _build(s.embedding_backend, s.embedding_model)

"""Plain data containers shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Cleaned text of one PDF page (page_number is 1-based)."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage of text that lives on exactly one page of one document."""

    id: str
    text: str
    doc_id: str  # SHA-256 of the file content
    doc_name: str  # original file name
    page: int  # 1-based page number
    chunk_index: int  # running index inside the document

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store together with its similarity score
    (cosine similarity, higher = more relevant, range roughly 0..1)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    doc_name: str
    chunks: int
    pages: int

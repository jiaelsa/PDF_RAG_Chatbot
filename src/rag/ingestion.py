"""End-to-end ingestion: PDF bytes -> clean pages -> chunks -> embeddings -> index."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_pages
from .config import get_settings
from .embedder import Embedder
from .pdf_loader import load_pdf_bytes
from .vector_store import VectorStore

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]  # (stage description, 0..1)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    skipped: bool  # True when the identical file was already indexed


def ingest_pdf_bytes(
    data: bytes,
    name: str,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    progress: ProgressFn | None = None,
) -> IngestResult:
    """Index one PDF. Re-uploading an identical file is a no-op."""
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    notify = progress or (lambda stage, frac: None)

    doc_id = hashlib.sha256(data).hexdigest()
    if store.has_document(doc_id):
        log.info("%s already indexed, skipping", name)
        notify("Already indexed", 1.0)
        return IngestResult(doc_id, name, pages=0, chunks=0, skipped=True)

    notify("Reading PDF", 0.1)
    pages = load_pdf_bytes(data)

    notify("Splitting into chunks", 0.3)
    chunks = chunk_pages(pages, doc_id, name, chunk_size, chunk_overlap)

    notify(f"Embedding {len(chunks)} chunks", 0.5)
    vectors = embedder.embed_documents([c.text for c in chunks])

    notify("Writing to vector index", 0.9)
    store.add_chunks(chunks, vectors)
    notify("Done", 1.0)
    log.info("Indexed %s: %d pages, %d chunks", name, len(pages), len(chunks))
    return IngestResult(doc_id, name, pages=len(pages), chunks=len(chunks), skipped=False)


def ingest_pdf_path(path: str | Path, store: VectorStore, embedder: Embedder, **kwargs) -> IngestResult:
    p = Path(path)
    return ingest_pdf_bytes(p.read_bytes(), p.name, store, embedder, **kwargs)

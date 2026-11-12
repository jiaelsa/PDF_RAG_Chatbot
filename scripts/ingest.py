"""Command-line ingestion.

    python scripts/ingest.py data/sample_docs            # index every PDF in a folder
    python scripts/ingest.py report.pdf paper.pdf        # index specific files
    python scripts/ingest.py --list                      # show what is indexed
    python scripts/ingest.py --reset data/sample_docs    # wipe the index first
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.embedder import get_embedder  # noqa: E402
from rag.ingestion import ingest_pdf_path  # noqa: E402
from rag.pdf_loader import PDFLoadError  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def collect_pdfs(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf" and p.exists():
            found.append(p)
        else:
            print(f"! skipping {raw} (not a PDF or folder)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Index PDFs into the vector store.")
    parser.add_argument("paths", nargs="*", help="PDF files or folders")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    parser.add_argument("--list", action="store_true", help="list indexed documents and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    store = VectorStore(settings.chroma_dir, settings.collection_name)

    if args.list:
        docs = store.list_documents()
        if not docs:
            print("Index is empty.")
        for d in docs:
            print(f"{d.doc_name:40s} pages={d.pages:<4d} chunks={d.chunks}")
        return 0

    if args.reset:
        store.reset()
        print("Index cleared.")

    pdfs = collect_pdfs(args.paths)
    if not pdfs:
        print("Nothing to index.")
        return 1

    embedder = get_embedder(settings)
    failures = 0
    for pdf in pdfs:
        try:
            r = ingest_pdf_path(pdf, store, embedder)
        except PDFLoadError as exc:
            failures += 1
            print(f"x {pdf.name}: {exc}")
            continue
        status = "already indexed" if r.skipped else f"{r.pages} pages -> {r.chunks} chunks"
        print(f"✓ {pdf.name}: {status}")
    print(f"Total chunks in index: {store.count()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

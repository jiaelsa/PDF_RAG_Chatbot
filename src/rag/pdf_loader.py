"""PDF parsing: bytes -> list of cleaned pages."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .models import PageText
from .text_cleaner import normalize_text, strip_headers_footers

log = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """The file could not be read as a PDF (corrupt, encrypted, ...)."""


class NoTextError(PDFLoadError):
    """The PDF has no extractable text (probably a scanned document)."""


def load_pdf_bytes(data: bytes) -> list[PageText]:
    """Extract and clean the text of every page. Empty pages are skipped but
    page numbers of the remaining pages are preserved (1-based)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty user password.
            if not reader.decrypt(""):
                raise PDFLoadError("The PDF is password protected.")
        raw_pages = [(page.extract_text() or "") for page in reader.pages]
    except PDFLoadError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise PDFLoadError(f"Could not read PDF: {exc}") from exc
    except Exception as exc:  # pypdf can raise many different errors on bad files
        raise PDFLoadError(f"Could not read PDF: {exc}") from exc

    stripped = strip_headers_footers(raw_pages)
    pages = []
    for number, raw in enumerate(stripped, start=1):
        text = normalize_text(raw)
        if text:
            pages.append(PageText(page_number=number, text=text))

    if not pages:
        raise NoTextError(
            "No extractable text found. The PDF may be a scanned image; "
            "run OCR on it first."
        )
    log.info("Loaded %d/%d pages with text", len(pages), len(raw_pages))
    return pages


def load_pdf(path: str | Path) -> list[PageText]:
    return load_pdf_bytes(Path(path).read_bytes())

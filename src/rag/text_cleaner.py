"""Text cleaning: fixes the usual damage done by PDF text extraction."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SPACES = re.compile(r"[ \t\u00a0]+")
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Return a clean version of ``text``.

    * Unicode NFKC (turns ligatures such as 'ﬁ' into 'fi')
    * removes control characters
    * re-joins words hyphenated across line breaks ("embed-\\nding" -> "embedding")
    * merges hard-wrapped lines into flowing paragraphs, keeps blank-line breaks
    * collapses runs of whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    paragraphs = []
    for para in _PARAGRAPH_BREAK.split(text):
        para = _SPACES.sub(" ", para.replace("\n", " ")).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


_TRAILING_NUMBER = re.compile(r"^(.*?\D)\s*\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)


def _line_key(line: str) -> str:
    """Key used to detect running headers/footers.

    Only a *trailing* page counter is ignored ("Report 2024 - 3" and
    "Report 2024 - 4" match); a line that ends in text, such as "Step 1: install",
    must match exactly, so real content is never mistaken for a footer.
    """
    line = line.strip().lower()
    m = _TRAILING_NUMBER.match(line)
    return (m.group(1).strip() + " #") if m else line


def strip_headers_footers(raw_pages: list[str], min_pages: int = 4) -> list[str]:
    """Remove running headers/footers and bare page numbers.

    A short line is treated as a header/footer when (ignoring digits) it appears
    on at least 60% of the pages. Only applied to documents with ``min_pages``
    or more pages, otherwise real content could be removed by mistake.
    """
    cleaned = [
        "\n".join(l for l in page.splitlines() if not _PAGE_NUMBER_LINE.match(l))
        for page in raw_pages
    ]
    if len(cleaned) < min_pages:
        return cleaned

    counts: Counter[str] = Counter()
    for page in cleaned:
        keys = {_line_key(l) for l in page.splitlines() if l.strip() and len(l.strip()) <= 100}
        counts.update(keys)

    threshold = max(2, int(0.6 * len(cleaned)))
    repeated = {k for k, c in counts.items() if c >= threshold}
    if not repeated:
        return cleaned

    return [
        "\n".join(l for l in page.splitlines() if _line_key(l) not in repeated)
        for page in cleaned
    ]

"""Page-aware text chunking with overlap.

Chunks never cross a page boundary, so every chunk has an *exact* page number
that can be shown as a citation.
"""
from __future__ import annotations

import re

from .models import Chunk, PageText

_UNIT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_MIN_CHARS = 30  # drop tiny fragments (stray headings, etc.) unless they are all a page has


def _hard_split(piece: str, size: int) -> list[str]:
    """Split a piece that is longer than ``size`` on word boundaries."""
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for word in piece.split():
        if len(word) > size:  # absurdly long token (URL, base64, ...)
            if cur:
                out.append(" ".join(cur))
                cur, cur_len = [], 0
            out.extend(word[i : i + size] for i in range(0, len(word), size))
            continue
        add = len(word) + (1 if cur else 0)
        if cur and cur_len + add > size:
            out.append(" ".join(cur))
            cur, cur_len, add = [], 0, len(word)
        cur.append(word)
        cur_len += add
    if cur:
        out.append(" ".join(cur))
    return out


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for part in _UNIT_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        units.extend([part] if len(part) <= size else _hard_split(part, size))
    return units


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Greedy sentence packing. Each chunk is at most ``chunk_size`` characters;
    consecutive chunks share up to ``overlap`` characters (whole sentences) so
    that facts on a boundary are not lost."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0  # length of " ".join(cur)

    for unit in _units(text, chunk_size):
        add = len(unit) + (1 if cur else 0)
        if cur and cur_len + add > chunk_size:
            chunks.append(" ".join(cur))
            # keep trailing sentences as overlap
            keep: list[str] = []
            kept_len = 0
            for prev in reversed(cur):
                extra = len(prev) + (1 if keep else 0)
                if kept_len + extra > overlap:
                    break
                keep.insert(0, prev)
                kept_len += extra
            cur, cur_len = keep, kept_len
            # make sure the new unit fits; shrink overlap if necessary
            while cur and cur_len + len(unit) + 1 > chunk_size:
                dropped = cur.pop(0)
                cur_len = len(" ".join(cur))
            add = len(unit) + (1 if cur else 0)
        cur.append(unit)
        cur_len += add

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        pieces = split_text(page.text, chunk_size, overlap)
        if len(pieces) > 1:
            pieces = [p for p in pieces if len(p) >= _MIN_CHARS] or pieces
        for piece in pieces:
            chunks.append(
                Chunk(
                    id=f"{doc_id[:12]}-p{page.page_number}-c{index}",
                    text=piece,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page.page_number,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks

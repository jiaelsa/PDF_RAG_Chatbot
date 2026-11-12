"""Central configuration. Every value can be overridden with an environment
variable or a ``.env`` file (see ``.env.example``)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection_name: str
    embedding_backend: str  # "sentence-transformers" | "onnx"
    embedding_model: str
    chunk_size: int  # characters
    chunk_overlap: int  # characters


def get_settings() -> Settings:
    chroma_dir = os.getenv("CHROMA_DIR", str(ROOT_DIR / "storage" / "chroma"))
    return Settings(
        chroma_dir=chroma_dir,
        collection_name=os.getenv("COLLECTION_NAME", "pdf_chunks"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 200),
    )

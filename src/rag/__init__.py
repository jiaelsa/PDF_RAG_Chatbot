"""PDF Document Q&A Bot using Retrieval-Augmented Generation (RAG)."""


def _patch_sqlite() -> None:
    """ChromaDB needs SQLite >= 3.35. Some hosts (e.g. Streamlit Community Cloud)
    ship an older one, so swap in ``pysqlite3`` when it is installed."""
    try:
        import sys

        import pysqlite3  # type: ignore

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass


_patch_sqlite()

__version__ = "1.0.0"

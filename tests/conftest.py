import pytest

from rag.vector_store import VectorStore
from tests.helpers import FakeEmbedder


@pytest.fixture()
def embedder():
    return FakeEmbedder()


@pytest.fixture()
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"), "test_chunks")

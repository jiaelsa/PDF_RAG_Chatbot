"""Persistent vector index on top of ChromaDB (HNSW approximate nearest neighbour,
cosine distance)."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from .models import Chunk, DocumentInfo, RetrievedChunk

_ADD_BATCH = 500


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "pdf_chunks"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._name = collection_name
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------------- write
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), _ADD_BATCH):
            part = chunks[i : i + _ADD_BATCH]
            self._collection.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=[list(e) for e in embeddings[i : i + _ADD_BATCH]],
                metadatas=[c.metadata() for c in part],
            )

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        """Delete everything."""
        self._client.delete_collection(self._name)
        self._collection = self._open()

    # ----------------------------------------------------------------- read
    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_id: str) -> bool:
        res = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(res["ids"]) > 0

    def list_documents(self) -> list[DocumentInfo]:
        res = self._collection.get(include=["metadatas"])
        by_doc: dict[str, dict] = defaultdict(lambda: {"name": "", "chunks": 0, "pages": set()})
        for meta in res["metadatas"] or []:
            d = by_doc[meta["doc_id"]]
            d["name"] = meta["doc_name"]
            d["chunks"] += 1
            d["pages"].add(meta["page"])
        docs = [
            DocumentInfo(doc_id=k, doc_name=v["name"], chunks=v["chunks"], pages=len(v["pages"]))
            for k, v in by_doc.items()
        ]
        return sorted(docs, key=lambda d: d.doc_name.lower())

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks (best first). Optionally
        restrict the search to the given documents."""
        total = self.count()
        if total == 0:
            return []
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, min(top_k, total)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(
                id=cid,
                text=text,
                doc_id=meta["doc_id"],
                doc_name=meta["doc_name"],
                page=int(meta["page"]),
                chunk_index=int(meta["chunk_index"]),
            )
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

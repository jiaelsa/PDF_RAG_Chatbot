# Ingestion pipeline (Member 1)

```
PDF bytes ──► pdf_loader ──► text_cleaner ──► chunker ──► embedder ──► vector_store (ChromaDB)
               (pypdf)      (normalise,        (page-aware,  (MiniLM,     (cosine HNSW,
                             strip headers)     overlap)      384-d)       persisted on disk)
```

| Module | Responsibility |
|---|---|
| `pdf_loader.py` | Read a PDF from bytes, handle empty-password encryption, corrupt files and scanned (image-only) PDFs with clear errors. Keeps 1-based page numbers. |
| `text_cleaner.py` | Unicode NFKC (ligatures), control chars, de-hyphenation across line breaks, line re-flow, running header/footer and page-number removal. |
| `chunker.py` | Sentence-aware packing to `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. **Chunks never cross a page**, so each has an exact page number for citations. |
| `embedder.py` | `sentence-transformers/all-MiniLM-L6-v2` (default) or the same model through ONNX Runtime (`EMBEDDING_BACKEND=onnx`, no PyTorch). Vectors are L2-normalised. |
| `vector_store.py` | ChromaDB persistent collection with cosine distance; add / query / filter by document / list / delete / reset. |
| `ingestion.py` | Orchestrates the steps; document id = SHA-256 of the file, so re-uploading the same file is skipped. |

## Design decisions (good material for the report)
* **Page-bounded chunks** trade a little cross-page context for exact page-level citations.
* **Content hashing** gives de-duplication independent of file names.
* **Character-based sizes** (default 1000 / 200) keep the pipeline model-agnostic; MiniLM truncates at ~256 word pieces, so ~1000 characters fits comfortably.
* **Header/footer detection** only runs on documents of 4+ pages and only ignores a *trailing* page counter, so genuine content such as "Step 1: install" is never removed.

## Interface used by Member 2
```python
store.query(embedding, top_k, doc_ids=None) -> list[RetrievedChunk]   # chunk.doc_name, chunk.page, chunk.text, score
store.list_documents() -> list[DocumentInfo]
ingest_pdf_bytes(data, name, store, embedder, progress=None) -> IngestResult
get_embedder().embed_query(text) / .embed_documents(texts)
```

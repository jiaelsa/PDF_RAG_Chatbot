# Team contributions

| Member | Scope | Main files |
|---|---|---|
| **Member 1** – Data & indexing | PDF parsing, text cleaning, chunking, embeddings, ChromaDB index, ingestion CLI, ingestion tests, sample document | `src/rag/{pdf_loader,text_cleaner,chunker,embedder,vector_store,ingestion,models,config}.py`, `scripts/ingest.py`, `tests/test_{text_cleaner,chunker,ingestion}.py`, `docs/ingestion_pipeline.md` |
| **Member 2** – Retrieval, generation & app | Retriever, prompts, LLM client, RAG pipeline, Streamlit UI, evaluation, deployment, README | `src/rag/{retriever,prompts,llm,pipeline,factory}.py`, `app.py`, `scripts/{ask,evaluate}.py`, `tests/test_retrieval_pipeline.py`, `Dockerfile`, `docs/{architecture,deployment}.md`, `README.md` |

Edit the names above and keep the commit history honest: each member commits their own half.

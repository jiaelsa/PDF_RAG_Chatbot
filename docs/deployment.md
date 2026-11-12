# Deployment

## Option A: Streamlit Community Cloud (free, simplest)
1. Push the repo to GitHub (public, or private with access granted).
2. https://share.streamlit.io -> **Create app** -> pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings -> Secrets**, paste (TOML):
   ```toml
   LLM_API_KEY = "your-key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   EMBEDDING_BACKEND = "onnx"   # lighter than PyTorch; recommended on free tiers
   ```
4. Deploy. Notes: storage is ephemeral (the index is rebuilt after a restart, so re-upload PDFs), and the first start downloads the embedding model.

## Option B: Hugging Face Spaces (Docker)
Create a Space with the **Docker** SDK, push this repo (the `Dockerfile` is included), add `LLM_API_KEY` etc. as Space secrets, and set `app_port: 8501` in the Space README metadata.

## Option C: Local / Docker
```bash
docker build -t pdf-rag-bot .
docker run -p 8501:8501 --env-file .env pdf-rag-bot
```

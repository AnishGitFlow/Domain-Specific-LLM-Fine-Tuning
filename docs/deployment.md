# Deployment Guide

## Local Streamlit Demo

Run:

```bash
streamlit run app/streamlit_app.py
```

The app loads:

```text
Qwen/Qwen2.5-7B-Instruct
models/adapters/qwen2_5_7b_pubmedqa_qlora
```

Disable the adapter checkbox to compare base model behavior interactively.

## Local FastAPI Demo

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Generation request:

```bash
curl -X POST http://localhost:8000/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"instruction\":\"Answer the question from the context.\",\"context\":\"Paste biomedical abstract here.\"}"
```

## Hugging Face Spaces

Recommended for a free public demo:

1. Create a new Space.
2. Choose Streamlit.
3. Upload this repo.
4. Add model adapter files or download them from the Hub.
5. Use CPU for a lightweight demo, or choose GPU if available.

For CPU-only Spaces, prefer GGUF plus llama.cpp or a smaller fallback model such as Phi-3 Mini.

## Docker

Build:

```bash
docker build -t pubmedqa-qlora .
```

Run:

```bash
docker run --gpus all -p 7860:7860 pubmedqa-qlora
```

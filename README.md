# RAG Console

Minimal Flask RAG app for uploading a document, indexing it locally, and asking questions with citations.

## Features
- Upload and index a single source (PDF/DOCX/TXT/MD/CSV/JSON/HTML/XML).
- In-memory hybrid retrieval: dense embeddings (Gemini) + BM25, fused with RRF.
- Top matches view plus grounded answers with numbered citations.
- Evidence preview: click a citation or a match card to open the referenced chunk.
- Ingestion progress UI (upload, extract, chunk, embed, index).

## Architecture
```mermaid
flowchart TD
  UI[Browser UI] -->|upload/query| API[Flask app]
  API -->|/api/upload| X[Extract text]
  X --> C[Chunker]
  C --> E[Gemini embed]
  E --> V[(In-memory vector index)]
  C --> B[(In-memory BM25)]
  API -->|/api/retrieve| R[Hybrid retrieve + RRF fuse]
  V --> R
  B --> R
  API -->|/api/answer| L["Gemini chat (grounded answer)"]
  R --> L
  L --> UI
```

See `architecture.md` for a deeper design doc.

## Configuration
Copy `.env.example` to `.env`. Key settings:
- `GOOGLE_API_KEY` (required)
- `MAX_UPLOAD_MB` (default `250`)
- `CHUNK_MAX_TOKENS` (default `512`)
- `CHUNK_OVERLAP_TOKENS` (default `32`)
- `GEMINI_EMBED_MODEL` (default `text-embedding-004`)
- `GEMINI_CHAT_MODEL` (default `gemini-2.5-flash`)
- `FLASK_HOST` (default `0.0.0.0`)
- `FLASK_PORT` (default `5000`)
- `FLASK_DEBUG` (default `0`)

Query tuning defaults live in `webapp/config.py` (`QuerySettings`).

## Pipeline
1. Upload + extract text (PDF/DOCX/TXT/etc).
2. Chunk + embed (Gemini) and build in-memory dense + BM25 indexes.
3. Query uses hybrid retrieval + RRF fusion, then the LLM answers with chunk citations (shown in the UI).

## Run
1. Create a virtualenv and install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Set your Gemini key:
   - `cp .env.example .env`
   - edit `.env` and set `GOOGLE_API_KEY`
3. Start the server:
   - `python app.py`
4. Open:
   - `http://localhost:5000`

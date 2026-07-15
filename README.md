# Enterprise RAG Assistant

FastAPI backend for owner-scoped document ingestion, chunking, pgvector
storage, embedding, retrieval, and grounded context preparation.

## Development embeddings

The application supports configuration-based embedding providers:

- `EMBEDDING_PROVIDER=fake` uses deterministic, normalized hash vectors. It
  requires no API key or network request and is intended only for local
  development, tests, and end-to-end pipeline validation.
- `EMBEDDING_PROVIDER=openai` uses the configured OpenAI embedding model and is
  the production setting.

Fake embeddings have limited lexical retrieval quality. They are not
production-quality semantic embeddings and should not be used to benchmark
retrieval relevance.

Chunks embedded by different providers or models must not be mixed. Retrieval
filters chunks by the active `EMBEDDING_MODEL`. Before changing providers or
models, clear the document's old embeddings and regenerate them.

## Fake-mode environment

```env
EMBEDDING_PROVIDER=fake
EMBEDDING_MODEL=fake-embedding-v1
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=50
OPENAI_API_KEY=
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SCORE=0.30
```

Keep the existing database, JWT, Redis, and upload settings. Copy
`.env.example` when creating a new local environment and replace placeholder
database/JWT values.

## Run and test

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The current schema already contains `embedding`, `embedding_model`, and
`embedded_at`, so this change needs no new migration. For a database that has
not applied the existing embedding migration:

```powershell
.\venv\Scripts\python.exe -m alembic upgrade head
```

## Development API flow

Open Swagger at `/docs`, authorize with the JWT returned by login, then:

1. `POST /api/v1/documents/upload`
2. `POST /api/v1/documents/{document_id}/process`
3. `POST /api/v1/documents/{document_id}/embed`
4. `POST /api/v1/retrieval/search`
5. `POST /api/v1/context/build`

`POST /api/v1/context/build` returns retrieved source context and
`llm_status=not_configured`; it does not call an LLM or fabricate an answer.

To remove vectors without deleting chunks, call:

```http
DELETE /api/v1/documents/{document_id}/embeddings
```

## Switching to OpenAI

1. While fake mode is active, clear embeddings for every document that will be
   re-embedded.
2. Set:

   ```env
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_DIMENSION=1536
   OPENAI_API_KEY=your_key
   ```

3. Restart the API.
4. Call the document embed endpoint again for each cleared document.
5. Verify retrieval; model filtering prevents old fake vectors from being
   compared with OpenAI query vectors.

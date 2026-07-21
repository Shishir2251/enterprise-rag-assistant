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
```

Apply all database migrations:

```powershell
.\venv\Scripts\python.exe -m alembic upgrade head
```

## Automatic document processing

Document upload saves the file and database record, then queues extraction,
chunking, and embedding work for Celery. The API request does not run those
operations synchronously.

Development startup uses three terminals.

Terminal 1:

```powershell
docker compose up -d redis
```

Terminal 2 (Windows):

```powershell
.\venv\Scripts\python.exe -m celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo
```

Terminal 3:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Required queue configuration:

```env
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
CELERY_TASK_ALWAYS_EAGER=False
CELERY_TASK_EAGER_PROPAGATES=True
DOCUMENT_PROCESSING_MAX_RETRIES=3
DOCUMENT_PROCESSING_RETRY_DELAY_SECONDS=30
```

For unit tests, Celery eager mode can be enabled without Redis:

```env
CELERY_TASK_ALWAYS_EAGER=True
CELERY_TASK_EAGER_PROPAGATES=True
```

## Development API flow

Open Swagger at `/docs`, authorize with the JWT returned by login, then:

1. `POST /api/v1/documents/upload`
2. Confirm the upload response has `status=queued`
3. Poll `GET /api/v1/documents/{document_id}/status`
4. Wait for `status=ready`
5. `POST /api/v1/retrieval/search`
6. `POST /api/v1/context/build`
7. Create a chat session and send a chat message

Failed documents can be queued again with:

```http
POST /api/v1/documents/{document_id}/retry
```

The synchronous `/process` and `/embed` routes remain available as deprecated
development endpoints. Production clients should use upload plus the status
endpoint.

`POST /api/v1/context/build` returns retrieved source context and
`llm_status=not_configured`; it does not call an LLM or fabricate an answer.

## Chat scaffolding

Chat sessions and conversation messages are persisted in PostgreSQL. The
default configuration is:

```env
LLM_PROVIDER=none
```

In this mode, sending a chat message still uses the owner-scoped
`RetrievalService` and `ContextBuilderService`, and returns retrieved citation
metadata. It deliberately returns:

```json
{
  "status": "llm_not_configured",
  "answer": null
}
```

The user message is persisted before retrieval. No assistant message is
persisted unless a configured provider returns a real, non-empty answer.

Chat API flow:

1. `POST /api/v1/chat/sessions`
2. `POST /api/v1/chat/sessions/{session_id}/messages`
3. `GET /api/v1/chat/sessions/{session_id}/messages`
4. `GET /api/v1/chat/sessions`

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

# Enterprise RAG Assistant

FastAPI backend for owner-scoped document ingestion, chunking, pgvector
storage, embedding, retrieval, and grounded RAG answer generation.

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

## Grounded chat

Chat sessions and conversation messages are persisted in PostgreSQL. The
default configuration disables generation while leaving owner-scoped retrieval
and citation inspection available:

```env
LLM_PROVIDER=disabled
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=1200
LLM_TIMEOUT_SECONDS=30
MAX_CONTEXT_CHARACTERS=12000
CHAT_HISTORY_MAX_MESSAGES=10
OPENAI_API_KEY=
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
persisted unless a configured provider returns a real, non-empty answer. If
retrieval supplies no usable context, the provider is not called and the
assistant persists the deterministic insufficient-context response.

With OpenAI generation enabled, use:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=1200
LLM_TIMEOUT_SECONDS=30
MAX_CONTEXT_CHARACTERS=12000
CHAT_HISTORY_MAX_MESSAGES=10
OPENAI_API_KEY=<real-key>
```

Restart FastAPI after changing provider configuration. Retrieved document text
is placed only in the user prompt as untrusted evidence. The system prompt
forbids outside knowledge, document-borne instructions, fabricated facts, and
fabricated source numbers. Returned citation metadata is limited to valid
`[SOURCE n]` markers actually used in the answer.

Chat API flow:

1. `POST /api/v1/chat/sessions`
2. `POST /api/v1/chat/sessions/{session_id}/messages`
3. `GET /api/v1/chat/sessions/{session_id}/messages`
4. `GET /api/v1/chat/sessions`

## Exact Swagger chat test

Start Redis, the Celery worker, and FastAPI as shown above, then open
`http://localhost:8000/docs`.

1. Call `POST /api/v1/auth/register` if a test user does not exist.
2. Call `POST /api/v1/auth/login`, copy `access_token`, select **Authorize**,
   and enter the bearer token.
3. Call `POST /api/v1/documents/upload` with a PDF, DOCX, or TXT file. Save the
   returned document ID.
4. Poll `GET /api/v1/documents/{document_id}/status` until `status` is `ready`.
5. Call `POST /api/v1/chat/sessions` and save the returned session ID.
6. Call `POST /api/v1/chat/sessions/{session_id}/messages` with a question whose
   answer appears in the uploaded document. In disabled mode, verify
   `status=llm_not_configured`, `answer=null`, and
   `assistant_message_id=null`.
7. For OpenAI mode, restart the API with the OpenAI environment block above and
   repeat step 6. Verify `status=completed`, a non-empty answer, a non-null
   `assistant_message_id`, and citations whose source numbers and metadata
   correspond to source markers in the answer.
8. Ask a question for which retrieval returns no chunks above
   `RETRIEVAL_MIN_SCORE`. Verify the exact answer
   `I could not find enough information in the provided documents.`,
   `status=completed`, and an empty citation list.

Use `GET /api/v1/chat/sessions/{session_id}/messages` to confirm that user turns
are retained after generation failures and assistant turns exist only for
successful or deterministic no-context answers.

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

Embedding and LLM providers are configured independently. OpenAI answer
generation does not require changing `EMBEDDING_PROVIDER`; fake embeddings can
exercise the architecture locally, but they are not a measure of semantic
retrieval quality.

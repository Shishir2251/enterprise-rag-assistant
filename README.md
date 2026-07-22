# Enterprise RAG Assistant

FastAPI backend for owner-scoped document ingestion, chunking, pgvector
storage, embedding, retrieval, and grounded RAG answer generation.

## Embedding architecture

The application selects an `IEmbeddingProvider` without changing business
services:

- `EMBEDDING_PROVIDER=fake` uses deterministic test vectors and no API key.
- `EMBEDDING_PROVIDER=local` keeps the in-process SentenceTransformers option
  for Linux, macOS, and unrestricted Windows environments.
- `EMBEDDING_PROVIDER=http` calls the CPU-only Docker embedding service. This
  is the supported development mode on the managed Windows host.
- `EMBEDDING_PROVIDER=openai` preserves the later cloud deployment path.

In Windows development the process boundary is:

```text
FastAPI and Celery on Windows
        -> HTTP on 127.0.0.1:8090
Linux Docker embedding-service
        -> sentence-transformers/all-MiniLM-L6-v2 on CPU
384-dimensional normalized vectors
        -> PostgreSQL with pgvector
```

HTTP mode neither needs `OPENAI_API_KEY` nor imports SentenceTransformers in
the FastAPI or Celery process. The Linux service loads the model once at
startup and batches `/embed` requests. Direct `local` mode remains available;
do not use it on a host where enterprise Application Control blocks its native
Python wheels, and do not bypass that policy.

Chunks embedded by different providers or models are not mixed. Retrieval is
owner scoped and filters by the active provider and model. A missing vector or
a provider/model mismatch is stale and is regenerated during processing or
reindexing.

## Docker embedding service

The Compose file preserves `redis` and adds `embedding-service`, built from
`embedding_service/`, published as `8090:8090`, configured for the MiniLM CPU
model, and guarded by its `/health` check. The named
`embedding_model_cache` volume persists the Hugging Face cache so the model is
not downloaded on every container restart.

Its container environment is
`LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`,
`LOCAL_EMBEDDING_BATCH_SIZE=32`, `LOCAL_EMBEDDING_DEVICE=cpu`,
`EMBEDDING_SERVICE_HOST=0.0.0.0`, and `EMBEDDING_SERVICE_PORT=8090`.

The first `docker compose up -d embedding-service` can take longer while
Docker builds the image, installs the Python dependencies, and downloads the
model. Later startups use the cached model and should be faster.

Start Redis and the embedding service:

```powershell
docker compose up -d redis embedding-service
docker ps
```

Check service health with either command:

```powershell
curl http://127.0.0.1:8090/health
Invoke-RestMethod http://127.0.0.1:8090/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimension": 384
}
```

Test a query embedding from PowerShell and verify its vector length:

```powershell
$response = Invoke-RestMethod `
  -Uri http://127.0.0.1:8090/embed-query `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"When does the football tournament begin?"}'

$response.embedding.Count  # expected: 384
```

## HTTP-mode environment

Keep the existing database, JWT, Redis, and upload settings, and use this exact
development embedding configuration:

```env
EMBEDDING_PROVIDER=http
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

HTTP_EMBEDDING_BASE_URL=http://127.0.0.1:8090
HTTP_EMBEDDING_TIMEOUT_SECONDS=30

RETRIEVAL_TOP_K_DEFAULT=5
RETRIEVAL_TOP_K_MAX=20
RETRIEVAL_MIN_SCORE=0.25

LLM_PROVIDER=fake
FAKE_LLM_MODEL=fake-grounded-llm-v1
OPENAI_API_KEY=
```

Copy `.env.example` when creating a new environment and replace its database
and JWT placeholders. No embedding-service API key is required.

## Run and test

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Apply all database migrations:

```powershell
python -m alembic upgrade head
```

The Phase 8 migration clears only disposable embedding values and their
model/provider timestamps before changing `document_chunks.embedding` from
`vector(1536)` to `vector(384)`. Documents, extracted text, and chunk rows are
preserved. Reindex existing ready documents afterward:

```powershell
python -m app.scripts.reindex_embeddings
```

## Automatic document processing

Document upload saves the file and database record, then queues extraction,
chunking, and embedding work for Celery. The API request does not run those
operations synchronously.

Development startup uses three terminals.

Terminal 1 (Docker Redis and Linux embedding service):

```powershell
docker compose up -d redis embedding-service
```

Terminal 2 (Windows):

```powershell
python -m celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo
```

Terminal 3:

```powershell
python -m uvicorn app.main:app --reload
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

The default suite excludes integration tests so it remains fast and never
requires a running embedding container or a real model download:

```powershell
python -m pytest -v
python -m pytest --cov=app --cov-report=term-missing
```

Run integration tests separately after the required embedding runtime is
available. Direct-local tests may download the model, while HTTP integration
tests skip cleanly when `http://127.0.0.1:8090` is unavailable:

```powershell
python -m pytest -m integration -v
```

Evaluate labelled retrieval cases for an existing owner without changing any
threshold automatically:

```powershell
python -m app.scripts.evaluate_retrieval --owner-id <owner-id>
```

The report prints Total cases, Hit@1, Hit@3, Hit@5, MRR, Average relevant
score, and a calibration recommendation. The fixture is synthetic; replace
its optional document IDs with IDs from your own synthetic development data.

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

Ready documents can be queued for embeddings-only reindexing without
re-extraction or chunk replacement:

```http
POST /api/v1/documents/{document_id}/reindex
```

The synchronous `/process` and `/embed` routes remain available as deprecated
development endpoints. Production clients should use upload plus the status
endpoint.

`POST /api/v1/context/build` returns retrieved source context and
`llm_status=not_configured`; it does not call an LLM or fabricate an answer.

## Grounded chat

Chat sessions and conversation messages are persisted in PostgreSQL. The
default configuration uses deterministic local generation, with no OpenAI key
or LLM network request required:

```env
LLM_PROVIDER=fake
FAKE_LLM_MODEL=fake-grounded-llm-v1
OPENAI_CHAT_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0
LLM_MAX_OUTPUT_TOKENS=800
LLM_TIMEOUT_SECONDS=45
CHAT_CONTEXT_MAX_CHARACTERS=12000
CHAT_HISTORY_MAX_MESSAGES=10
CHAT_HISTORY_MAX_CHARACTERS=6000
CHAT_DEFAULT_TOP_K=5
CHAT_MAX_TOP_K=10
OPENAI_API_KEY=
```

Fake mode uses the owner-scoped `RetrievalService` and
`ContextBuilderService`, selects evidence deterministically, returns a concise
answer with a valid `[SOURCE n]` marker, and persists only validated citation
metadata. To exercise retrieval without generation, set
`LLM_PROVIDER=disabled`. For backward compatibility, the existing
`POST /api/v1/chat/sessions/{session_id}/messages` route returns HTTP 200 with:

```json
{
  "status": "llm_not_configured",
  "answer": null
}
```

The new `POST /api/v1/chat` route instead maps that configured-but-disabled
state to HTTP 503 with the sanitized detail `LLM provider is not configured.`
Both paths persist the user message and do not persist an assistant message.

The user message is persisted before retrieval. If generation fails, no fake
completed assistant message is persisted. If retrieval supplies no usable
context, the provider is not called and the assistant persists the exact
deterministic insufficient-context response.

With OpenAI generation enabled, use:

```env
LLM_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0
LLM_MAX_OUTPUT_TOKENS=800
LLM_TIMEOUT_SECONDS=45
CHAT_CONTEXT_MAX_CHARACTERS=12000
CHAT_HISTORY_MAX_MESSAGES=10
CHAT_HISTORY_MAX_CHARACTERS=6000
OPENAI_API_KEY=<real-key>
```

Restart FastAPI after changing provider configuration. Retrieved document text
is placed only in the user prompt as untrusted evidence. The system prompt
forbids outside knowledge, document-borne instructions, fabricated facts, and
fabricated source numbers. Returned citation metadata is limited to valid
`[SOURCE n]` markers actually used in the answer.

Chat API flow:

1. `POST /api/v1/chat` creates or continues a grounded conversation.
2. `GET /api/v1/conversations` lists owned conversations.
3. `GET /api/v1/conversations/{conversation_id}` returns conversation detail.
4. `GET /api/v1/conversations/{conversation_id}/messages` returns messages.

The existing `/api/v1/chat/sessions` routes remain supported. Complete setup,
security behavior, Windows-safe commands, and manual acceptance examples are
documented in [Phase 9 grounded chat](docs/phase_9_grounded_chat.md).

## Exact Swagger chat test

Start Redis, the Celery worker, and FastAPI as shown above, then open
`http://localhost:8000/docs`.

1. Call `POST /api/v1/auth/register` if a test user does not exist.
2. Call `POST /api/v1/auth/login`, copy `access_token`, select **Authorize**,
   and enter the bearer token.
3. Call `POST /api/v1/documents/upload` with a PDF, DOCX, or TXT file. Save the
   returned document ID.
4. Poll `GET /api/v1/documents/{document_id}/status` until `status` is `ready`.
5. Call `POST /api/v1/chat` with the question and ready document ID. In default
   fake mode, save the returned `conversation_id` and verify
   `status=completed`, `llm_provider=fake`, a non-empty `message_id`, and at
   least one citation matching a marker in the answer.
6. Call `POST /api/v1/chat` again with that `conversation_id`, the same document
   ID, and a follow-up question. Verify the returned conversation ID is
   unchanged and the new assistant message and citations are persisted.
7. To inspect disabled behavior, restart with `LLM_PROVIDER=disabled` and send
   a supported question to `POST /api/v1/chat`. Verify HTTP 503 and the
   sanitized detail `LLM provider is not configured.` Use the legacy session
   message route only when testing its HTTP 200 `llm_not_configured`
   compatibility response.
8. Ask a question for which retrieval returns no chunks above
   `RETRIEVAL_MIN_SCORE`. Verify the exact answer
   `I could not find enough information in the selected documents.`,
   `status=completed`, and an empty citation list.

Use `GET /api/v1/chat/sessions/{session_id}/messages` to confirm that user turns
are retained after generation failures and assistant turns exist only for
successful or deterministic no-context answers.

To remove vectors without deleting chunks, call:

```http
DELETE /api/v1/documents/{document_id}/embeddings
```

## Exact HTTP semantic retrieval test flow

Use the HTTP-mode environment block above, then run these commands in order:

```powershell
docker compose up -d redis embedding-service
python -m alembic upgrade head
python -m celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo
python -m uvicorn app.main:app --reload
```

The Celery and FastAPI commands run in separate terminals and must start even
when Windows blocks the native `sentence_transformers`/`regex` DLL import,
because only the Linux container imports that package in HTTP mode.

For a repeatable end-to-end semantic check:

1. Confirm `Invoke-RestMethod http://127.0.0.1:8090/health` reports the MiniLM
   model and dimension 384.
2. Run `python -m alembic upgrade head` before processing documents.
3. Start the Windows Celery worker with the exact command above.
4. Start FastAPI with the exact command above and open
   `http://127.0.0.1:8000/docs`.
5. Register or log in and authorize Swagger with the returned JWT.
6. Upload a TXT file containing these separate synthetic passages:

   ```text
   The FIFA World Cup 2026 will be hosted by the United States, Canada, and Mexico.
   Python is widely used for machine learning and data science.
   Redis is commonly used for caching and message queues.
   ```

7. Poll `GET /api/v1/documents/{document_id}/status` until `status=ready`.
8. Inspect the database in a development/admin session and confirm chunk
   vectors exist with `embedding_provider=http` and
   `embedding_model=sentence-transformers/all-MiniLM-L6-v2`. Normal API
   responses must not expose the vector arrays.
9. Search with the paraphrase `Which countries are hosting the 2026 football
   tournament?`; do not use exact phrase matching as the test.
10. Verify the chunk naming the United States, Canada, and Mexico ranks near
    the top.
11. Search for an unrelated subject and verify its score is lower or its result
    is filtered by `RETRIEVAL_MIN_SCORE`.
12. Call `POST /api/v1/context/build` with the related question and verify the
    relevant source context is present.
13. A chat request returns a deterministic grounded answer in fake mode. It
    returns `status=llm_not_configured` and `answer=null` only when explicitly
    configured with `LLM_PROVIDER=disabled`.
14. Queue `POST /api/v1/documents/{document_id}/reindex` twice, waiting for the
    first run to finish, and confirm the existing chunks are reused rather than
    duplicated and remain owner scoped.

## Switching to OpenAI

The current schema intentionally has one fixed dimension: 384. Do not switch
only the environment variables; a 1536-dimensional OpenAI vector cannot be
inserted into `vector(384)`.

1. Stop FastAPI and all Celery workers.
2. Change the ORM vector declaration to `Vector(1536)` and create a new forward
   Alembic revision. In that revision, set `embedding`, `embedding_model`,
   `embedding_provider`, and `embedded_at` to `NULL`, then alter the column to
   `extensions.vector(1536)`. Keep the provider metadata column.
3. Apply that forward revision with `python -m alembic upgrade head`. Documents and chunk
   text remain intact; only disposable embeddings are reset.
4. Set:

   ```env
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_DIMENSION=1536
   OPENAI_API_KEY=<real-key>
   ```

5. Run `python -m app.scripts.reindex_embeddings`.
6. Restart FastAPI and Celery, then verify retrieval. Provider/model filtering
   excludes anything not produced by the active configuration. No business
   service change is required.

For a full Phase 8 schema-and-code rollback, the migration downgrade command is
`python -m alembic downgrade d4a6f21c8b90`; it also clears embeddings and restores
`vector(1536)`. Do not run that downgrade while using the current Phase 8 code,
because the downgraded schema removes `embedding_provider`.

A future multi-provider deployment may use separate provider-specific vector
tables or another fixed-dimension strategy. This phase deliberately keeps one
fixed vector column and does not implement that redesign.

Embedding and LLM providers are configured independently. OpenAI answer
generation does not require changing `EMBEDDING_PROVIDER`; keep
`LLM_PROVIDER=fake` until a real OpenAI key is intentionally configured.

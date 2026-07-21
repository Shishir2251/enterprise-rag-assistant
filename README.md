# Enterprise RAG Assistant

FastAPI backend for owner-scoped document ingestion, chunking, pgvector
storage, embedding, retrieval, and grounded RAG answer generation.

## Local semantic embeddings

The application supports configuration-based embedding providers:

- `EMBEDDING_PROVIDER=fake` uses deterministic, normalized hash vectors. It
  requires no API key and is intended for fast architecture tests.
- `EMBEDDING_PROVIDER=local` uses
  `sentence-transformers/all-MiniLM-L6-v2` on CPU. It produces real semantic,
  normalized 384-dimensional vectors without an API key.
- `EMBEDDING_PROVIDER=openai` uses the configured OpenAI embedding model and is
  preserved for a later cloud deployment.

Fake embeddings have limited lexical retrieval quality. They are not
production-quality semantic embeddings and should not be used to benchmark
retrieval relevance.

Chunks embedded by different providers or models must not be mixed. Retrieval
filters chunks by both the active provider and model. Missing embeddings and
model/provider mismatches are treated as stale and regenerated.

Install the local provider dependency:

```powershell
pip install sentence-transformers
```

The first use may download the configured model. Later runs try the local
SentenceTransformers cache first and can run without a network connection.
On managed Windows machines, organization Application Control must permit the
native wheels used by SentenceTransformers (for example PyTorch and `regex`).
If policy blocks one of those DLLs, ask the administrator to approve the
project environment; do not bypass the policy.

## Local-mode environment

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=50
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_BATCH_SIZE=32
LOCAL_EMBEDDING_DEVICE=cpu
OPENAI_API_KEY=
RETRIEVAL_TOP_K_DEFAULT=5
RETRIEVAL_TOP_K_MAX=20
RETRIEVAL_MIN_SCORE=0.25
LLM_PROVIDER=disabled
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
alembic upgrade head
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

Terminal 1:

```powershell
docker compose up -d redis
```

Terminal 2 (Windows):

```powershell
celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo
```

Terminal 3:

```powershell
uvicorn app.main:app --reload
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

The default suite excludes the real-model integration marker so it remains
fast and does not download a model:

```powershell
python -m pytest -v
python -m pytest --cov=app --cov-report=term-missing
```

Run the two real semantic-ranking cases separately (the first run may download
the model):

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

## Exact local semantic retrieval test flow

1. Set `EMBEDDING_PROVIDER=local`.
2. Set `EMBEDDING_DIMENSION=384`.
3. Run `alembic upgrade head`.
4. Start Redis with `docker compose up -d redis`.
5. Start the Celery worker with
   `celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo`.
6. Start FastAPI with `uvicorn app.main:app --reload`.
7. Log in and authorize Swagger with the returned JWT.
8. Upload a new PDF, DOCX, or TXT document.
9. Poll its status until `status=ready`.
10. Verify its chunks report the active local embedding model/provider metadata
    through a database/admin inspection; API responses never expose vectors.
11. Call retrieval with a semantically related question.
12. Verify the relevant chunk ranks near the top.
13. Ask a semantically unrelated query.
14. Verify its score is lower or the result is filtered by the configured
    threshold.
15. Test `POST /api/v1/context/build` with the related question.
16. Chat may still return `llm_not_configured` because
    `LLM_PROVIDER=disabled`.

## Switching to OpenAI

The current schema intentionally has one fixed dimension: 384. Do not switch
only the environment variables; a 1536-dimensional OpenAI vector cannot be
inserted into `vector(384)`.

1. Stop FastAPI and all Celery workers.
2. Change the ORM vector declaration to `Vector(1536)` and create a new forward
   Alembic revision. In that revision, set `embedding`, `embedding_model`,
   `embedding_provider`, and `embedded_at` to `NULL`, then alter the column to
   `extensions.vector(1536)`. Keep the provider metadata column.
3. Apply that forward revision with `alembic upgrade head`. Documents and chunk
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
   excludes anything not produced by the active configuration.

For a full Phase 8 schema-and-code rollback, the migration downgrade command is
`alembic downgrade d4a6f21c8b90`; it also clears embeddings and restores
`vector(1536)`. Do not run that downgrade while using the current Phase 8 code,
because the downgraded schema removes `embedding_provider`.

A future multi-provider deployment may use separate provider-specific vector
tables or another fixed-dimension strategy. This phase deliberately keeps one
fixed vector column and does not implement that redesign.

Embedding and LLM providers are configured independently. OpenAI answer
generation does not require changing `EMBEDDING_PROVIDER`; keep
`LLM_PROVIDER=disabled` until a working generation provider is configured.

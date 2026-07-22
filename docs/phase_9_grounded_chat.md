# Phase 9: Production-ready grounded chat

Phase 9 adds an authenticated, owner-scoped grounded-chat workflow that can be
used locally without an OpenAI key. The default `fake` LLM reads the real
retrieved source blocks and returns a deterministic answer with validated
`[SOURCE n]` markers.

## Architecture flow

1. Authenticate the caller.
2. Resolve the caller's conversation or create one with a deterministic title.
3. Deduplicate and explicitly validate every selected document by owner and
   processing status.
4. Persist the normalized user message.
5. Load only bounded, completed conversation history.
6. Run the existing owner-scoped semantic retrieval service.
7. Build complete, size-bounded source blocks after BOM/NUL sanitation.
8. Skip generation and persist the deterministic fallback if there is no
   usable context.
9. Otherwise, put history, untrusted retrieved context, and the current
   question in separate XML-like user-prompt blocks.
10. Generate through the configured provider, validate source markers, persist
    the assistant result and citations, and return only safe metadata.

The embedding and LLM providers are independent. The Windows application uses
the HTTP embedding provider, so the FastAPI and Celery processes do not import
`sentence_transformers`; that native dependency remains inside the Linux
embedding-service container.

## Local environment

Use this configuration for the complete local flow without a GPT key:

```env
EMBEDDING_PROVIDER=http
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
HTTP_EMBEDDING_BASE_URL=http://127.0.0.1:8090

LLM_PROVIDER=fake
FAKE_LLM_MODEL=fake-grounded-llm-v1
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4.1-mini

LLM_TIMEOUT_SECONDS=45
LLM_MAX_OUTPUT_TOKENS=800
LLM_TEMPERATURE=0

CHAT_HISTORY_MAX_MESSAGES=10
CHAT_HISTORY_MAX_CHARACTERS=6000
CHAT_CONTEXT_MAX_CHARACTERS=12000
CHAT_DEFAULT_TOP_K=5
CHAT_MAX_TOP_K=10
CHAT_NO_CONTEXT_MESSAGE=I could not find enough information in the selected documents.
```

`CHAT_HISTORY_MAX_MESSAGES` and `CHAT_HISTORY_MAX_CHARACTERS` prefer the newest
complete messages without cutting a message, then return them chronologically.
`CHAT_CONTEXT_MAX_CHARACTERS` includes only complete source blocks.

## LLM modes

- `fake` requires no API key or network access. It parses the structured source
  blocks, ranks their safe sentences by deterministic lexical overlap with the
  question, and cites only the selected source number. It filters document-borne
  instruction text, injected source markers, and credential-like values.
- `disabled` (and the legacy alias `none`) requires no key and does not import
  or construct OpenAI. With retrieved context, the new `POST /api/v1/chat`
  route returns a sanitized HTTP 503 after the user message is persisted. The
  legacy session-message route retains its HTTP 200 compatibility response
  with `status=llm_not_configured`, `answer=null`, and no assistant message.
  The no-context path still persists and returns the fallback because it never
  invokes a provider.
- `openai` lazily imports the SDK and constructs `AsyncOpenAI` only after a
  non-placeholder key has been validated. Authentication, rate-limit, timeout,
  transport, blank-output, malformed-output, and non-completed-response errors
  are mapped to sanitized application errors.

To activate OpenAI answer generation later, set only:

```env
LLM_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_API_KEY=<real-key>
```

Restart FastAPI after the change. This does not require changing the HTTP
embedding provider or the existing `vector(384)` schema.

## API

All routes require the existing bearer-token authentication.

- `POST /api/v1/chat` creates or continues a grounded conversation.
- `GET /api/v1/conversations` lists the caller's conversations.
- `GET /api/v1/conversations/{conversation_id}` returns one owned conversation
  and its visible messages.
- `GET /api/v1/conversations/{conversation_id}/messages` returns its messages.
- The existing `/api/v1/chat/sessions` and
  `/api/v1/chat/sessions/{session_id}/messages` routes remain supported.

New grounded chat request:

```json
{
  "message": "Which countries host the 2026 World Cup?",
  "conversation_id": null,
  "document_ids": ["REAL_DOCUMENT_ID"],
  "top_k": 5
}
```

Representative response:

```json
{
  "conversation_id": "REAL_CONVERSATION_ID",
  "message_id": "REAL_ASSISTANT_MESSAGE_ID",
  "answer": "The 2026 FIFA World Cup will be hosted by the United States, Canada, and Mexico. [SOURCE 1]",
  "status": "completed",
  "llm_provider": "fake",
  "llm_model": "fake-grounded-llm-v1",
  "citations": [
    {
      "source_number": 1,
      "document_id": "REAL_DOCUMENT_ID",
      "document_name": "phase9_multitopic.txt",
      "chunk_id": "REAL_CHUNK_ID",
      "chunk_index": 0,
      "page_number": null,
      "similarity_score": 0.57
    }
  ]
}
```

The API intentionally keeps validated markers in the displayed answer and also
returns structured citations. Repeated valid markers are deduplicated in
first-reference order. Zero, negative, malformed, missing, and out-of-range
markers never produce citations. Responses never contain source content,
storage paths, prompts, raw provider objects, embeddings, or credentials.

Unknown or cross-owner conversation/document IDs return the same not-found
behavior. A known owned document that is not `ready` (or legacy `completed`)
returns a conflict before retrieval. The endpoint deduplicates document IDs and
enforces the configured `top_k` limit.

## No-context behavior

When retrieval yields no usable source above the configured threshold, the LLM
provider is not called. The assistant message is persisted with lifecycle
status `fallback`, provider/model metadata remain empty, citations are empty,
and the API returns the exact configured answer:

```text
I could not find enough information in the selected documents.
```

The public chat response retains the existing `status=completed` convention.

## Security behavior

Retrieved text is untrusted user-prompt data, never a system instruction. XML
delimiter text in questions, history, or documents is encoded so it cannot
close the containing block. The system rules prohibit outside knowledge,
invented citations, following document instructions, prompt/credential
disclosure, and exposing retrieval internals. Leading UTF-8 BOMs and NUL bytes
are removed before chunking and again while building context. Structured logs
contain identifiers, counts, provider names, latency, and safe failure
categories—not prompts, document text, user questions, keys, tokens, database
URLs, raw vectors, or SDK errors.

## Windows commands

Activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Start the required containers:

```powershell
docker compose up -d redis embedding-service
```

Apply migrations:

```powershell
python -m alembic upgrade head
```

Run FastAPI:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If application control permits reload mode:

```powershell
python -m uvicorn app.main:app --reload
```

Run the existing Celery application in a separate terminal:

```powershell
python -m celery -A app.infrastructure.queue.celery_app.celery_app worker --loglevel=info --pool=solo
```

Run unit/regression tests and coverage:

```powershell
python -m pytest -v
python -m pytest --cov=app --cov-report=term-missing
```

Run optional Docker-backed integration tests:

```powershell
python -m pytest -m integration -v
```

The module commands above avoid relying on Windows-blocked `uvicorn.exe` or
`celery.exe` launchers.

## Manual fake-mode acceptance test

1. Set `LLM_PROVIDER=fake`, start Redis and the HTTP embedding service, apply
   migrations, then start Celery and FastAPI with the commands above.
2. Authenticate through `/api/v1/auth/register` and `/api/v1/auth/login`, then
   use the returned bearer token.
3. Upload a sufficiently long multi-topic TXT file with
   `POST /api/v1/documents/upload`.
4. Poll `GET /api/v1/documents/{document_id}/status` until `status=ready`.
5. Send the World Cup request shown above. Confirm the answer mentions the
   United States, Canada, and Mexico, includes a valid marker and citation,
   identifies the fake provider/model, and exposes no vector.
6. Continue the same conversation:

   ```json
   {
     "message": "When does it begin?",
     "conversation_id": "REAL_CONVERSATION_ID",
     "document_ids": ["REAL_DOCUMENT_ID"],
     "top_k": 5
   }
   ```

   Confirm the conversation ID is unchanged and the new grounded assistant
   message and citation appear in conversation history.
7. Ask `How do I cook chicken biryani?` against the same document. When the
   retrieval score is below `RETRIEVAL_MIN_SCORE`, confirm the exact fallback,
   empty citations, and no provider/model usage.

OpenAI tests use a mocked async SDK. A real OpenAI API key remains the only
external dependency for production OpenAI responses, and production OpenAI
generation has not been live-tested here.

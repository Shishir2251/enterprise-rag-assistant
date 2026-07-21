from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"
MAX_TEXTS_PER_REQUEST = 512
MAX_TEXT_LENGTH = 100_000

logger = logging.getLogger(__name__)


class ServiceConfigurationError(RuntimeError):
    """Raised when the embedding service configuration is invalid."""


class EmbeddingGenerationError(RuntimeError):
    """Raised when the model cannot produce a safe embedding response."""


@dataclass(frozen=True)
class ServiceSettings:
    model_name: str
    batch_size: int
    device: str

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            DEFAULT_MODEL_NAME,
        ).strip()
        device = os.getenv(
            "LOCAL_EMBEDDING_DEVICE",
            DEFAULT_DEVICE,
        ).strip().lower()
        raw_batch_size = os.getenv(
            "LOCAL_EMBEDDING_BATCH_SIZE",
            str(DEFAULT_BATCH_SIZE),
        ).strip()

        if not model_name:
            raise ServiceConfigurationError(
                "LOCAL_EMBEDDING_MODEL must not be empty"
            )
        if device != "cpu":
            raise ServiceConfigurationError(
                "LOCAL_EMBEDDING_DEVICE must be 'cpu'"
            )
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ServiceConfigurationError(
                "LOCAL_EMBEDDING_BATCH_SIZE must be a positive integer"
            ) from exc
        if batch_size <= 0:
            raise ServiceConfigurationError(
                "LOCAL_EMBEDDING_BATCH_SIZE must be a positive integer"
            )

        return cls(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )


@dataclass
class EmbeddingRuntime:
    model: Any
    settings: ServiceSettings
    dimension: int
    encode_lock: threading.Lock = field(default_factory=threading.Lock)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        try:
            with self.encode_lock:
                encoded = self.model.encode(
                    inputs,
                    batch_size=self.settings.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.settings.device,
                )
        except Exception as exc:
            raise EmbeddingGenerationError(
                "The embedding model failed to encode the request"
            ) from exc

        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        try:
            vectors = [
                [float(value) for value in raw_vector]
                for raw_vector in raw_vectors
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingGenerationError(
                "The embedding model returned invalid vectors"
            ) from exc

        if len(vectors) != len(inputs):
            raise EmbeddingGenerationError(
                "The embedding model returned an unexpected result count"
            )

        normalized_vectors: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingGenerationError(
                    "The embedding model returned an invalid vector dimension"
                )
            if any(not math.isfinite(value) for value in vector):
                raise EmbeddingGenerationError(
                    "The embedding model returned non-finite values"
                )
            magnitude = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(magnitude) or magnitude == 0.0:
                raise EmbeddingGenerationError(
                    "The embedding model returned an invalid vector magnitude"
                )
            normalized_vectors.append(
                [float(value / magnitude) for value in vector]
            )

        return normalized_vectors


class EmbedRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        max_length=MAX_TEXTS_PER_REQUEST,
    )

    @field_validator("texts")
    @classmethod
    def reject_blank_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding texts must not be blank")
        if any(len(text) > MAX_TEXT_LENGTH for text in texts):
            raise ValueError("Embedding text exceeds the maximum length")
        return texts


class EmbedQueryRequest(BaseModel):
    query: str = Field(max_length=MAX_TEXT_LENGTH)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("Query must not be blank")
        return query


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


class EmbedQueryResponse(BaseModel):
    model: str
    dimension: int
    embedding: list[float]


def _load_model(settings: ServiceSettings) -> Any:
    # Keep this import inside startup so importing the service module remains
    # lightweight and unit tests never need the native model dependencies.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        settings.model_name,
        device=settings.device,
    )


def _model_dimension(model: Any) -> int:
    try:
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = model.get_sentence_embedding_dimension
        dimension = int(dimension_getter())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ServiceConfigurationError(
            "The embedding model dimension could not be determined"
        ) from exc
    if dimension <= 0:
        raise ServiceConfigurationError(
            "The embedding model dimension must be positive"
        )
    return dimension


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = ServiceSettings.from_environment()
    logger.info(
        "Loading embedding model %s on %s",
        settings.model_name,
        settings.device,
    )
    model = _load_model(settings)
    application.state.embedding_runtime = EmbeddingRuntime(
        model=model,
        settings=settings,
        dimension=_model_dimension(model),
    )
    logger.info(
        "Embedding model is ready (model=%s, dimension=%d)",
        settings.model_name,
        application.state.embedding_runtime.dimension,
    )
    try:
        yield
    finally:
        application.state.embedding_runtime = None


app = FastAPI(
    title="Local Embedding Service",
    version="1.0.0",
    lifespan=lifespan,
)


def _runtime(request: Request) -> EmbeddingRuntime:
    runtime = getattr(request.app.state, "embedding_runtime", None)
    if not isinstance(runtime, EmbeddingRuntime):
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not ready.",
        )
    return runtime


def _safe_encode(
    runtime: EmbeddingRuntime,
    texts: Sequence[str],
) -> list[list[float]]:
    try:
        return runtime.encode(texts)
    except EmbeddingGenerationError:
        logger.exception("Embedding generation failed")
        raise HTTPException(
            status_code=500,
            detail="Embedding generation failed.",
        ) from None


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runtime = _runtime(request)
    return HealthResponse(
        status="ok",
        model=runtime.settings.model_name,
        dimension=runtime.dimension,
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, request: Request) -> EmbedResponse:
    runtime = _runtime(request)
    vectors = _safe_encode(runtime, payload.texts)
    return EmbedResponse(
        model=runtime.settings.model_name,
        dimension=runtime.dimension,
        embeddings=vectors,
    )


@app.post("/embed-query", response_model=EmbedQueryResponse)
def embed_query(
    payload: EmbedQueryRequest,
    request: Request,
) -> EmbedQueryResponse:
    runtime = _runtime(request)
    vectors = _safe_encode(runtime, [payload.query])
    if len(vectors) != 1:
        # Defensive check even though EmbeddingRuntime validates result counts.
        raise HTTPException(
            status_code=500,
            detail="Embedding generation failed.",
        )
    return EmbedQueryResponse(
        model=runtime.settings.model_name,
        dimension=runtime.dimension,
        embedding=vectors[0],
    )

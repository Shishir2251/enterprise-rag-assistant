import logging
import math
import weakref
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    ValidationError,
)


logger = logging.getLogger(__name__)

_INVALID_RESPONSE_MESSAGE = (
    "Embedding service returned an invalid response."
)
_TIMEOUT_MESSAGE = "Embedding service timed out."
_UNAVAILABLE_MESSAGE = "Embedding service is unavailable."


class HTTPEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by the private Docker HTTP service."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        dimensions: int,
        timeout_seconds: float = 30,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_base_url = (
            base_url.strip() if isinstance(base_url, str) else ""
        ).rstrip("/")
        normalized_model_name = (
            model_name.strip() if isinstance(model_name, str) else ""
        )
        parsed_base_url = urlsplit(normalized_base_url)

        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
            or parsed_base_url.query
            or parsed_base_url.fragment
        ):
            raise ConfigurationError(
                "HTTP_EMBEDDING_BASE_URL must be a valid HTTP URL"
            )
        if not normalized_model_name:
            raise ConfigurationError("EMBEDDING_MODEL must not be empty")
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ConfigurationError(
                "EMBEDDING_DIMENSION must be greater than zero"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ConfigurationError(
                "HTTP_EMBEDDING_TIMEOUT_SECONDS must be greater than zero"
            )

        self._base_url = normalized_base_url
        self._model_name = normalized_model_name
        self._dimensions = dimensions
        self._timeout_seconds = float(timeout_seconds)
        self._client = (
            client
            if client is not None
            else httpx.Client(timeout=self._timeout_seconds)
        )
        # FastAPI currently constructs providers per dependency resolution.
        # The finalizer prevents an internally-owned connection pool from
        # leaking when a caller does not have an explicit lifecycle hook.
        self._client_finalizer = (
            weakref.finalize(
                self,
                self._close_client_safely,
                self._client,
            )
            if client is None
            else None
        )

    @property
    def provider_name(self) -> str:
        return "http"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        if any(
            not isinstance(text, str) or not text.strip()
            for text in inputs
        ):
            raise ValidationError("Embedding text must not be empty")

        payload = self._post_json(
            "/embed",
            {"texts": inputs},
        )
        self._validate_response_metadata(payload)
        raw_embeddings = payload.get("embeddings")
        if (
            not isinstance(raw_embeddings, list)
            or len(raw_embeddings) != len(inputs)
        ):
            self._raise_invalid_response("unexpected embedding count")

        return [
            self._validate_vector(raw_vector)
            for raw_vector in raw_embeddings
        ]

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Query text must not be empty")

        payload = self._post_json(
            "/embed-query",
            {"query": text},
        )
        self._validate_response_metadata(payload)
        if "embedding" not in payload:
            self._raise_invalid_response("query embedding is missing")
        return self._validate_vector(payload["embedding"])

    def close(self) -> None:
        """Close the internally-created connection pool, if any."""
        if (
            self._client_finalizer is not None
            and self._client_finalizer.alive
        ):
            self._client_finalizer()

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> Mapping[str, Any]:
        try:
            response = self._client.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Embedding service request timed out")
            raise EmbeddingError(_TIMEOUT_MESSAGE) from exc
        except httpx.RequestError as exc:
            logger.error(
                "Embedding service request failed (%s)",
                type(exc).__name__,
            )
            raise EmbeddingError(_UNAVAILABLE_MESSAGE) from exc
        except Exception as exc:
            logger.error(
                "Embedding service request failed (%s)",
                type(exc).__name__,
            )
            raise EmbeddingError(_UNAVAILABLE_MESSAGE) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Embedding service returned HTTP status %s",
                response.status_code,
            )
            raise EmbeddingError(_INVALID_RESPONSE_MESSAGE) from exc
        except Exception as exc:
            logger.error(
                "Embedding service returned an unusable HTTP response (%s)",
                type(exc).__name__,
            )
            raise EmbeddingError(_INVALID_RESPONSE_MESSAGE) from exc

        try:
            response_payload = response.json()
        except (TypeError, ValueError) as exc:
            logger.error("Embedding service returned malformed JSON")
            raise EmbeddingError(_INVALID_RESPONSE_MESSAGE) from exc

        if not isinstance(response_payload, Mapping):
            self._raise_invalid_response("response body is not an object")
        return response_payload

    def _validate_response_metadata(
        self,
        payload: Mapping[str, Any],
    ) -> None:
        if payload.get("model") != self.model_name:
            self._raise_invalid_response("model does not match configuration")
        response_dimension = payload.get("dimension")
        if (
            isinstance(response_dimension, bool)
            or not isinstance(response_dimension, int)
            or response_dimension != self.dimensions
        ):
            self._raise_invalid_response(
                "declared dimension does not match configuration"
            )

    def _validate_vector(self, raw_vector: Any) -> list[float]:
        if (
            not isinstance(raw_vector, list)
            or len(raw_vector) != self.dimensions
        ):
            self._raise_invalid_response("vector dimension is invalid")

        vector: list[float] = []
        for value in raw_vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                self._raise_invalid_response("vector contains non-numeric data")
            converted_value = float(value)
            if not math.isfinite(converted_value):
                self._raise_invalid_response("vector contains non-finite data")
            vector.append(converted_value)
        return vector

    @staticmethod
    def _raise_invalid_response(reason: str) -> None:
        logger.error("Embedding service response validation failed (%s)", reason)
        raise EmbeddingError(_INVALID_RESPONSE_MESSAGE)

    @staticmethod
    def _close_client_safely(client: httpx.Client) -> None:
        try:
            client.close()
        except Exception as exc:
            logger.warning(
                "Embedding HTTP client could not be closed (%s)",
                type(exc).__name__,
            )

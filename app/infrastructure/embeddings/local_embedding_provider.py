import math
import threading
from collections.abc import Sequence
from typing import Any

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    ValidationError,
)


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _create_sentence_transformer(model_name: str, device: str) -> Any:
    """Import and construct the optional model only when local mode is used."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ConfigurationError(
            "sentence-transformers could not be imported for local embeddings"
        ) from exc

    try:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
    except OSError:
        # A cache miss is expected on first use. Permit the library to fetch
        # the model only after the cache-only attempt has failed.
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise ConfigurationError(
                "The configured local embedding model could not be loaded"
            ) from exc
    except Exception as exc:
        raise ConfigurationError(
            "The cached local embedding model could not be loaded"
        ) from exc


def _load_sentence_transformer(model_name: str, device: str) -> Any:
    """Return one model instance per model/device pair in this process."""
    cache_key = (model_name, device)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = _create_sentence_transformer(model_name, device)
            _MODEL_CACHE[cache_key] = model
        return model


class LocalEmbeddingProvider(IEmbeddingProvider):
    """CPU-friendly semantic embeddings backed by SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int = 384,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        normalized_model_name = model_name.strip()
        normalized_device = device.strip()
        if not normalized_model_name:
            raise ConfigurationError("LOCAL_EMBEDDING_MODEL must not be empty")
        if dimensions <= 0:
            raise ConfigurationError(
                "EMBEDDING_DIMENSION must be greater than zero"
            )
        if batch_size <= 0:
            raise ConfigurationError(
                "LOCAL_EMBEDDING_BATCH_SIZE must be greater than zero"
            )
        if not normalized_device:
            raise ConfigurationError("LOCAL_EMBEDDING_DEVICE must not be empty")

        self._model_name = normalized_model_name
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._device = normalized_device
        self._model: Any | None = None

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []

        normalized_inputs = [self._normalize_text(text) for text in inputs]
        if any(not text for text in normalized_inputs):
            raise ValidationError("Embedding text must not be empty")
        return self._encode(normalized_inputs)

    def embed_query(self, text: str) -> list[float]:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise ValidationError("Query text must not be empty")

        embeddings = self._encode([normalized_text])
        if len(embeddings) != 1:
            raise EmbeddingError(
                "Local embedding provider returned an unexpected result count"
            )
        return embeddings[0]

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_sentence_transformer(
                self.model_name,
                self._device,
            )
            try:
                model_dimensions = (
                    self._model.get_sentence_embedding_dimension()
                )
            except Exception as exc:
                raise EmbeddingError(
                    "Local embedding model dimension could not be determined"
                ) from exc
            if model_dimensions != self.dimensions:
                raise ConfigurationError(
                    "EMBEDDING_DIMENSION does not match the local model dimension"
                )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self._device,
            )
        except Exception as exc:
            raise EmbeddingError("Local embedding generation failed") from exc

        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        try:
            vectors = [
                [float(value) for value in raw_vector]
                for raw_vector in raw_vectors
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingError(
                "Local embedding provider returned invalid vectors"
            ) from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                "Local embedding provider returned an unexpected result count"
            )

        normalized_vectors: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    "Local embedding provider returned an invalid vector dimension"
                )
            if any(not math.isfinite(value) for value in vector):
                raise EmbeddingError(
                    "Local embedding provider returned non-finite values"
                )
            magnitude = math.sqrt(sum(value * value for value in vector))
            if magnitude == 0.0:
                raise EmbeddingError(
                    "Local embedding provider returned a zero vector"
                )
            normalized_vectors.append(
                [float(value / magnitude) for value in vector]
            )

        return normalized_vectors

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

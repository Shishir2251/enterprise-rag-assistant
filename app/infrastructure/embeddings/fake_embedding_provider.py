import hashlib
import math
import re
from collections.abc import Sequence

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.exceptions import ConfigurationError, ValidationError


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class FakeEmbeddingProvider(IEmbeddingProvider):
    """Deterministic hash embeddings for local pipeline development only."""

    def __init__(
        self,
        dimensions: int,
        model_name: str = "fake-embedding-v1",
    ) -> None:
        normalized_model_name = model_name.strip()
        if dimensions <= 0:
            raise ConfigurationError(
                "EMBEDDING_DIMENSION must be greater than zero"
            )
        if not normalized_model_name:
            raise ConfigurationError("EMBEDDING_MODEL must not be empty")

        self._dimensions = dimensions
        self._model_name = normalized_model_name

    @property
    def provider_name(self) -> str:
        return "fake"

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
        return [self._embed_text(text) for text in inputs]

    def embed_query(self, query: str) -> list[float]:
        if not self._normalize_text(query):
            raise ValidationError("Query text must not be empty")
        return self._embed_text(query)

    def _embed_text(self, text: str) -> list[float]:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise ValidationError("Embedding text must not be empty")

        tokens = TOKEN_PATTERN.findall(normalized_text)
        features = list(tokens)
        features.extend(
            f"{left}\x1f{right}"
            for left, right in zip(tokens, tokens[1:])
        )
        if not features:
            features.append(normalized_text)

        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            digest = hashlib.sha256(normalized_text.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:8], "big") % self.dimensions] = 1.0
            magnitude = 1.0

        return [float(value / magnitude) for value in vector]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

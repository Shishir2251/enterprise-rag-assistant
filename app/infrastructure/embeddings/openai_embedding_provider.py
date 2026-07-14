import logging
from collections.abc import Sequence

from openai import OpenAI

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.exceptions import EmbeddingError, ValidationError


logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(IEmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model_name: str,
        dimensions: int,
        client: OpenAI | None = None,
    ) -> None:
        normalized_api_key = api_key.strip()
        if not normalized_api_key or normalized_api_key.startswith("your_"):
            raise ValueError("OPENAI_API_KEY is not configured")
        if not model_name.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty")
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be greater than zero")

        self._model_name = model_name.strip()
        self._dimensions = dimensions
        self._client = client or OpenAI(api_key=normalized_api_key)

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
        if any(not text.strip() for text in inputs):
            raise ValidationError("Embedding text must not be empty")

        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=inputs,
                dimensions=self.dimensions,
            )
        except Exception as exc:
            logger.exception("OpenAI embedding request failed")
            raise EmbeddingError("Embedding provider request failed") from exc

        ordered_data = sorted(response.data, key=lambda item: item.index)
        indexes = [item.index for item in ordered_data]
        if indexes != list(range(len(inputs))):
            raise EmbeddingError(
                "Embedding provider returned invalid result indexes"
            )
        return [
            [float(value) for value in item.embedding]
            for item in ordered_data
        ]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValidationError("Query text must not be empty")

        embeddings = self.embed_texts([text])
        if len(embeddings) != 1:
            raise EmbeddingError(
                "Embedding provider returned an unexpected result count"
            )
        return embeddings[0]

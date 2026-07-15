import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.core.exceptions import ConfigurationError, ValidationError
from app.infrastructure.embeddings.embedding_provider_factory import (
    create_embedding_provider,
)
from app.infrastructure.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)


class FakeEmbeddingProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FakeEmbeddingProvider(dimensions=64)

    def test_same_normalized_text_produces_same_vector(self) -> None:
        first = self.provider.embed_query("  Enterprise   RAG\nAssistant ")
        second = self.provider.embed_query("enterprise rag assistant")

        self.assertEqual(first, second)

    def test_different_text_produces_different_vector(self) -> None:
        self.assertNotEqual(
            self.provider.embed_query("vector retrieval"),
            self.provider.embed_query("document ingestion"),
        )

    def test_vector_dimension_and_normalization_are_correct(self) -> None:
        vector = self.provider.embed_query("normalized local embedding")

        self.assertEqual(len(vector), 64)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in vector)),
            1.0,
            places=12,
        )

    def test_embed_texts_preserves_input_order(self) -> None:
        inputs = ["first input", "second input", "third input"]

        vectors = self.provider.embed_texts(inputs)

        self.assertEqual(
            vectors,
            [self.provider.embed_query(text) for text in inputs],
        )

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "Query text must not be empty",
        ):
            self.provider.embed_query("  \n\t ")

    def test_fake_mode_needs_no_api_key_and_does_not_create_openai(self) -> None:
        config = SimpleNamespace(
            EMBEDDING_PROVIDER="FaKe",
            EMBEDDING_MODEL="fake-embedding-v1",
            EMBEDDING_DIMENSION=32,
            OPENAI_API_KEY=SecretStr(""),
        )

        with patch(
            "app.infrastructure.embeddings.embedding_provider_factory."
            "OpenAIEmbeddingProvider"
        ) as openai_provider:
            provider = create_embedding_provider(config)

        self.assertIsInstance(provider, FakeEmbeddingProvider)
        self.assertEqual(provider.provider_name, "fake")
        openai_provider.assert_not_called()

    def test_unsupported_provider_has_clear_configuration_error(self) -> None:
        config = SimpleNamespace(
            EMBEDDING_PROVIDER="unknown",
            EMBEDDING_MODEL="model",
            EMBEDDING_DIMENSION=32,
            OPENAI_API_KEY=SecretStr(""),
        )

        with self.assertRaisesRegex(
            ConfigurationError,
            "Expected 'fake' or 'openai'",
        ):
            create_embedding_provider(config)


if __name__ == "__main__":
    unittest.main()

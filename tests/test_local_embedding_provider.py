import hashlib
import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

from pydantic import SecretStr

from app.core.exceptions import ConfigurationError, EmbeddingError, ValidationError
from app.infrastructure.embeddings.embedding_provider_factory import (
    create_embedding_provider,
)
from app.infrastructure.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from app.infrastructure.embeddings.local_embedding_provider import (
    LocalEmbeddingProvider,
    _create_sentence_transformer,
)


class FakeSentenceTransformer:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.encode_calls: list[tuple[list[str], dict]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(self, texts, **kwargs):
        inputs = list(texts)
        self.encode_calls.append((inputs, kwargs))
        vectors: list[list[float]] = []
        for text in inputs:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [0.0] * self.dimensions
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] = 2.0
            vector[int.from_bytes(digest[4:8], "big") % self.dimensions] += 1.0
            vectors.append(vector)
        return vectors


def make_config(provider_name: str) -> SimpleNamespace:
    model_name = (
        "sentence-transformers/all-MiniLM-L6-v2"
        if provider_name.lower() == "local"
        else "test-model"
    )
    return SimpleNamespace(
        EMBEDDING_PROVIDER=provider_name,
        EMBEDDING_MODEL=model_name,
        EMBEDDING_DIMENSION=384,
        LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        LOCAL_EMBEDDING_BATCH_SIZE=32,
        LOCAL_EMBEDDING_DEVICE="cpu",
        OPENAI_API_KEY=SecretStr("test-key"),
    )


class LocalEmbeddingProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeSentenceTransformer()
        self.model_loader = patch(
            "app.infrastructure.embeddings.local_embedding_provider."
            "_load_sentence_transformer",
            return_value=self.model,
        )
        self.load_model = self.model_loader.start()
        self.addCleanup(self.model_loader.stop)

    def test_defaults_require_no_api_key_and_create_384_dimensions(self) -> None:
        provider = LocalEmbeddingProvider()

        self.assertEqual(provider.provider_name, "local")
        self.assertEqual(provider.dimensions, 384)
        self.assertEqual(
            provider.model_name,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.load_model.assert_not_called()

    def test_model_is_loaded_lazily_once_per_provider(self) -> None:
        provider = LocalEmbeddingProvider()

        provider.embed_query("first query")
        provider.embed_query("second query")

        self.load_model.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            "cpu",
        )

    def test_same_normalized_text_produces_stable_vector(self) -> None:
        provider = LocalEmbeddingProvider()

        first = provider.embed_query("  When   is the World Cup?\n")
        second = provider.embed_query("When is the World Cup?")

        self.assertEqual(first, second)
        self.assertEqual(
            [call[0] for call in self.model.encode_calls],
            [["When is the World Cup?"], ["When is the World Cup?"]],
        )

    def test_different_text_produces_different_normalized_vectors(self) -> None:
        provider = LocalEmbeddingProvider()

        first = provider.embed_query("FIFA World Cup")
        second = provider.embed_query("database connection pooling")

        self.assertNotEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(value**2 for value in first)), 1.0)
        self.assertAlmostEqual(math.sqrt(sum(value**2 for value in second)), 1.0)

    def test_embed_texts_preserves_order_and_uses_configured_batch(self) -> None:
        provider = LocalEmbeddingProvider(batch_size=7)
        inputs = ["first", "second", "third"]

        vectors = provider.embed_texts(inputs)

        self.assertEqual(len(vectors), 3)
        self.assertEqual(self.model.encode_calls[0][0], inputs)
        self.assertEqual(self.model.encode_calls[0][1]["batch_size"], 7)
        self.assertEqual(
            self.model.encode_calls[0][1],
            {
                "batch_size": 7,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "device": "cpu",
            },
        )
        self.assertNotEqual(vectors[0], vectors[1])
        self.assertNotEqual(vectors[1], vectors[2])

    def test_empty_inputs_do_not_load_model_and_empty_query_is_rejected(self) -> None:
        provider = LocalEmbeddingProvider()

        self.assertEqual(provider.embed_texts([]), [])
        with self.assertRaisesRegex(ValidationError, "Query text"):
            provider.embed_query(" \n\t ")

        self.load_model.assert_not_called()

    def test_model_dimension_must_match_configuration(self) -> None:
        self.model.dimensions = 3
        provider = LocalEmbeddingProvider(dimensions=384)

        with self.assertRaisesRegex(ConfigurationError, "does not match"):
            provider.embed_query("dimension check")

    def test_invalid_encoder_vector_dimension_is_rejected(self) -> None:
        self.model.encode = Mock(return_value=[[1.0, 2.0]])
        provider = LocalEmbeddingProvider()

        with self.assertRaisesRegex(EmbeddingError, "vector dimension"):
            provider.embed_query("dimension check")


class LocalEmbeddingModelCacheTests(unittest.TestCase):
    def test_model_loader_uses_local_cache_before_network_enabled_load(self) -> None:
        sentence_transformers = ModuleType("sentence_transformers")
        constructor = Mock(return_value=FakeSentenceTransformer())
        sentence_transformers.SentenceTransformer = constructor

        with patch.dict(
            sys.modules,
            {"sentence_transformers": sentence_transformers},
        ):
            result = _create_sentence_transformer("test-model", "cpu")

        self.assertIsInstance(result, FakeSentenceTransformer)
        constructor.assert_called_once_with(
            "test-model",
            device="cpu",
            local_files_only=True,
        )

    def test_model_loader_allows_download_only_after_cache_miss(self) -> None:
        sentence_transformers = ModuleType("sentence_transformers")
        model = FakeSentenceTransformer()
        constructor = Mock(side_effect=[OSError("cache miss"), model])
        sentence_transformers.SentenceTransformer = constructor

        with patch.dict(
            sys.modules,
            {"sentence_transformers": sentence_transformers},
        ):
            result = _create_sentence_transformer("test-model", "cpu")

        self.assertIs(result, model)
        self.assertEqual(
            constructor.call_args_list,
            [
                call(
                    "test-model",
                    device="cpu",
                    local_files_only=True,
                ),
                call("test-model", device="cpu"),
            ],
        )

    def test_model_loader_does_not_enable_network_after_runtime_error(
        self,
    ) -> None:
        sentence_transformers = ModuleType("sentence_transformers")
        constructor = Mock(side_effect=RuntimeError("device unavailable"))
        sentence_transformers.SentenceTransformer = constructor

        with patch.dict(
            sys.modules,
            {"sentence_transformers": sentence_transformers},
        ):
            with self.assertRaisesRegex(
                ConfigurationError,
                "cached local embedding model",
            ):
                _create_sentence_transformer("test-model", "cpu")

        constructor.assert_called_once_with(
            "test-model",
            device="cpu",
            local_files_only=True,
        )

    def test_model_is_cached_per_process_model_and_device(self) -> None:
        from app.infrastructure.embeddings import local_embedding_provider

        model = FakeSentenceTransformer()
        local_embedding_provider._MODEL_CACHE.clear()
        self.addCleanup(local_embedding_provider._MODEL_CACHE.clear)

        with patch(
            "app.infrastructure.embeddings.local_embedding_provider."
            "_create_sentence_transformer",
            return_value=model,
        ) as create_model:
            first = LocalEmbeddingProvider()
            second = LocalEmbeddingProvider()
            first.embed_query("first")
            second.embed_query("second")

        create_model.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            "cpu",
        )


class EmbeddingProviderFactoryTests(unittest.TestCase):
    def test_local_selection_is_lazy_and_does_not_create_openai(self) -> None:
        config = make_config("LoCaL")
        config.OPENAI_API_KEY = None
        with (
            patch(
                "app.infrastructure.embeddings.local_embedding_provider."
                "_load_sentence_transformer"
            ) as load_model,
            patch(
                "app.infrastructure.embeddings.openai_embedding_provider."
                "OpenAIEmbeddingProvider"
            ) as openai_provider,
        ):
            provider = create_embedding_provider(config)

        self.assertIsInstance(provider, LocalEmbeddingProvider)
        self.assertEqual(provider.provider_name, "local")
        load_model.assert_not_called()
        openai_provider.assert_not_called()

    def test_fake_selection_does_not_load_local_model(self) -> None:
        with patch(
            "app.infrastructure.embeddings.local_embedding_provider."
            "_load_sentence_transformer"
        ) as load_model:
            provider = create_embedding_provider(make_config("fake"))

        self.assertIsInstance(provider, FakeEmbeddingProvider)
        load_model.assert_not_called()

    def test_openai_selection_preserves_existing_configuration(self) -> None:
        config = make_config("openai")
        configured_provider = Mock()
        with patch(
            "app.infrastructure.embeddings.openai_embedding_provider."
            "OpenAIEmbeddingProvider",
            return_value=configured_provider,
        ) as openai_provider:
            result = create_embedding_provider(config)

        self.assertIs(result, configured_provider)
        openai_provider.assert_called_once_with(
            api_key="test-key",
            model_name="test-model",
            dimensions=384,
        )

    def test_mismatched_local_model_names_are_rejected(self) -> None:
        config = make_config("local")
        config.EMBEDDING_MODEL = "different-model"

        with self.assertRaisesRegex(ConfigurationError, "must match"):
            create_embedding_provider(config)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from app.core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    ValidationError,
)
from app.infrastructure.embeddings.embedding_provider_factory import (
    create_embedding_provider,
)
from app.infrastructure.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from app.infrastructure.embeddings.http_embedding_provider import (
    HTTPEmbeddingProvider,
)


BASE_URL = "http://127.0.0.1:8090"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_response(
    payload: object,
    *,
    status_code: int = 200,
    path: str = "/embed",
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", f"{BASE_URL}{path}"),
    )


def make_provider(
    client: Mock,
    *,
    dimensions: int = 3,
) -> HTTPEmbeddingProvider:
    return HTTPEmbeddingProvider(
        base_url=f" {BASE_URL}/ ",
        model_name=MODEL_NAME,
        dimensions=dimensions,
        timeout_seconds=30,
        client=client,
    )


def valid_batch_payload() -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "dimension": 3,
        "embeddings": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    }


class HTTPEmbeddingProviderTests(unittest.TestCase):
    def test_embed_texts_sends_one_batch_and_preserves_response_order(
        self,
    ) -> None:
        client = Mock(spec=httpx.Client)
        client.post.return_value = make_response(valid_batch_payload())
        provider = make_provider(client)
        texts = ["first", "second", "third"]

        result = provider.embed_texts(texts)

        self.assertEqual(
            result,
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        client.post.assert_called_once_with(
            f"{BASE_URL}/embed",
            json={"texts": texts},
            timeout=30.0,
        )

    def test_embed_query_uses_query_endpoint(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.return_value = make_response(
            {
                "model": MODEL_NAME,
                "dimension": 3,
                "embedding": [0.25, 0.5, 0.75],
            },
            path="/embed-query",
        )
        provider = make_provider(client)

        result = provider.embed_query("semantic query")

        self.assertEqual(result, [0.25, 0.5, 0.75])
        client.post.assert_called_once_with(
            f"{BASE_URL}/embed-query",
            json={"query": "semantic query"},
            timeout=30.0,
        )

    def test_empty_inputs_are_rejected_without_http_requests(self) -> None:
        client = Mock(spec=httpx.Client)
        provider = make_provider(client)

        self.assertEqual(provider.embed_texts([]), [])
        with self.assertRaisesRegex(
            ValidationError,
            "Embedding text must not be empty",
        ):
            provider.embed_texts(["valid", " \n "])
        with self.assertRaisesRegex(
            ValidationError,
            "Query text must not be empty",
        ):
            provider.embed_query(" \t ")

        client.post.assert_not_called()

    def test_timeout_is_sanitized(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.side_effect = httpx.ReadTimeout(
            "internal timeout details",
            request=httpx.Request("POST", f"{BASE_URL}/embed"),
        )
        provider = make_provider(client)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_texts(["sensitive document content"])

        self.assertEqual(str(caught.exception), "Embedding service timed out.")
        self.assertNotIn("internal", str(caught.exception))

    def test_connection_failure_is_sanitized(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.side_effect = httpx.ConnectError(
            "docker host and port details",
            request=httpx.Request("POST", f"{BASE_URL}/embed"),
        )
        provider = make_provider(client)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_texts(["content"])

        self.assertEqual(
            str(caught.exception),
            "Embedding service is unavailable.",
        )
        self.assertNotIn("docker", str(caught.exception))

    def test_malformed_json_is_rejected_safely(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.return_value = httpx.Response(
            status_code=200,
            content=b"not-json",
            request=httpx.Request("POST", f"{BASE_URL}/embed"),
        )
        provider = make_provider(client)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_texts(["content"])

        self.assertEqual(
            str(caught.exception),
            "Embedding service returned an invalid response.",
        )

    def test_unexpected_http_status_is_rejected_safely(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.return_value = make_response(
            {"detail": "private container stack trace"},
            status_code=500,
        )
        provider = make_provider(client)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_texts(["content"])

        self.assertEqual(
            str(caught.exception),
            "Embedding service returned an invalid response.",
        )
        self.assertNotIn("stack trace", str(caught.exception))

    def test_wrong_model_is_rejected(self) -> None:
        client = Mock(spec=httpx.Client)
        payload = valid_batch_payload()
        payload["model"] = "unexpected-model"
        client.post.return_value = make_response(payload)
        provider = make_provider(client)

        with self.assertRaisesRegex(
            EmbeddingError,
            "Embedding service returned an invalid response",
        ):
            provider.embed_texts(["first", "second", "third"])

    def test_wrong_declared_dimension_is_rejected(self) -> None:
        client = Mock(spec=httpx.Client)
        payload = valid_batch_payload()
        payload["dimension"] = 4
        client.post.return_value = make_response(payload)
        provider = make_provider(client)

        with self.assertRaisesRegex(
            EmbeddingError,
            "Embedding service returned an invalid response",
        ):
            provider.embed_texts(["first", "second", "third"])

    def test_partial_batch_is_rejected(self) -> None:
        client = Mock(spec=httpx.Client)
        payload = valid_batch_payload()
        payload["embeddings"] = [[1, 0, 0], [0, 1, 0]]
        client.post.return_value = make_response(payload)
        provider = make_provider(client)

        with self.assertRaisesRegex(
            EmbeddingError,
            "Embedding service returned an invalid response",
        ):
            provider.embed_texts(["first", "second", "third"])

    def test_wrong_vector_dimension_and_non_finite_values_are_rejected(
        self,
    ) -> None:
        invalid_vectors = ([1, 0], [1, float("nan"), 0])
        for invalid_vector in invalid_vectors:
            with self.subTest(invalid_vector=invalid_vector):
                client = Mock(spec=httpx.Client)
                response = Mock(spec=httpx.Response)
                response.json.return_value = {
                    "model": MODEL_NAME,
                    "dimension": 3,
                    "embedding": invalid_vector,
                }
                client.post.return_value = response
                provider = make_provider(client)

                with self.assertRaisesRegex(
                    EmbeddingError,
                    "Embedding service returned an invalid response",
                ):
                    provider.embed_query("query")

    def test_provider_reuses_its_client(self) -> None:
        client = Mock(spec=httpx.Client)
        client.post.side_effect = [
            make_response(
                {
                    "model": MODEL_NAME,
                    "dimension": 3,
                    "embedding": [1, 0, 0],
                },
                path="/embed-query",
            ),
            make_response(
                {
                    "model": MODEL_NAME,
                    "dimension": 3,
                    "embedding": [0, 1, 0],
                },
                path="/embed-query",
            ),
        ]
        provider = make_provider(client)

        provider.embed_query("first")
        provider.embed_query("second")

        self.assertEqual(client.post.call_count, 2)

    def test_internally_owned_client_is_closed_once(self) -> None:
        client = Mock(spec=httpx.Client)
        with patch(
            "app.infrastructure.embeddings.http_embedding_provider.httpx.Client",
            return_value=client,
        ) as client_constructor:
            provider = HTTPEmbeddingProvider(
                base_url=BASE_URL,
                model_name=MODEL_NAME,
                dimensions=3,
                timeout_seconds=30,
            )

        provider.close()
        provider.close()

        client_constructor.assert_called_once_with(timeout=30.0)
        client.close.assert_called_once_with()

    def test_invalid_configuration_is_rejected_before_client_creation(
        self,
    ) -> None:
        with patch(
            "app.infrastructure.embeddings.http_embedding_provider.httpx.Client"
        ) as client_constructor:
            with self.assertRaises(ConfigurationError):
                HTTPEmbeddingProvider(
                    base_url="file:///tmp/embedding.sock",
                    model_name=MODEL_NAME,
                    dimensions=3,
                )

        client_constructor.assert_not_called()


class HTTPEmbeddingProviderFactoryTests(unittest.TestCase):
    @staticmethod
    def make_config(provider_name: str = "http") -> SimpleNamespace:
        return SimpleNamespace(
            EMBEDDING_PROVIDER=provider_name,
            EMBEDDING_MODEL=MODEL_NAME,
            EMBEDDING_DIMENSION=384,
            HTTP_EMBEDDING_BASE_URL=BASE_URL,
            HTTP_EMBEDDING_TIMEOUT_SECONDS=30,
            OPENAI_API_KEY=None,
        )

    def test_http_provider_is_selected_without_an_api_key(self) -> None:
        config = self.make_config("HtTp")
        configured_provider = Mock()
        with patch(
            "app.infrastructure.embeddings.http_embedding_provider."
            "HTTPEmbeddingProvider",
            return_value=configured_provider,
        ) as provider_constructor:
            result = create_embedding_provider(config)

        self.assertIs(result, configured_provider)
        provider_constructor.assert_called_once_with(
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            dimensions=384,
            timeout_seconds=30,
        )

    def test_http_selection_does_not_import_sentence_transformers(self) -> None:
        config = self.make_config()
        client = Mock(spec=httpx.Client)

        with (
            patch.dict(sys.modules, {"sentence_transformers": None}),
            patch(
                "app.infrastructure.embeddings.http_embedding_provider."
                "httpx.Client",
                return_value=client,
            ),
        ):
            provider = create_embedding_provider(config)

        self.assertIsInstance(provider, HTTPEmbeddingProvider)
        self.assertEqual(provider.provider_name, "http")

    def test_http_selection_does_not_construct_openai(self) -> None:
        config = self.make_config()
        client = Mock(spec=httpx.Client)
        with (
            patch(
                "app.infrastructure.embeddings.http_embedding_provider."
                "httpx.Client",
                return_value=client,
            ),
            patch(
                "app.infrastructure.embeddings.openai_embedding_provider."
                "OpenAIEmbeddingProvider"
            ) as openai_provider,
        ):
            provider = create_embedding_provider(config)

        self.assertIsInstance(provider, HTTPEmbeddingProvider)
        openai_provider.assert_not_called()

    def test_fake_selection_does_not_construct_an_http_client(self) -> None:
        config = SimpleNamespace(
            EMBEDDING_PROVIDER="fake",
            EMBEDDING_MODEL="fake-embedding-v1",
            EMBEDDING_DIMENSION=3,
            OPENAI_API_KEY=None,
        )
        with patch(
            "app.infrastructure.embeddings.http_embedding_provider.httpx.Client"
        ) as client_constructor:
            provider = create_embedding_provider(config)

        self.assertIsInstance(provider, FakeEmbeddingProvider)
        client_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()

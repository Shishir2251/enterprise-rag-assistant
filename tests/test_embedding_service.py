import math
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from embedding_service import app as service_module


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.encode_calls: list[tuple[list[str], dict]] = []
        self.error: Exception | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        inputs = list(texts)
        self.encode_calls.append((inputs, kwargs))
        if self.error is not None:
            raise self.error

        vectors = []
        for index, _text in enumerate(inputs):
            vector = [0.0] * self.dimension
            vector[index % self.dimension] = float(index + 2)
            vector[(index + 7) % self.dimension] += 1.0
            vectors.append(vector)
        return vectors


class EmbeddingServiceEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeEmbeddingModel()
        self.environment = patch.dict(
            os.environ,
            {
                "LOCAL_EMBEDDING_MODEL": (
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                "LOCAL_EMBEDDING_BATCH_SIZE": "7",
                "LOCAL_EMBEDDING_DEVICE": "cpu",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.model_loader = patch.object(
            service_module,
            "_load_model",
            return_value=self.model,
        )
        self.load_model = self.model_loader.start()
        self.addCleanup(self.model_loader.stop)

        self.client_context = TestClient(service_module.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_startup_loads_model_once_and_health_reports_metadata(self) -> None:
        first = self.client.get("/health")
        second = self.client.get("/health")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(
            first.json(),
            {
                "status": "ok",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
        )
        self.assertEqual(second.json(), first.json())
        self.load_model.assert_called_once()

    def test_embed_uses_one_batched_call_and_preserves_input_order(self) -> None:
        texts = ["first text", "second text", "third text"]

        response = self.client.post("/embed", json={"texts": texts})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(payload["dimension"], 384)
        self.assertEqual(len(payload["embeddings"]), len(texts))
        self.assertEqual(self.model.encode_calls[0][0], texts)
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
        self.assertGreater(payload["embeddings"][0][0], 0.0)
        self.assertGreater(payload["embeddings"][1][1], 0.0)
        self.assertGreater(payload["embeddings"][2][2], 0.0)
        for vector in payload["embeddings"]:
            self.assertEqual(len(vector), 384)
            self.assertAlmostEqual(
                math.sqrt(sum(value * value for value in vector)),
                1.0,
            )

    def test_embed_query_returns_one_normalized_vector(self) -> None:
        response = self.client.post(
            "/embed-query",
            json={"query": "semantic query"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["dimension"], 384)
        self.assertEqual(len(payload["embedding"]), 384)
        self.assertEqual(self.model.encode_calls[0][0], ["semantic query"])
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in payload["embedding"])),
            1.0,
        )

    def test_empty_batch_and_blank_values_are_rejected_before_encoding(self) -> None:
        responses = [
            self.client.post("/embed", json={"texts": []}),
            self.client.post("/embed", json={"texts": ["valid", "  \n"]}),
            self.client.post("/embed-query", json={"query": " \t "}),
        ]

        self.assertTrue(all(response.status_code == 422 for response in responses))
        self.assertEqual(self.model.encode_calls, [])

    def test_model_failure_is_sanitized(self) -> None:
        self.model.error = RuntimeError("private model implementation detail")

        response = self.client.post(
            "/embed-query",
            json={"query": "safe query"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Embedding generation failed."},
        )
        self.assertNotIn("private model", response.text)

    def test_invalid_model_vector_dimension_is_sanitized(self) -> None:
        self.model.dimension = 2

        response = self.client.post(
            "/embed-query",
            json={"query": "safe query"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Embedding generation failed."},
        )


class EmbeddingServiceSettingsTests(unittest.TestCase):
    def test_non_cpu_device_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_EMBEDDING_DEVICE": "cuda"},
        ):
            with self.assertRaisesRegex(
                service_module.ServiceConfigurationError,
                "must be 'cpu'",
            ):
                service_module.ServiceSettings.from_environment()

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_EMBEDDING_BATCH_SIZE": "0"},
        ):
            with self.assertRaisesRegex(
                service_module.ServiceConfigurationError,
                "positive integer",
            ):
                service_module.ServiceSettings.from_environment()


if __name__ == "__main__":
    unittest.main()

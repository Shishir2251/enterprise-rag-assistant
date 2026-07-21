import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from openai import AuthenticationError as OpenAIAuthenticationError
from sqlalchemy.dialects import postgresql

from app.business.services.embedding_service import EmbeddingService
from app.core.exceptions import (
    ConfigurationError,
    ConflictError,
    EmbeddingError,
    NotFoundError,
    ValidationError,
)
from app.data_access.models.document_chunk_model import DocumentChunkModel
from app.data_access.models.document_model import DocumentModel, DocumentStatus
from app.data_access.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_embedding_provider,
    get_embedding_service,
)


def make_document(status: str = DocumentStatus.COMPLETED.value) -> DocumentModel:
    return DocumentModel(
        id="document-id",
        owner_id="owner-id",
        original_name="report.txt",
        stored_name="generated.txt",
        file_path="uploads/owner-id/generated.txt",
        mime_type="text/plain",
        file_size=100,
        status=status,
    )


def make_chunks(count: int) -> list[DocumentChunkModel]:
    return [
        DocumentChunkModel(
            id=f"chunk-{index}",
            document_id="document-id",
            chunk_index=index,
            content=f"chunk content {index}",
            character_count=15,
        )
        for index in range(count)
    ]


class FakeDocumentRepository:
    def __init__(self, document: DocumentModel | None) -> None:
        self.document = document

    def get_by_id(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel | None:
        if (
            self.document is not None
            and self.document.id == document_id
            and self.document.owner_id == owner_id
        ):
            return self.document
        return None


class FakeChunkRepository:
    def __init__(self, chunks: list[DocumentChunkModel]) -> None:
        self.chunks = chunks
        self.saved_batches: list[list[DocumentChunkModel]] = []

    def list_without_embeddings(
        self,
        document_id: str,
    ) -> list[DocumentChunkModel]:
        return [
            chunk
            for chunk in self.chunks
            if chunk.document_id == document_id and chunk.embedding is None
        ]

    def list_stale_embeddings(
        self,
        document_id: str,
        model_name: str,
        provider_name: str,
    ) -> list[DocumentChunkModel]:
        return [
            chunk
            for chunk in self.chunks
            if chunk.document_id == document_id
            and (
                chunk.embedding is None
                or chunk.embedding_model != model_name
                or chunk.embedding_provider != provider_name
            )
        ]

    def save_embeddings(
        self,
        chunks: list[DocumentChunkModel],
        model_name: str,
        provider_name: str,
        embedded_at,
    ) -> None:
        for chunk in chunks:
            chunk.embedding_model = model_name
            chunk.embedding_provider = provider_name
            chunk.embedded_at = embedded_at
        self.saved_batches.append(list(chunks))

    def clear_embeddings(self, document_id: str) -> int:
        cleared_count = 0
        for chunk in self.chunks:
            if chunk.document_id == document_id and (
                chunk.embedding is not None
                or chunk.embedding_model is not None
                or chunk.embedding_provider is not None
                or chunk.embedded_at is not None
            ):
                chunk.embedding = None
                chunk.embedding_model = None
                chunk.embedding_provider = None
                chunk.embedded_at = None
                cleared_count += 1
        return cleared_count


class FakeEmbeddingProvider:
    provider_name = "test-provider"
    model_name = "test-embedding-model"

    def __init__(
        self,
        dimensions: int = 3,
        result_count_offset: int = 0,
        vector_dimensions: int | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.result_count_offset = result_count_offset
        self.vector_dimensions = vector_dimensions or dimensions
        self.calls: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        inputs = list(texts)
        self.calls.append(inputs)
        result_count = max(0, len(inputs) + self.result_count_offset)
        return [
            [float(index)] * self.vector_dimensions
            for index in range(result_count)
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class EmbeddingProviderTests(unittest.TestCase):
    def test_placeholder_api_key_is_reported_as_configuration_error(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "OPENAI_API_KEY is not configured",
        ):
            OpenAIEmbeddingProvider(
                api_key="your_openai_api_key",
                model_name="test-model",
                dimensions=3,
            )

    def test_empty_embedding_input_does_not_call_openai(self) -> None:
        client = Mock()
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="test-model",
            dimensions=3,
            client=client,
        )

        self.assertEqual(provider.embed_texts([]), [])
        client.embeddings.create.assert_not_called()

    def test_embedding_result_order_follows_input_indexes(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2, 2, 2]),
                SimpleNamespace(index=0, embedding=[1, 1, 1]),
            ]
        )
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="test-model",
            dimensions=3,
            client=client,
        )

        result = provider.embed_texts(["first", "second"])

        self.assertEqual(result, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        client.embeddings.create.assert_called_once_with(
            model="test-model",
            input=["first", "second"],
            dimensions=3,
        )

    def test_rejected_api_key_returns_safe_configuration_error(self) -> None:
        client = Mock()
        response = httpx.Response(
            status_code=401,
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/embeddings",
            ),
        )
        client.embeddings.create.side_effect = OpenAIAuthenticationError(
            "Incorrect API key",
            response=response,
            body={"error": {"code": "invalid_api_key"}},
        )
        provider = OpenAIEmbeddingProvider(
            api_key="sk-proj-test-key",
            model_name="test-model",
            dimensions=3,
            client=client,
        )

        with patch(
            "app.infrastructure.embeddings.openai_embedding_provider."
            "logger.warning"
        ):
            with self.assertRaisesRegex(
                ConfigurationError,
                "was rejected by OpenAI",
            ):
                provider.embed_texts(["content"])

    def test_empty_query_is_rejected(self) -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="test-model",
            dimensions=3,
            client=Mock(),
        )

        with self.assertRaisesRegex(
            ValidationError,
            "Query text must not be empty",
        ):
            provider.embed_query("   ")


class EmbeddingServiceTests(unittest.TestCase):
    def make_service(
        self,
        chunks: list[DocumentChunkModel],
        provider: FakeEmbeddingProvider,
        document: DocumentModel | None = None,
        batch_size: int = 2,
    ) -> tuple[EmbeddingService, FakeChunkRepository]:
        repository = FakeChunkRepository(chunks)
        service = EmbeddingService(
            document_repository=FakeDocumentRepository(
                document if document is not None else make_document()
            ),
            chunk_repository=repository,
            embedding_provider=provider,
            batch_size=batch_size,
        )
        return service, repository

    def test_embedding_count_mismatch_is_rejected(self) -> None:
        provider = FakeEmbeddingProvider(result_count_offset=-1)
        service, _ = self.make_service(make_chunks(2), provider)

        with self.assertRaisesRegex(EmbeddingError, "count"):
            service.embed_document("document-id", "owner-id")

    def test_vector_dimension_mismatch_is_rejected(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=3, vector_dimensions=2)
        service, _ = self.make_service(make_chunks(2), provider)

        with self.assertRaisesRegex(EmbeddingError, "dimension"):
            service.embed_document("document-id", "owner-id")

    def test_no_stale_chunks_returns_zero_without_provider_call(self) -> None:
        chunks = make_chunks(1)
        chunks[0].embedding = [0.0, 0.0, 0.0]
        provider = FakeEmbeddingProvider()
        chunks[0].embedding_model = provider.model_name
        chunks[0].embedding_provider = provider.provider_name
        service, _ = self.make_service(chunks, provider)

        self.assertEqual(
            service.embed_document("document-id", "owner-id"),
            0,
        )
        self.assertEqual(provider.calls, [])

    def test_model_or_provider_mismatch_is_reembedded(self) -> None:
        for stale_field in ("model", "provider"):
            with self.subTest(stale_field=stale_field):
                chunks = make_chunks(1)
                chunks[0].embedding = [0.0, 0.0, 0.0]
                provider = FakeEmbeddingProvider()
                chunks[0].embedding_model = provider.model_name
                chunks[0].embedding_provider = provider.provider_name
                if stale_field == "model":
                    chunks[0].embedding_model = "legacy-model"
                else:
                    chunks[0].embedding_provider = "legacy-provider"
                service, repository = self.make_service(chunks, provider)

                self.assertEqual(
                    service.embed_document("document-id", "owner-id"),
                    1,
                )
                self.assertEqual(
                    repository.chunks[0].embedding_model,
                    provider.model_name,
                )
                self.assertEqual(
                    repository.chunks[0].embedding_provider,
                    provider.provider_name,
                )

    def test_successful_batch_embedding_and_second_call_is_idempotent(self) -> None:
        provider = FakeEmbeddingProvider()
        service, repository = self.make_service(
            make_chunks(5),
            provider,
            batch_size=2,
        )

        first_count = service.embed_document("document-id", "owner-id")
        second_count = service.embed_document("document-id", "owner-id")

        self.assertEqual(first_count, 5)
        self.assertEqual(second_count, 0)
        self.assertEqual([len(call) for call in provider.calls], [2, 2, 1])
        self.assertEqual([len(batch) for batch in repository.saved_batches], [2, 2, 1])
        self.assertTrue(all(chunk.embedding is not None for chunk in repository.chunks))
        self.assertTrue(
            all(
                chunk.embedding_model == provider.model_name
                and chunk.embedding_provider == provider.provider_name
                for chunk in repository.chunks
            )
        )

    def test_document_owned_by_another_user_is_not_found(self) -> None:
        provider = FakeEmbeddingProvider()
        service, _ = self.make_service(make_chunks(1), provider)

        with self.assertRaises(NotFoundError):
            service.embed_document("document-id", "different-owner")
        self.assertEqual(provider.calls, [])

    def test_document_must_be_processed_before_embedding(self) -> None:
        provider = FakeEmbeddingProvider()
        service, _ = self.make_service(
            make_chunks(1),
            provider,
            document=make_document(DocumentStatus.UPLOADED.value),
        )

        with self.assertRaises(ConflictError):
            service.embed_document("document-id", "owner-id")
        self.assertEqual(provider.calls, [])

    def test_embeddings_can_be_cleared_and_regenerated(self) -> None:
        provider = FakeEmbeddingProvider()
        service, repository = self.make_service(make_chunks(2), provider)
        self.assertEqual(service.embed_document("document-id", "owner-id"), 2)

        cleared_count = service.clear_document_embeddings(
            "document-id",
            "owner-id",
        )

        self.assertEqual(cleared_count, 2)
        self.assertTrue(
            all(
                chunk.embedding is None
                and chunk.embedding_model is None
                and chunk.embedding_provider is None
                and chunk.embedded_at is None
                for chunk in repository.chunks
            )
        )
        self.assertEqual(service.embed_document("document-id", "owner-id"), 2)
        self.assertEqual([len(call) for call in provider.calls], [2, 2])

    def test_reindex_replaces_embeddings_without_duplicating_chunks(self) -> None:
        chunks = make_chunks(2)
        original_chunk_ids = [chunk.id for chunk in chunks]
        provider = FakeEmbeddingProvider()
        service, repository = self.make_service(chunks, provider)

        first_count = service.reindex_document_chunks(
            "document-id",
            "owner-id",
        )
        second_count = service.reindex_document_chunks(
            "document-id",
            "owner-id",
        )

        self.assertEqual((first_count, second_count), (2, 2))
        self.assertEqual(
            [chunk.id for chunk in repository.chunks],
            original_chunk_ids,
        )
        self.assertEqual(len(repository.chunks), 2)
        self.assertTrue(
            all(
                chunk.embedding_model == provider.model_name
                and chunk.embedding_provider == provider.provider_name
                for chunk in repository.chunks
            )
        )

    def test_clear_embeddings_checks_document_ownership(self) -> None:
        provider = FakeEmbeddingProvider()
        service, _ = self.make_service(make_chunks(1), provider)

        with self.assertRaises(NotFoundError):
            service.clear_document_embeddings(
                "document-id",
                "different-owner",
            )


class EmbeddingEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_missing_authentication_returns_401(self) -> None:
        response = TestClient(app).post(
            "/api/v1/documents/document-id/embed"
        )

        self.assertEqual(response.status_code, 401)

    def test_missing_openai_configuration_returns_503(self) -> None:
        def missing_provider():
            raise ConfigurationError("OPENAI_API_KEY is not configured")

        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_embedding_provider] = missing_provider

        response = TestClient(app).post(
            "/api/v1/documents/document-id/embed"
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "OPENAI_API_KEY is not configured"},
        )

    def test_unprocessed_document_returns_409(self) -> None:
        embedding_service = Mock()
        embedding_service.embed_document.side_effect = ConflictError(
            "Document processing must complete before embedding"
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_embedding_service] = lambda: embedding_service

        response = TestClient(app).post(
            "/api/v1/documents/document-id/embed"
        )

        self.assertEqual(response.status_code, 409)

    def test_success_response_does_not_expose_vectors(self) -> None:
        embedding_service = Mock()
        embedding_service.embed_document.return_value = 3
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_embedding_service] = lambda: embedding_service
        app.dependency_overrides[get_embedding_provider] = lambda: SimpleNamespace(
            provider_name="fake",
            model_name="fake-embedding-v1",
        )

        response = TestClient(app).post(
            "/api/v1/documents/document-id/embed"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "document_id": "document-id",
                "embedded_chunks": 3,
                "status": "completed",
                "embedding_provider": "fake",
                "embedding_model": "fake-embedding-v1",
            },
        )
        self.assertNotIn("embedding", response.json())

    def test_clear_embeddings_endpoint_returns_count(self) -> None:
        embedding_service = Mock()
        embedding_service.clear_document_embeddings.return_value = 3
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_embedding_service] = lambda: embedding_service

        response = TestClient(app).delete(
            "/api/v1/documents/document-id/embeddings"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "document_id": "document-id",
                "cleared_chunks": 3,
                "status": "cleared",
            },
        )


class DocumentChunkRepositoryResetTests(unittest.TestCase):
    def test_reset_nulls_only_embedding_metadata_for_selected_document(self) -> None:
        session = Mock()
        session.execute.return_value.rowcount = 2
        repository = DocumentChunkRepository(session)

        cleared_count = repository.clear_embeddings("document-id")

        statement = session.execute.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertEqual(cleared_count, 2)
        self.assertIn("UPDATE document_chunks SET", sql)
        self.assertIn("embedding=", sql)
        self.assertIn("embedding_model=", sql)
        self.assertIn("embedding_provider=", sql)
        self.assertIn("embedded_at=", sql)
        self.assertIn("document_chunks.document_id =", sql)
        self.assertNotIn("DELETE", sql)
        self.assertEqual(compiled.params["document_id_1"], "document-id")
        self.assertIsNone(compiled.params["embedding"])
        self.assertIsNone(compiled.params["embedding_model"])
        self.assertIsNone(compiled.params["embedding_provider"])
        self.assertIsNone(compiled.params["embedded_at"])
        session.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

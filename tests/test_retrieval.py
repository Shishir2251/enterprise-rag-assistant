import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.services.retrieval_service import RetrievalService
from app.core.exceptions import EmbeddingError, RetrievalError, ValidationError
from app.data_access.repositories.pgvector_repository import PgVectorRepository
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_retrieval_service,
)


class FakeEmbeddingProvider:
    model_name = "test-model"

    def __init__(
        self,
        vector: list[float] | None = None,
        dimensions: int = 3,
    ) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.dimensions = dimensions
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector

    def embed_texts(self, texts):
        return [self.vector for _ in texts]


class FakeVectorRepository:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict] = []

    def similarity_search(self, **kwargs) -> list[RetrievalResult]:
        self.calls.append(kwargs)
        return self.results


def make_result(score: float = 0.91) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-id",
        document_id="document-id",
        document_name="report.pdf",
        chunk_index=0,
        content="Relevant document content",
        page_number=1,
        similarity_score=score,
    )


class RetrievalServiceTests(unittest.TestCase):
    def make_service(
        self,
        provider: FakeEmbeddingProvider | None = None,
        repository: FakeVectorRepository | None = None,
    ) -> tuple[RetrievalService, FakeEmbeddingProvider, FakeVectorRepository]:
        embedding_provider = provider or FakeEmbeddingProvider()
        vector_repository = repository or FakeVectorRepository()
        service = RetrievalService(
            vector_repository=vector_repository,
            embedding_provider=embedding_provider,
            default_top_k=5,
            minimum_score=0.30,
        )
        return service, embedding_provider, vector_repository

    def test_empty_search_query_is_rejected(self) -> None:
        service, provider, repository = self.make_service()

        with self.assertRaisesRegex(ValidationError, "must not be empty"):
            service.search("   \n\t", "owner-id")

        self.assertEqual(provider.queries, [])
        self.assertEqual(repository.calls, [])

    def test_top_k_greater_than_maximum_is_rejected(self) -> None:
        service, provider, repository = self.make_service()

        with self.assertRaisesRegex(ValidationError, "between 1 and 20"):
            service.search("valid query", "owner-id", top_k=21)

        self.assertEqual(provider.queries, [])
        self.assertEqual(repository.calls, [])

    def test_successful_query_embedding_and_repository_call(self) -> None:
        repository = FakeVectorRepository([make_result()])
        service, provider, _ = self.make_service(repository=repository)

        results = service.search(
            "  enterprise   retrieval\nsearch ",
            "owner-id",
            document_ids=["document-id"],
        )

        self.assertEqual(provider.queries, ["enterprise retrieval search"])
        self.assertEqual(results, [make_result()])
        self.assertEqual(
            repository.calls,
            [
                {
                    "query_embedding": [0.1, 0.2, 0.3],
                    "owner_id": "owner-id",
                    "top_k": 5,
                    "minimum_score": 0.30,
                    "document_ids": ["document-id"],
                }
            ],
        )

    def test_query_vector_dimension_mismatch_is_rejected(self) -> None:
        provider = FakeEmbeddingProvider(vector=[0.1, 0.2], dimensions=3)
        service, _, repository = self.make_service(provider=provider)

        with self.assertRaisesRegex(EmbeddingError, "dimension"):
            service.search("valid query", "owner-id")

        self.assertEqual(repository.calls, [])

    def test_repository_errors_are_sanitized(self) -> None:
        repository = FakeVectorRepository()
        repository.similarity_search = Mock(
            side_effect=RuntimeError("postgresql://user:secret@internal")
        )
        service, _, _ = self.make_service(repository=repository)

        with patch("app.business.services.retrieval_service.logger.exception"):
            with self.assertRaisesRegex(
                RetrievalError,
                "Retrieval service is unavailable",
            ):
                service.search("valid query", "owner-id")


class PgVectorRepositoryStatementTests(unittest.TestCase):
    def execute_search(
        self,
        owner_id: str = "owner-a",
        document_ids: list[str] | None = None,
        minimum_score: float = 0.30,
        rows: list[dict] | None = None,
    ):
        session = Mock()
        session.execute.return_value.mappings.return_value.all.return_value = (
            rows or []
        )
        repository = PgVectorRepository(session)
        results = repository.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            owner_id=owner_id,
            top_k=5,
            minimum_score=minimum_score,
            document_ids=document_ids,
        )
        statement = session.execute.call_args.args[0]
        compiled = statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"render_postcompile": True},
        )
        return str(compiled), compiled.params, results

    def test_search_is_scoped_to_current_owner(self) -> None:
        sql, params, _ = self.execute_search(owner_id="owner-a")

        self.assertIn("documents.owner_id =", sql)
        self.assertEqual(params["owner_id_1"], "owner-a")

    def test_user_b_query_cannot_use_user_a_scope(self) -> None:
        _, params, _ = self.execute_search(owner_id="owner-b")

        self.assertEqual(params["owner_id_1"], "owner-b")
        self.assertNotIn("owner-a", params.values())

    def test_optional_document_id_filter_is_applied(self) -> None:
        sql, params, _ = self.execute_search(
            document_ids=["document-a", "document-b"]
        )

        self.assertIn("document_chunks.document_id IN", sql)
        self.assertEqual(params["document_id_1_1"], "document-a")
        self.assertEqual(params["document_id_1_2"], "document-b")

    def test_similarity_threshold_filter_is_applied(self) -> None:
        sql, params, _ = self.execute_search(minimum_score=0.72)

        self.assertIn("embedding <=>", sql)
        self.assertIn(") >=", sql)
        self.assertEqual(params["param_2"], 0.72)

    def test_only_embedded_chunks_and_completed_documents_are_searched(self) -> None:
        sql, params, _ = self.execute_search()

        self.assertIn("document_chunks.embedding IS NOT NULL", sql)
        self.assertIn("documents.status =", sql)
        self.assertEqual(params["status_1"], "completed")

    def test_results_are_ordered_by_smallest_cosine_distance(self) -> None:
        sql, _, results = self.execute_search(
            rows=[
                {
                    "chunk_id": "high",
                    "document_id": "document-id",
                    "document_name": "report.pdf",
                    "chunk_index": 0,
                    "content": "Most relevant",
                    "page_number": 1,
                    "similarity_score": 0.95,
                },
                {
                    "chunk_id": "lower",
                    "document_id": "document-id",
                    "document_name": "report.pdf",
                    "chunk_index": 1,
                    "content": "Less relevant",
                    "page_number": 2,
                    "similarity_score": 0.75,
                },
            ]
        )

        self.assertIn("ORDER BY (document_chunks.embedding <=>", sql)
        self.assertIn(" ASC", sql)
        self.assertEqual(
            [result.chunk_id for result in results],
            ["high", "lower"],
        )
        self.assertFalse(hasattr(results[0], "embedding"))


class RetrievalEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_response_does_not_expose_embedding_vectors(self) -> None:
        retrieval_service = Mock()
        retrieval_service.search.return_value = [make_result()]
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

        response = TestClient(app).post(
            "/api/v1/retrieval/search",
            json={"query": "enterprise search", "top_k": 5},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_results"], 1)
        self.assertNotIn("embedding", body["results"][0])
        self.assertEqual(body["results"][0]["similarity_score"], 0.91)

    def test_expected_search_validation_error_returns_400(self) -> None:
        retrieval_service = Mock()
        retrieval_service.search.side_effect = ValidationError(
            "Search query must not be empty"
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

        response = TestClient(app).post(
            "/api/v1/retrieval/search",
            json={"query": "   "},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "Search query must not be empty"},
        )


if __name__ == "__main__":
    unittest.main()

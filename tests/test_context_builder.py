import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.services.context_builder_service import ContextBuilderService
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_context_builder_service,
    get_retrieval_service,
)


def make_result(
    chunk_id: str,
    document_name: str,
    page_number: int | None,
    chunk_index: int,
    content: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        document_name=document_name,
        page_number=page_number,
        chunk_index=chunk_index,
        content=content,
        similarity_score=score,
    )


class ContextBuilderServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ContextBuilderService()

    def test_sources_are_numbered_and_metadata_is_preserved(self) -> None:
        results = [
            make_result("one", "Candidate CV.pdf", 1, 0, "AI tools", 0.91),
            make_result("two", "Projects.pdf", 3, 4, "RAG systems", 0.82),
        ]

        _, sources = self.service.build_context(results)

        self.assertEqual([source.source_number for source in sources], [1, 2])
        self.assertEqual(
            [source.document_name for source in sources],
            ["Candidate CV.pdf", "Projects.pdf"],
        )
        self.assertEqual([source.page_number for source in sources], [1, 3])
        self.assertEqual([source.content for source in sources], ["AI tools", "RAG systems"])

    def test_context_formatting_is_grounded_in_retrieval_results(self) -> None:
        result = make_result(
            "one",
            "Candidate CV.pdf",
            None,
            2,
            "TensorFlow and PyTorch",
            0.91,
        )

        context, _ = self.service.build_context([result])

        self.assertEqual(
            context,
            "[SOURCE 1]\n"
            "Document: Candidate CV.pdf\n"
            "Page: N/A\n"
            "Chunk: 2\n"
            "Content:\nTensorFlow and PyTorch",
        )
        self.assertNotIn("answer", context.lower())

    def test_empty_results_return_empty_context_and_sources(self) -> None:
        self.assertEqual(self.service.build_context([]), ("", []))


class ContextBuildEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_endpoint_uses_owner_scoped_retrieval_without_fabricating_answer(self) -> None:
        result = make_result(
            "one",
            "Candidate CV.pdf",
            1,
            0,
            "TensorFlow and PyTorch",
            0.91,
        )
        retrieval_service = Mock()
        retrieval_service.search.return_value = [result]
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
        app.dependency_overrides[get_context_builder_service] = (
            ContextBuilderService
        )

        response = TestClient(app).post(
            "/api/v1/context/build",
            json={
                "query": "What tools are mentioned?",
                "top_k": 5,
                "document_ids": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["llm_status"], "not_configured")
        self.assertNotIn("answer", body)
        self.assertIn("TensorFlow and PyTorch", body["context"])
        self.assertNotIn("content", body["sources"][0])
        retrieval_service.search.assert_called_once_with(
            query="What tools are mentioned?",
            owner_id="owner-id",
            top_k=5,
            document_ids=None,
        )


if __name__ == "__main__":
    unittest.main()

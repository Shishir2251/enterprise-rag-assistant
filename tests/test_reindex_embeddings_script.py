import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.exceptions import ValidationError
from app.data_access.models.document_model import DocumentStatus
from app.scripts.reindex_embeddings import (
    SAFE_FAILURE_MESSAGE,
    BulkReindexDependencies,
    _configured_dependencies,
    main,
    reindex_stale_documents,
)


class FakeDocumentRepository:
    def __init__(self, documents: dict[str, SimpleNamespace]) -> None:
        self.documents = documents
        self.processing_calls: list[str] = []
        self.progress_calls: list[tuple[str, int, str]] = []
        self.ready_calls: list[str] = []
        self.failed_calls: list[tuple[str, str]] = []

    def get_by_id_internal(self, document_id: str):
        return self.documents.get(document_id)

    def mark_processing(self, document_id: str):
        document = self.documents[document_id]
        document.status = DocumentStatus.PROCESSING.value
        self.processing_calls.append(document_id)
        return document

    def update_progress(
        self,
        document_id: str,
        progress: int,
        current_step: str,
    ):
        document = self.documents[document_id]
        document.progress = progress
        document.current_step = current_step
        self.progress_calls.append(
            (document_id, progress, current_step)
        )
        return document

    def mark_ready(self, document_id: str):
        document = self.documents[document_id]
        document.status = DocumentStatus.READY.value
        self.ready_calls.append(document_id)
        return document

    def mark_failed(self, document_id: str, error_message: str):
        document = self.documents[document_id]
        document.status = DocumentStatus.FAILED.value
        document.error_message = error_message
        self.failed_calls.append((document_id, error_message))
        return document


class FakeChunkRepository:
    def __init__(self, stale_document_ids: list[str]) -> None:
        self.stale_document_ids = stale_document_ids
        self.remaining_stale = {
            document_id: [object()] for document_id in stale_document_ids
        }
        self.discovery_calls: list[tuple[str, str]] = []

    def list_stale_document_ids(
        self,
        model_name: str,
        provider_name: str,
    ) -> list[str]:
        self.discovery_calls.append((model_name, provider_name))
        return list(self.stale_document_ids)

    def list_stale_embeddings(
        self,
        document_id: str,
        model_name: str,
        provider_name: str,
    ) -> list[object]:
        return list(self.remaining_stale.get(document_id, []))


class FakeEmbeddingService:
    def __init__(
        self,
        chunk_repository: FakeChunkRepository,
        *,
        failures: set[str] | None = None,
        embedded_counts: dict[str, int] | None = None,
        leave_stale: set[str] | None = None,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.failures = failures or set()
        self.embedded_counts = embedded_counts or {}
        self.leave_stale = leave_stale or set()
        self.calls: list[tuple[str, str]] = []

    def reindex_document_chunks(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        self.calls.append((document_id, owner_id))
        if document_id in self.failures:
            raise RuntimeError(
                "secret-key at C:/private/source-document.txt"
            )
        if document_id not in self.leave_stale:
            self.chunk_repository.remaining_stale[document_id] = []
        return self.embedded_counts.get(document_id, 1)


def make_document(document_id: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=document_id,
        owner_id=f"owner-{document_id}",
        status=status,
        progress=100,
        current_step="completed",
        error_message=None,
    )


def make_dependencies(
    document_repository: FakeDocumentRepository,
    chunk_repository: FakeChunkRepository,
    embedding_service: FakeEmbeddingService,
    rollback=None,
) -> BulkReindexDependencies:
    return BulkReindexDependencies(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        embedding_service=embedding_service,
        embedding_provider=SimpleNamespace(
            model_name="active-model",
            provider_name="local",
        ),
        rollback=rollback or (lambda: None),
    )


class BulkReindexTests(unittest.TestCase):
    def test_only_ready_or_completed_stale_documents_are_reindexed(self) -> None:
        document_repository = FakeDocumentRepository(
            {
                "ready": make_document("ready", DocumentStatus.READY.value),
                "completed": make_document(
                    "completed",
                    DocumentStatus.COMPLETED.value,
                ),
                "processing": make_document(
                    "processing",
                    DocumentStatus.PROCESSING.value,
                ),
            }
        )
        chunk_repository = FakeChunkRepository(
            ["ready", "completed", "processing", "missing", "ready"]
        )
        embedding_service = FakeEmbeddingService(
            chunk_repository,
            embedded_counts={"ready": 2, "completed": 3},
        )

        summary = reindex_stale_documents(
            make_dependencies(
                document_repository,
                chunk_repository,
                embedding_service,
            )
        )

        self.assertEqual(summary.stale_documents_found, 4)
        self.assertEqual(summary.documents_attempted, 2)
        self.assertEqual(summary.documents_succeeded, 2)
        self.assertEqual(summary.documents_failed, 0)
        self.assertEqual(summary.documents_skipped, 2)
        self.assertEqual(summary.chunks_embedded, 5)
        self.assertEqual(
            embedding_service.calls,
            [
                ("ready", "owner-ready"),
                ("completed", "owner-completed"),
            ],
        )
        self.assertEqual(
            chunk_repository.discovery_calls,
            [("active-model", "local")],
        )
        self.assertEqual(
            document_repository.processing_calls,
            ["ready", "completed"],
        )
        self.assertEqual(
            document_repository.ready_calls,
            ["ready", "completed"],
        )

    def test_limit_counts_eligible_documents_not_skipped_rows(self) -> None:
        document_repository = FakeDocumentRepository(
            {
                "processing": make_document(
                    "processing",
                    DocumentStatus.PROCESSING.value,
                ),
                "first": make_document("first", DocumentStatus.READY.value),
                "second": make_document(
                    "second",
                    DocumentStatus.READY.value,
                ),
            }
        )
        chunk_repository = FakeChunkRepository(
            ["processing", "first", "second"]
        )
        embedding_service = FakeEmbeddingService(chunk_repository)

        summary = reindex_stale_documents(
            make_dependencies(
                document_repository,
                chunk_repository,
                embedding_service,
            ),
            limit=1,
        )

        self.assertEqual(summary.documents_attempted, 1)
        self.assertEqual(summary.documents_skipped, 1)
        self.assertEqual(embedding_service.calls, [("first", "owner-first")])

    def test_failure_is_safely_marked_and_next_document_continues(self) -> None:
        document_repository = FakeDocumentRepository(
            {
                "bad": make_document("bad", DocumentStatus.READY.value),
                "good": make_document("good", DocumentStatus.READY.value),
            }
        )
        chunk_repository = FakeChunkRepository(["bad", "good"])
        embedding_service = FakeEmbeddingService(
            chunk_repository,
            failures={"bad"},
            embedded_counts={"good": 4},
        )
        rollback = Mock()

        with patch("app.scripts.reindex_embeddings.logger.error"):
            summary = reindex_stale_documents(
                make_dependencies(
                    document_repository,
                    chunk_repository,
                    embedding_service,
                    rollback,
                )
            )

        self.assertEqual(summary.documents_attempted, 2)
        self.assertEqual(summary.documents_failed, 1)
        self.assertEqual(summary.documents_succeeded, 1)
        self.assertEqual(summary.chunks_embedded, 4)
        self.assertEqual(
            embedding_service.calls,
            [("bad", "owner-bad"), ("good", "owner-good")],
        )
        self.assertEqual(
            document_repository.failed_calls,
            [("bad", SAFE_FAILURE_MESSAGE)],
        )
        self.assertEqual(
            document_repository.documents["bad"].status,
            DocumentStatus.FAILED.value,
        )
        self.assertEqual(
            document_repository.documents["good"].status,
            DocumentStatus.READY.value,
        )
        rollback.assert_called_once_with()

    def test_document_is_not_ready_when_metadata_remains_stale(self) -> None:
        document_repository = FakeDocumentRepository(
            {"stale": make_document("stale", DocumentStatus.READY.value)}
        )
        chunk_repository = FakeChunkRepository(["stale"])
        embedding_service = FakeEmbeddingService(
            chunk_repository,
            leave_stale={"stale"},
        )

        with patch("app.scripts.reindex_embeddings.logger.error"):
            summary = reindex_stale_documents(
                make_dependencies(
                    document_repository,
                    chunk_repository,
                    embedding_service,
                )
            )

        self.assertEqual(summary.documents_failed, 1)
        self.assertEqual(document_repository.ready_calls, [])
        self.assertEqual(
            document_repository.failed_calls,
            [("stale", SAFE_FAILURE_MESSAGE)],
        )

    def test_main_reports_only_aggregate_data_after_partial_failure(self) -> None:
        document_repository = FakeDocumentRepository(
            {"bad": make_document("bad", DocumentStatus.READY.value)}
        )
        chunk_repository = FakeChunkRepository(["bad"])
        embedding_service = FakeEmbeddingService(
            chunk_repository,
            failures={"bad"},
        )
        output = io.StringIO()
        errors = io.StringIO()

        with (
            patch("app.scripts.reindex_embeddings.logger.error"),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            exit_code = main(
                [],
                dependencies=make_dependencies(
                    document_repository,
                    chunk_repository,
                    embedding_service,
                ),
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["documents_failed"], 1)
        rendered_output = output.getvalue() + errors.getvalue()
        self.assertNotIn("secret-key", rendered_output)
        self.assertNotIn("source-document", rendered_output)
        self.assertNotIn("bad", rendered_output)

    def test_invalid_programmatic_limit_is_rejected(self) -> None:
        dependencies = make_dependencies(
            FakeDocumentRepository({}),
            FakeChunkRepository([]),
            FakeEmbeddingService(FakeChunkRepository([])),
        )

        with self.assertRaisesRegex(ValidationError, "greater than zero"):
            reindex_stale_documents(dependencies, limit=0)

    def test_configured_dependencies_constructs_provider_once_per_run(self) -> None:
        provider = SimpleNamespace(
            provider_name="local",
            model_name="active-model",
            dimensions=384,
        )
        session = Mock()
        create_provider = Mock(return_value=provider)

        with (
            patch(
                "app.infrastructure.database.session.SessionLocal",
                return_value=session,
            ),
            patch(
                "app.infrastructure.embeddings.embedding_provider_factory."
                "create_embedding_provider",
                create_provider,
            ),
        ):
            with _configured_dependencies(7) as dependencies:
                self.assertIs(dependencies.embedding_provider, provider)
                self.assertIs(
                    dependencies.embedding_service.embedding_provider,
                    provider,
                )
                self.assertEqual(dependencies.embedding_service.batch_size, 7)

        create_provider.assert_called_once()
        session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

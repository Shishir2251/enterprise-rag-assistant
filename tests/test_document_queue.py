import io
import unittest
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from celery.exceptions import Retry
from fastapi import UploadFile
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from starlette.datastructures import Headers

from app.business.services.document_service import DocumentService
from app.core.config import settings
from app.core.exceptions import (
    ConflictError,
    NotFoundError,
    QueueUnavailableError,
    ValidationError,
)
from app.data_access.models.document_model import DocumentModel, DocumentStatus
from app.data_access.repositories.document_repository import DocumentRepository
from app.infrastructure.queue.celery_document_processing_queue import (
    CeleryDocumentProcessingQueue,
)
from app.infrastructure.queue.celery_app import celery_app
from app.infrastructure.queue.task_dependencies import (
    DocumentTaskDependencies,
)
from app.infrastructure.queue.tasks.document_tasks import (
    process_document_task,
)
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_document_service,
)


NOW = datetime(2026, 7, 19, 10, 0, 0)


def make_upload() -> UploadFile:
    return UploadFile(
        filename="report.pdf",
        file=io.BytesIO(b"%PDF-test"),
        headers=Headers({"content-type": "application/pdf"}),
    )


def make_document(
    status: str = DocumentStatus.UPLOADED.value,
) -> DocumentModel:
    return DocumentModel(
        id="document-id",
        owner_id="owner-id",
        original_name="report.pdf",
        stored_name="stored.pdf",
        file_path="uploads/owner-id/stored.pdf",
        mime_type="application/pdf",
        file_size=9,
        status=status,
        progress=0,
        retry_count=0,
        created_at=NOW,
        updated_at=NOW,
    )


class QueueAdapterTests(unittest.TestCase):
    def test_queue_adapter_returns_celery_task_id(self) -> None:
        with patch(
            "app.infrastructure.queue.celery_document_processing_queue."
            "process_document_task.apply_async",
            return_value=SimpleNamespace(id="celery-task-id"),
        ) as apply_async:
            task_id = CeleryDocumentProcessingQueue().enqueue("document-id")

        self.assertEqual(task_id, "celery-task-id")
        apply_async.assert_called_once_with(args=["document-id"])


class DocumentRepositoryScopeTests(unittest.TestCase):
    def test_public_lookup_is_owner_scoped_but_worker_lookup_is_internal(
        self,
    ) -> None:
        db = Mock()
        repository = DocumentRepository(db)

        repository.get_by_id("document-id", "owner-id")
        public_statement = db.scalar.call_args.args[0]
        public_sql = str(
            public_statement.compile(dialect=postgresql.dialect())
        )

        repository.get_by_id_internal("document-id")
        internal_statement = db.scalar.call_args.args[0]
        internal_sql = str(
            internal_statement.compile(dialect=postgresql.dialect())
        )

        self.assertIn("documents.owner_id =", public_sql)
        self.assertNotIn("documents.owner_id =", internal_sql)


class DocumentAsyncServiceTests(unittest.TestCase):
    def make_service(self, queue=None):
        document = make_document()
        repository = Mock()

        def create(created_document):
            created_document.id = "document-id"
            created_document.progress = 0
            created_document.retry_count = 0
            created_document.created_at = NOW
            created_document.updated_at = NOW
            return created_document

        def mark_queued(document_id, task_id=None):
            self.assertEqual(document_id, "document-id")
            document.status = DocumentStatus.QUEUED.value
            document.progress = 5
            document.current_step = "queued"
            document.error_message = None
            document.task_id = task_id
            return document

        def set_task_id(document_id, task_id):
            self.assertEqual(document_id, "document-id")
            document.task_id = task_id
            return document

        repository.create.side_effect = create
        repository.mark_queued.side_effect = mark_queued
        repository.set_task_id.side_effect = set_task_id
        repository.get_by_id.return_value = document
        repository.mark_failed.return_value = document

        storage = Mock()
        storage.save.return_value = (
            "stored.pdf",
            "uploads/owner-id/stored.pdf",
        )
        processing_queue = queue or Mock()
        processing_queue.enqueue.return_value = "task-id"
        service = DocumentService(
            document_repository=repository,
            file_storage=storage,
            processing_queue=processing_queue,
        )
        return service, document, repository, storage, processing_queue

    def test_upload_enqueues_and_returns_queued_without_ingestion(self) -> None:
        service, _, repository, _, queue = self.make_service()

        result = service.upload(make_upload(), "owner-id")

        self.assertEqual(result.status, DocumentStatus.QUEUED.value)
        self.assertEqual(result.task_id, "task-id")
        queue.enqueue.assert_called_once_with("document-id")
        repository.mark_queued.assert_called_once_with("document-id")
        repository.set_task_id.assert_called_once_with(
            "document-id",
            "task-id",
        )

    def test_queue_failure_marks_record_failed_and_preserves_file(self) -> None:
        queue = Mock()
        queue.enqueue.side_effect = ConnectionError(
            "redis://user:secret@internal"
        )
        service, _, repository, storage, _ = self.make_service(queue)

        with patch(
            "app.business.services.document_service.logger.exception"
        ):
            with self.assertRaises(QueueUnavailableError):
                service.upload(make_upload(), "owner-id")

        repository.mark_failed.assert_called_once_with(
            "document-id",
            "Document processing could not be queued",
        )
        storage.delete.assert_not_called()

    def test_retry_only_allows_failed_documents(self) -> None:
        service, document, repository, _, queue = self.make_service()
        document.status = DocumentStatus.FAILED.value

        result = service.retry_document("document-id", "owner-id")

        self.assertEqual(result.status, DocumentStatus.QUEUED.value)
        self.assertEqual(result.task_id, "task-id")
        queue.enqueue.assert_called_once_with("document-id")
        repository.mark_queued.assert_called_once_with("document-id")

    def test_retry_is_blocked_for_ready_document(self) -> None:
        service, document, _, _, queue = self.make_service()
        document.status = DocumentStatus.READY.value

        with self.assertRaises(ConflictError):
            service.retry_document("document-id", "owner-id")

        queue.enqueue.assert_not_called()


class FakeTaskDocumentRepository:
    def __init__(self, document) -> None:
        self.document = document
        self.calls: list[tuple] = []

    def get_by_id_internal(self, document_id: str):
        self.calls.append(("get_internal", document_id))
        return self.document

    def set_task_id(self, document_id: str, task_id: str):
        self.document.task_id = task_id
        self.calls.append(("set_task_id", document_id, task_id))
        return self.document

    def mark_processing(self, document_id: str):
        self.document.status = DocumentStatus.PROCESSING.value
        self.calls.append(("mark_processing", document_id))
        return self.document

    def update_progress(
        self,
        document_id: str,
        progress: int,
        current_step: str,
    ):
        self.document.progress = progress
        self.document.current_step = current_step
        self.calls.append(
            ("progress", document_id, progress, current_step)
        )
        return self.document

    def mark_ready(self, document_id: str):
        self.document.status = DocumentStatus.READY.value
        self.document.progress = 100
        self.calls.append(("mark_ready", document_id))
        return self.document

    def increment_retry_count(self, document_id: str):
        self.document.retry_count += 1
        self.calls.append(("increment_retry", document_id))
        return self.document

    def mark_failed(self, document_id: str, error_message: str):
        self.document.status = DocumentStatus.FAILED.value
        self.document.error_message = error_message
        self.calls.append(("mark_failed", document_id, error_message))
        return self.document

    def mark_retry_scheduled(
        self,
        document_id: str,
        error_message: str,
        task_id: str | None,
    ):
        self.document.status = DocumentStatus.QUEUED.value
        self.document.error_message = error_message
        self.calls.append(
            ("retry_scheduled", document_id, error_message, task_id)
        )
        return self.document


class DocumentTaskTests(unittest.TestCase):
    def make_dependencies(
        self,
        status: str = DocumentStatus.QUEUED.value,
        ingestion_error: Exception | None = None,
    ):
        document = SimpleNamespace(
            id="document-id",
            owner_id="owner-id",
            status=status,
            task_id=None,
            progress=5,
            retry_count=0,
        )
        repository = FakeTaskDocumentRepository(document)
        chunk_repository = Mock()
        chunk_repository.list_by_document.return_value = [
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ]
        ingestion_service = Mock()

        if ingestion_error is None:
            def ingest_document(**kwargs):
                callback = kwargs["progress_callback"]
                callback(45, "chunking")
                callback(60, "saving_chunks")
                return 2

            ingestion_service.ingest_document.side_effect = ingest_document
        else:
            ingestion_service.ingest_document.side_effect = ingestion_error

        embedding_service = Mock()
        embedding_service.embed_document_chunks.return_value = 2
        dependencies = DocumentTaskDependencies(
            session=Mock(),
            document_repository=repository,
            chunk_repository=chunk_repository,
            ingestion_service=ingestion_service,
            embedding_service=embedding_service,
        )
        return (
            dependencies,
            document,
            repository,
            chunk_repository,
            ingestion_service,
            embedding_service,
        )

    @staticmethod
    def dependency_context(dependencies):
        @contextmanager
        def context():
            yield dependencies

        return context

    def run_task(self, dependencies, retries: int = 0):
        context = self.dependency_context(dependencies)
        with patch(
            "app.infrastructure.queue.tasks.document_tasks."
            "get_document_task_dependencies",
            context,
        ):
            process_document_task.push_request(
                id="task-id",
                retries=retries,
            )
            try:
                return process_document_task.run("document-id")
            finally:
                process_document_task.pop_request()

    def test_task_runs_ingestion_embedding_progress_and_marks_ready(
        self,
    ) -> None:
        (
            dependencies,
            document,
            repository,
            _,
            ingestion_service,
            embedding_service,
        ) = self.make_dependencies()

        result = self.run_task(dependencies)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["chunks_processed"], 2)
        self.assertEqual(result["chunks_embedded"], 2)
        self.assertEqual(document.status, DocumentStatus.READY.value)
        self.assertIn(("mark_processing", "document-id"), repository.calls)
        self.assertIn(
            ("progress", "document-id", 20, "extracting_text"),
            repository.calls,
        )
        self.assertIn(
            ("progress", "document-id", 45, "chunking"),
            repository.calls,
        )
        self.assertIn(
            ("progress", "document-id", 75, "generating_embeddings"),
            repository.calls,
        )
        self.assertIn(("mark_ready", "document-id"), repository.calls)
        ingestion_service.ingest_document.assert_called_once()
        embedding_service.embed_document_chunks.assert_called_once_with(
            document_id="document-id",
            owner_id="owner-id",
        )

    def test_permanent_validation_failure_does_not_retry(self) -> None:
        dependencies, document, repository, _, _, _ = (
            self.make_dependencies(
                ingestion_error=ValidationError(
                    "No readable text was extracted from the document"
                )
            )
        )

        with patch.object(process_document_task, "retry") as retry:
            result = self.run_task(dependencies)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(document.status, DocumentStatus.FAILED.value)
        self.assertEqual(
            document.error_message,
            "No readable text was extracted from the document",
        )
        self.assertIn(("increment_retry", "document-id"), repository.calls)
        retry.assert_not_called()

    def test_transient_failure_schedules_retry_without_marking_failed(
        self,
    ) -> None:
        dependencies, document, repository, _, _, _ = (
            self.make_dependencies(
                ingestion_error=RuntimeError(
                    "postgresql://user:secret@internal"
                )
            )
        )
        context = self.dependency_context(dependencies)

        with patch(
            "app.infrastructure.queue.tasks.document_tasks."
            "get_document_task_dependencies",
            context,
        ):
            process_document_task.push_request(
                id="task-id",
                retries=0,
            )
            try:
                with patch.object(
                    process_document_task,
                    "retry",
                    side_effect=Retry(),
                ) as retry:
                    with self.assertRaises(Retry):
                        process_document_task.run("document-id")
            finally:
                process_document_task.pop_request()

        self.assertEqual(document.status, DocumentStatus.QUEUED.value)
        self.assertEqual(document.error_message, "Document processing failed")
        self.assertFalse(
            any(call[0] == "mark_failed" for call in repository.calls)
        )
        retry.assert_called_once()

    def test_exhausted_transient_failure_marks_failed(self) -> None:
        dependencies, document, repository, _, _, _ = (
            self.make_dependencies(
                ingestion_error=RuntimeError("temporary database outage")
            )
        )

        result = self.run_task(
            dependencies,
            retries=settings.DOCUMENT_PROCESSING_MAX_RETRIES,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(document.status, DocumentStatus.FAILED.value)
        self.assertEqual(document.error_message, "Document processing failed")
        self.assertTrue(
            any(call[0] == "mark_failed" for call in repository.calls)
        )

    def test_ready_document_is_idempotent(self) -> None:
        (
            dependencies,
            _,
            repository,
            _,
            ingestion_service,
            embedding_service,
        ) = self.make_dependencies(status=DocumentStatus.READY.value)

        result = self.run_task(dependencies)

        self.assertEqual(result["status"], "ready")
        ingestion_service.ingest_document.assert_not_called()
        embedding_service.embed_document_chunks.assert_not_called()
        self.assertFalse(
            any(call[0] == "mark_processing" for call in repository.calls)
        )

    def test_celery_eager_mode_runs_without_live_redis(self) -> None:
        dependencies, _, _, _, _, _ = self.make_dependencies()
        context = self.dependency_context(dependencies)
        previous_eager = celery_app.conf.task_always_eager
        celery_app.conf.task_always_eager = True
        try:
            with patch(
                "app.infrastructure.queue.tasks.document_tasks."
                "get_document_task_dependencies",
                context,
            ):
                result = process_document_task.apply_async(
                    args=["document-id"]
                ).get(propagate=True)
        finally:
            celery_app.conf.task_always_eager = previous_eager

        self.assertEqual(result["status"], "ready")


class DocumentStatusEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def status_document(self, status: str = "processing"):
        return SimpleNamespace(
            id="document-id",
            status=status,
            progress=75,
            current_step="generating_embeddings",
            error_message=None,
            retry_count=0,
            task_id="task-id",
            processing_started_at=NOW,
            processing_completed_at=None,
            created_at=NOW,
            updated_at=NOW,
            file_path="uploads/private.pdf",
            stored_name="private.pdf",
        )

    def test_status_response_is_owner_scoped_and_hides_internal_fields(
        self,
    ) -> None:
        service = Mock()
        service.get_document_status.return_value = self.status_document()
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_document_service] = lambda: service

        response = TestClient(app).get(
            "/api/v1/documents/document-id/status"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["queued_task_id"], "task-id")
        self.assertNotIn("task_id", body)
        self.assertNotIn("file_path", body)
        self.assertNotIn("stored_name", body)
        service.get_document_status.assert_called_once_with(
            document_id="document-id",
            owner_id="owner-id",
        )

    def test_another_user_cannot_inspect_status(self) -> None:
        service = Mock()
        service.get_document_status.side_effect = NotFoundError(
            "Document not found"
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="other-owner"
        )
        app.dependency_overrides[get_document_service] = lambda: service

        response = TestClient(app).get(
            "/api/v1/documents/document-id/status"
        )

        self.assertEqual(response.status_code, 404)

    def test_retry_endpoint_returns_202_for_failed_document(self) -> None:
        service = Mock()
        service.retry_document.return_value = SimpleNamespace(
            id="document-id",
            status="queued",
            task_id="new-task-id",
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_document_service] = lambda: service

        response = TestClient(app).post(
            "/api/v1/documents/document-id/retry"
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "document_id": "document-id",
                "status": "queued",
                "queued_task_id": "new-task-id",
            },
        )

    def test_retry_endpoint_rejects_ready_document(self) -> None:
        service = Mock()
        service.retry_document.side_effect = ConflictError(
            "Only failed documents can be queued for retry"
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_document_service] = lambda: service

        response = TestClient(app).post(
            "/api/v1/documents/document-id/retry"
        )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()

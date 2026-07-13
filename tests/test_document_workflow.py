import io
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.business.dtos.extracted_text_dto import (
    ExtractedDocument,
    ExtractedPage,
)
from app.business.services.chunking_service import ChunkingService
from app.business.services.document_service import DocumentService
from app.business.services.ingestion_service import IngestionService
from app.core.config import settings
from app.core.exceptions import (
    DocumentProcessingError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from app.data_access.models.document_model import DocumentModel, DocumentStatus
from app.infrastructure.file_storage.local_storage_provider import (
    LocalStorageProvider,
)
from app.presentation.schemas.document_schema import DocumentResponse


def make_upload(
    filename: str,
    content: bytes,
    content_type: str,
) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


class FakeDocumentRepository:
    def __init__(self, document: DocumentModel | None = None) -> None:
        self.document = document
        self.statuses: list[tuple[str, str | None]] = []

    def create(self, document: DocumentModel) -> DocumentModel:
        self.document = document
        return document

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

    def list_by_owner(self, owner_id: str) -> list[DocumentModel]:
        if self.document is not None and self.document.owner_id == owner_id:
            return [self.document]
        return []

    def delete(self, document: DocumentModel) -> None:
        self.document = None

    def update_status(
        self,
        document: DocumentModel,
        document_status: str,
        error_message: str | None = None,
    ) -> DocumentModel:
        document.status = document_status
        document.error_message = error_message
        self.statuses.append((document_status, error_message))
        return document


class FakeStorage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def save(
        self,
        file: UploadFile,
        owner_id: str,
        extension: str,
    ) -> tuple[str, str]:
        return "generated.pdf", f"uploads/{owner_id}/generated.pdf"

    def delete(self, file_path: str) -> None:
        self.deleted.append(file_path)


class FakeChunkRepository:
    def __init__(self) -> None:
        self.chunks = []

    def create_many(self, chunks):
        self.chunks.extend(chunks)
        return chunks

    def list_by_document(self, document_id: str):
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    def delete_by_document(self, document_id: str) -> None:
        self.chunks = [
            chunk for chunk in self.chunks if chunk.document_id != document_id
        ]

    def replace_by_document(self, document_id: str, chunks):
        self.delete_by_document(document_id)
        self.chunks.extend(chunks)
        return chunks


class FakeExtractor:
    def extract(self, file_path: Path) -> ExtractedDocument:
        return ExtractedDocument(
            pages=[ExtractedPage(page_number=1, content="abcdefghij")]
        )


class FakeExtractorFactory:
    def get_extractor(self, file_path: Path) -> FakeExtractor:
        return FakeExtractor()


class FailingExtractor:
    def extract(self, file_path: Path) -> ExtractedDocument:
        raise RuntimeError("internal database password should stay private")


class FailingExtractorFactory:
    def get_extractor(self, file_path: Path) -> FailingExtractor:
        return FailingExtractor()


class DocumentWorkflowTests(unittest.TestCase):
    def test_public_document_response_hides_internal_storage_fields(self) -> None:
        response = DocumentResponse.model_validate(
            SimpleNamespace(
                id="document-id",
                owner_id="owner-id",
                original_name="report.pdf",
                stored_name="internal.pdf",
                file_path="uploads/internal.pdf",
                mime_type="application/pdf",
                file_size=10,
                status="uploaded",
                error_message="postgresql://user:secret@internal/database",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            )
        ).model_dump()

        self.assertNotIn("owner_id", response)
        self.assertNotIn("stored_name", response)
        self.assertNotIn("file_path", response)
        self.assertEqual(response["error_message"], "Document processing failed")

    def test_upload_rejects_empty_file(self) -> None:
        service = DocumentService(FakeDocumentRepository(), FakeStorage())
        upload = make_upload("empty.pdf", b"", "application/pdf")

        with self.assertRaisesRegex(ValidationError, "must not be empty"):
            service.upload(upload, "owner-id")

    def test_upload_rejects_mime_extension_mismatch(self) -> None:
        service = DocumentService(FakeDocumentRepository(), FakeStorage())
        upload = make_upload("report.pdf", b"%PDF-data", "text/plain")

        with self.assertRaisesRegex(ValidationError, "MIME type"):
            service.upload(upload, "owner-id")

    def test_upload_rejects_file_over_configured_limit(self) -> None:
        service = DocumentService(FakeDocumentRepository(), FakeStorage())
        upload = make_upload("report.pdf", b"%PDF-data", "application/pdf")

        with patch.object(settings, "MAX_UPLOAD_SIZE_BYTES", 5):
            with self.assertRaises(PayloadTooLargeError):
                service.upload(upload, "owner-id")

    def test_upload_sanitizes_client_filename(self) -> None:
        repository = FakeDocumentRepository()
        service = DocumentService(repository, FakeStorage())
        upload = make_upload(
            "../../report.pdf",
            b"%PDF-data",
            "application/pdf",
        )

        document = service.upload(upload, "owner-id")

        self.assertEqual(document.original_name, "report.pdf")

    def test_local_storage_rejects_deletion_outside_upload_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(settings, "UPLOAD_DIR", directory):
                storage = LocalStorageProvider()
                outside = Path(directory).parent / "outside.txt"

                with self.assertRaisesRegex(ValueError, "outside"):
                    storage.delete(str(outside))

    def test_local_storage_uses_uuid_name_and_user_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(settings, "UPLOAD_DIR", directory):
                storage = LocalStorageProvider()
                upload = make_upload(
                    "../../private.pdf",
                    b"%PDF-data",
                    "application/pdf",
                )
                stored_name, file_path = storage.save(
                    upload,
                    "owner-id",
                    ".pdf",
                )

                self.assertEqual(Path(file_path).parent.name, "owner-id")
                self.assertEqual(Path(stored_name).suffix, ".pdf")
                uuid.UUID(Path(stored_name).stem)
                self.assertNotIn("private", stored_name)

                storage.delete(file_path)
                self.assertFalse(Path(file_path).exists())

    def test_character_chunking_preserves_overlap_and_metadata(self) -> None:
        service = ChunkingService(chunk_size=6, chunk_overlap=2)
        chunks = service.create_chunks(
            ExtractedDocument(
                pages=[ExtractedPage(page_number=3, content="abcdefghij")]
            )
        )

        self.assertEqual([chunk.content for chunk in chunks], ["abcdef", "efghij"])
        self.assertEqual([chunk.index for chunk in chunks], [0, 1])
        self.assertEqual([chunk.character_count for chunk in chunks], [6, 6])
        self.assertEqual([chunk.page_number for chunk in chunks], [3, 3])

    def test_processing_checks_ownership_and_completes(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as uploaded_file:
            document = DocumentModel(
                id="document-id",
                owner_id="owner-id",
                original_name="report.txt",
                stored_name="generated.txt",
                file_path=uploaded_file.name,
                mime_type="text/plain",
                file_size=10,
                status=DocumentStatus.UPLOADED.value,
            )
            document_repository = FakeDocumentRepository(document)
            chunk_repository = FakeChunkRepository()
            service = IngestionService(
                document_repository=document_repository,
                chunk_repository=chunk_repository,
                chunking_service=ChunkingService(
                    chunk_size=6,
                    chunk_overlap=2,
                ),
                extractor_factory=FakeExtractorFactory(),
            )

            result = service.process_document("document-id", "owner-id")

        self.assertEqual(result.status, DocumentStatus.COMPLETED.value)
        self.assertEqual(
            [status for status, _ in document_repository.statuses],
            [DocumentStatus.PROCESSING.value, DocumentStatus.COMPLETED.value],
        )
        self.assertEqual(len(chunk_repository.chunks), 2)

        with self.assertRaises(NotFoundError):
            service.list_document_chunks("document-id", "different-owner")

    def test_processing_hides_internal_exception_details(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as uploaded_file:
            document = DocumentModel(
                id="document-id",
                owner_id="owner-id",
                original_name="report.txt",
                stored_name="generated.txt",
                file_path=uploaded_file.name,
                mime_type="text/plain",
                file_size=10,
                status=DocumentStatus.UPLOADED.value,
            )
            repository = FakeDocumentRepository(document)
            service = IngestionService(
                document_repository=repository,
                chunk_repository=FakeChunkRepository(),
                chunking_service=ChunkingService(),
                extractor_factory=FailingExtractorFactory(),
            )

            with patch(
                "app.business.services.ingestion_service.logger.exception"
            ):
                with self.assertRaisesRegex(
                    DocumentProcessingError,
                    "Document processing failed",
                ):
                    service.process_document("document-id", "owner-id")

        self.assertEqual(document.status, DocumentStatus.FAILED.value)
        self.assertEqual(document.error_message, "Document processing failed")


if __name__ == "__main__":
    unittest.main()

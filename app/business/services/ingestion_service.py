import logging
from pathlib import Path

from app.business.interfaces.chunking_service_interface import (
    IChunkingService,
)
from app.business.interfaces.ingestion_service_interface import (
    IIngestionService,
)
from app.business.interfaces.text_extractor_factory_interface import (
    ITextExtractorFactory,
)
from app.core.exceptions import (
    DocumentProcessingError,
    NotFoundError,
    ValidationError,
)
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_chunk_model import DocumentChunkModel
from app.data_access.models.document_model import (
    DocumentModel,
    DocumentStatus,
)

logger = logging.getLogger(__name__)


class IngestionService(IIngestionService):

    def __init__(
        self,
        document_repository: IDocumentRepository,
        chunk_repository: IDocumentChunkRepository,
        chunking_service: IChunkingService,
        extractor_factory: ITextExtractorFactory,
    ):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.chunking_service = chunking_service
        self.extractor_factory = extractor_factory

    def process_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        document = self._get_owned_document(document_id, owner_id)

        self.document_repository.update_status(
            document=document,
            document_status=DocumentStatus.PROCESSING.value,
            error_message=None,
        )

        try:
            file_path = Path(document.file_path)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Uploaded file not found: {file_path}"
                )

            extractor = self.extractor_factory.get_extractor(file_path)

            extracted_document = extractor.extract(file_path)

            if not extracted_document.full_text.strip():
                raise ValidationError(
                    "No readable text was extracted from the document"
                )

            chunks = self.chunking_service.create_chunks(
                extracted_document
            )

            if not chunks:
                raise ValidationError(
                    "No chunks were generated from the document"
                )

            chunk_models = [
                DocumentChunkModel(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    character_count=chunk.character_count,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                )
                for chunk in chunks
            ]

            self.chunk_repository.replace_by_document(
                document_id=document.id,
                chunks=chunk_models,
            )

            return self.document_repository.update_status(
                document=document,
                document_status=DocumentStatus.COMPLETED.value,
                error_message=None,
            )

        except Exception as exc:
            logger.exception(
                "Document processing failed",
                extra={"document_id": document.id},
            )
            safe_message = self._safe_error_message(exc)

            try:
                self.document_repository.update_status(
                    document=document,
                    document_status=DocumentStatus.FAILED.value,
                    error_message=safe_message,
                )
            except Exception:
                logger.exception(
                    "Unable to persist failed document status",
                    extra={"document_id": document.id},
                )

            raise DocumentProcessingError(safe_message) from exc

    def list_document_chunks(
        self,
        document_id: str,
        owner_id: str,
    ) -> list[DocumentChunkModel]:
        self._get_owned_document(document_id, owner_id)
        return self.chunk_repository.list_by_document(document_id)

    def _get_owned_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        document = self.document_repository.get_by_id(
            document_id=document_id,
            owner_id=owner_id,
        )
        if document is None:
            raise NotFoundError("Document not found")
        return document

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        if isinstance(exc, FileNotFoundError):
            return "Uploaded file is unavailable"
        if isinstance(exc, ValidationError):
            return exc.detail[:1000]
        return "Document processing failed"

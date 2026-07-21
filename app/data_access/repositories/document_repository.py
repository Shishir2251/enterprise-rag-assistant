from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentModel, DocumentStatus


class DocumentRepository(IDocumentRepository):

    def __init__(self, db: Session):
        self.db = db

    def create(self, document: DocumentModel) -> DocumentModel:
        try:
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            return document
        except Exception:
            self.db.rollback()
            raise

    def get_by_id(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel | None:
        statement = select(DocumentModel).where(
            DocumentModel.id == document_id,
            DocumentModel.owner_id == owner_id,
        )

        return self.db.scalar(statement)

    def get_by_id_internal(
        self,
        document_id: str,
    ) -> DocumentModel | None:
        statement = select(DocumentModel).where(
            DocumentModel.id == document_id
        )
        return self.db.scalar(statement)

    def list_by_owner(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        statement = (
            select(DocumentModel)
            .where(DocumentModel.owner_id == owner_id)
            .order_by(DocumentModel.created_at.desc())
        )

        return list(self.db.scalars(statement).all())

    def delete(self, document: DocumentModel) -> None:
        try:
            self.db.delete(document)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def update_status(
        self,
        document: DocumentModel,
        document_status: str,
        error_message: str | None = None,
    ) -> DocumentModel:
        document.status = document_status
        document.error_message = error_message
        return self._save(document)

    def mark_queued(
        self,
        document_id: str,
        task_id: str | None = None,
    ) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.status = DocumentStatus.QUEUED.value
        document.progress = 5
        document.current_step = "queued"
        document.error_message = None
        document.task_id = task_id
        document.processing_started_at = None
        document.processing_completed_at = None
        return self._save(document)

    def claim_ready_for_reindex(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel | None:
        statement = (
            update(DocumentModel)
            .where(
                DocumentModel.id == document_id,
                DocumentModel.owner_id == owner_id,
                DocumentModel.status.in_(
                    DocumentStatus.process_complete_values()
                ),
            )
            .values(
                status=DocumentStatus.QUEUED.value,
                progress=5,
                current_step="queued",
                error_message=None,
                task_id=None,
                processing_started_at=None,
                processing_completed_at=None,
            )
            .returning(DocumentModel)
        )
        try:
            document = self.db.scalar(statement)
            self.db.commit()
            if document is not None:
                self.db.refresh(document)
            return document
        except Exception:
            self.db.rollback()
            raise

    def mark_processing(self, document_id: str) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.status = DocumentStatus.PROCESSING.value
        document.current_step = "processing"
        document.error_message = None
        document.processing_started_at = datetime.now(timezone.utc)
        document.processing_completed_at = None
        return self._save(document)

    def update_progress(
        self,
        document_id: str,
        progress: int,
        current_step: str,
    ) -> DocumentModel:
        if not 0 <= progress <= 100:
            raise ValueError("Document progress must be between 0 and 100")
        document = self._get_internal_or_raise(document_id)
        document.progress = progress
        document.current_step = current_step[:100]
        return self._save(document)

    def mark_ready(self, document_id: str) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.status = DocumentStatus.READY.value
        document.progress = 100
        document.current_step = "completed"
        document.error_message = None
        document.processing_completed_at = datetime.now(timezone.utc)
        return self._save(document)

    def mark_failed(
        self,
        document_id: str,
        error_message: str,
    ) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.status = DocumentStatus.FAILED.value
        document.current_step = "failed"
        document.error_message = error_message[:1000]
        return self._save(document)

    def increment_retry_count(self, document_id: str) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.retry_count = (document.retry_count or 0) + 1
        return self._save(document)

    def set_task_id(
        self,
        document_id: str,
        task_id: str,
    ) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.task_id = task_id[:255]
        return self._save(document)

    def mark_retry_scheduled(
        self,
        document_id: str,
        error_message: str,
        task_id: str | None,
    ) -> DocumentModel:
        document = self._get_internal_or_raise(document_id)
        document.status = DocumentStatus.QUEUED.value
        document.current_step = "retry_scheduled"
        document.error_message = error_message[:1000]
        document.task_id = task_id[:255] if task_id else document.task_id
        return self._save(document)

    def _get_internal_or_raise(self, document_id: str) -> DocumentModel:
        document = self.get_by_id_internal(document_id)
        if document is None:
            raise LookupError("Document not found")
        return document

    def _save(self, document: DocumentModel) -> DocumentModel:
        try:
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            return document
        except Exception:
            self.db.rollback()
            raise

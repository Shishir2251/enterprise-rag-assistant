from abc import ABC, abstractmethod

from app.data_access.models.document_model import DocumentModel


class IDocumentRepository(ABC):

    @abstractmethod
    def create(self, document: DocumentModel) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_id_internal(
        self,
        document_id: str,
    ) -> DocumentModel | None:
        raise NotImplementedError

    @abstractmethod
    def list_by_owner(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document: DocumentModel) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_status(
        self,
        document: DocumentModel,
        document_status: str,
        error_message: str | None = None,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def mark_queued(
        self,
        document_id: str,
        task_id: str | None = None,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def mark_processing(self, document_id: str) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def update_progress(
        self,
        document_id: str,
        progress: int,
        current_step: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def mark_ready(self, document_id: str) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(
        self,
        document_id: str,
        error_message: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def increment_retry_count(self, document_id: str) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def set_task_id(
        self,
        document_id: str,
        task_id: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def mark_retry_scheduled(
        self,
        document_id: str,
        error_message: str,
        task_id: str | None,
    ) -> DocumentModel:
        raise NotImplementedError

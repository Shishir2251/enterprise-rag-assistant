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
    def list_by_owner(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document: DocumentModel) -> None:
        raise NotImplementedError
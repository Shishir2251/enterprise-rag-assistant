from abc import ABC, abstractmethod

from fastapi import UploadFile

from app.data_access.models.document_model import DocumentModel


class IDocumentService(ABC):

    @abstractmethod
    def upload(
        self,
        file: UploadFile,
        owner_id: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def list_documents(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        raise NotImplementedError

    @abstractmethod
    def get_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> None:
        raise NotImplementedError
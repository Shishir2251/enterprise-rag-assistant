from abc import ABC, abstractmethod

from app.business.interfaces.uploaded_file_interface import IUploadedFile

from app.data_access.models.document_model import DocumentModel


class IDocumentService(ABC):

    @abstractmethod
    def upload(
        self,
        file: IUploadedFile,
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
    def get_document_status(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def retry_document(
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

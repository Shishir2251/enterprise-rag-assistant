from abc import ABC, abstractmethod

from app.data_access.models.document_chunk_model import DocumentChunkModel


class IDocumentChunkRepository(ABC):

    @abstractmethod
    def create_many(
        self,
        chunks: list[DocumentChunkModel],
    ) -> list[DocumentChunkModel]:
        raise NotImplementedError

    @abstractmethod
    def list_by_document(
        self,
        document_id: str,
    ) -> list[DocumentChunkModel]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_document(
        self,
        document_id: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_by_document(
        self,
        document_id: str,
        chunks: list[DocumentChunkModel],
    ) -> list[DocumentChunkModel]:
        raise NotImplementedError

from abc import ABC, abstractmethod
from datetime import datetime

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

    @abstractmethod
    def list_without_embeddings(
        self,
        document_id: str,
    ) -> list[DocumentChunkModel]:
        raise NotImplementedError

    @abstractmethod
    def list_stale_embeddings(
        self,
        document_id: str,
        model_name: str,
        provider_name: str,
    ) -> list[DocumentChunkModel]:
        """Return chunks missing the active model/provider embedding."""
        raise NotImplementedError

    @abstractmethod
    def list_stale_document_ids(
        self,
        model_name: str,
        provider_name: str,
    ) -> list[str]:
        """Return document IDs with at least one stale embedding."""
        raise NotImplementedError

    @abstractmethod
    def save_embeddings(
        self,
        chunks: list[DocumentChunkModel],
        model_name: str,
        provider_name: str,
        embedded_at: datetime,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_embeddings(
        self,
        document_id: str,
    ) -> int:
        raise NotImplementedError

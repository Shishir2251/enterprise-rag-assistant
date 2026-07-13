from abc import ABC, abstractmethod

from app.data_access.models.document_chunk_model import DocumentChunkModel
from app.data_access.models.document_model import DocumentModel


class IIngestionService(ABC):

    @abstractmethod
    def process_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        raise NotImplementedError

    @abstractmethod
    def list_document_chunks(
        self,
        document_id: str,
        owner_id: str,
    ) -> list[DocumentChunkModel]:
        raise NotImplementedError

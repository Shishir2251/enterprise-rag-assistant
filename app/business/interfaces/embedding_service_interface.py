from abc import ABC, abstractmethod


class IEmbeddingService(ABC):

    @abstractmethod
    def embed_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_document_chunks(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear_document_embeddings(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        raise NotImplementedError

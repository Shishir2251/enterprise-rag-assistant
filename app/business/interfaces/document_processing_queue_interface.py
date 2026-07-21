from abc import ABC, abstractmethod


class IDocumentProcessingQueue(ABC):

    @abstractmethod
    def enqueue(self, document_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def enqueue_reindex(self, document_id: str) -> str:
        raise NotImplementedError

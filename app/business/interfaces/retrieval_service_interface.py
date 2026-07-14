from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.retrieval_result_dto import RetrievalResult


class IRetrievalService(ABC):

    @abstractmethod
    def search(
        self,
        query: str,
        owner_id: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

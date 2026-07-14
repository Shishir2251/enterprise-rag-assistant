from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.retrieval_result_dto import RetrievalResult


class IVectorRepository(ABC):

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        owner_id: str,
        top_k: int,
        minimum_score: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

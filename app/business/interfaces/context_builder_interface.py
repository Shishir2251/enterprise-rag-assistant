from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.retrieval_result_dto import RetrievalResult


class IContextBuilder(ABC):

    @abstractmethod
    def build_context(
        self,
        retrieval_results: Sequence[RetrievalResult],
    ) -> tuple[str, list[ContextSourceDTO]]:
        raise NotImplementedError

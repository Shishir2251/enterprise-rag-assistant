from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.retrieval_evaluation_dto import (
    RetrievalEvaluationCaseDTO,
    RetrievalEvaluationReportDTO,
)


class IRetrievalEvaluationService(ABC):

    @abstractmethod
    def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCaseDTO],
        owner_id: str,
    ) -> RetrievalEvaluationReportDTO:
        raise NotImplementedError

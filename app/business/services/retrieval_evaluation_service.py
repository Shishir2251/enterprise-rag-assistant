import math
from collections.abc import Sequence

from app.business.dtos.retrieval_evaluation_dto import (
    RetrievalEvaluationCaseDTO,
    RetrievalEvaluationCaseResultDTO,
    RetrievalEvaluationReportDTO,
)
from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.interfaces.retrieval_evaluation_service_interface import (
    IRetrievalEvaluationService,
)
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.core.exceptions import RetrievalError, ValidationError


class RetrievalEvaluationService(IRetrievalEvaluationService):
    """Evaluate ranked retrieval against human-labelled chunk positions."""

    EVALUATION_TOP_K = 5

    def __init__(self, retrieval_service: IRetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCaseDTO],
        owner_id: str,
    ) -> RetrievalEvaluationReportDTO:
        normalized_owner_id = (
            owner_id.strip() if isinstance(owner_id, str) else ""
        )
        if not normalized_owner_id:
            raise ValidationError("Evaluation owner_id must not be empty")

        try:
            evaluation_cases = tuple(cases)
        except TypeError as exc:
            raise ValidationError(
                "Retrieval evaluation requires a sequence of cases"
            ) from exc
        if not evaluation_cases:
            raise ValidationError(
                "Retrieval evaluation requires at least one case"
            )

        case_results: list[RetrievalEvaluationCaseResultDTO] = []
        relevant_scores: list[float] = []
        for evaluation_case in evaluation_cases:
            self._validate_case(evaluation_case)
            retrieval_results = self.retrieval_service.search(
                query=evaluation_case.query.strip(),
                owner_id=normalized_owner_id,
                top_k=self.EVALUATION_TOP_K,
                document_ids=None,
            )
            relevant_rank, relevant_score = self._find_relevant_result(
                evaluation_case,
                retrieval_results,
            )
            if relevant_score is not None:
                relevant_scores.append(relevant_score)
            case_results.append(
                RetrievalEvaluationCaseResultDTO(
                    query=evaluation_case.query.strip(),
                    expected_chunk_index=(
                        evaluation_case.expected_chunk_index
                    ),
                    expected_document_id=(
                        evaluation_case.expected_document_id.strip()
                        if evaluation_case.expected_document_id is not None
                        else None
                    ),
                    relevant_rank=relevant_rank,
                    relevant_score=relevant_score,
                )
            )

        total_cases = len(case_results)
        return RetrievalEvaluationReportDTO(
            total_cases=total_cases,
            hit_at_1=self._hit_rate(case_results, cutoff=1),
            hit_at_3=self._hit_rate(case_results, cutoff=3),
            hit_at_5=self._hit_rate(case_results, cutoff=5),
            mrr=(
                sum(
                    1.0 / result.relevant_rank
                    for result in case_results
                    if result.relevant_rank is not None
                )
                / total_cases
            ),
            average_relevant_score=(
                sum(relevant_scores) / len(relevant_scores)
                if relevant_scores
                else None
            ),
            cases=tuple(case_results),
        )

    @staticmethod
    def _validate_case(evaluation_case: RetrievalEvaluationCaseDTO) -> None:
        if not isinstance(evaluation_case, RetrievalEvaluationCaseDTO):
            raise ValidationError(
                "Retrieval evaluation case has an invalid shape"
            )
        normalized_query = (
            evaluation_case.query.strip()
            if isinstance(evaluation_case.query, str)
            else ""
        )
        if not 2 <= len(normalized_query) <= 2000:
            raise ValidationError(
                "Evaluation query must contain between 2 and 2000 characters"
            )
        expected_chunk_index = evaluation_case.expected_chunk_index
        if (
            isinstance(expected_chunk_index, bool)
            or not isinstance(expected_chunk_index, int)
            or expected_chunk_index < 0
        ):
            raise ValidationError(
                "expected_chunk_index must be a non-negative integer"
            )
        expected_document_id = evaluation_case.expected_document_id
        if expected_document_id is not None and (
            not isinstance(expected_document_id, str)
            or not expected_document_id.strip()
        ):
            raise ValidationError(
                "expected_document_id must not be empty when provided"
            )

    @classmethod
    def _find_relevant_result(
        cls,
        evaluation_case: RetrievalEvaluationCaseDTO,
        retrieval_results: Sequence[RetrievalResult],
    ) -> tuple[int | None, float | None]:
        expected_document_id = evaluation_case.expected_document_id
        if expected_document_id is not None:
            expected_document_id = expected_document_id.strip()

        for rank, result in enumerate(
            retrieval_results[: cls.EVALUATION_TOP_K],
            start=1,
        ):
            if result.chunk_index != evaluation_case.expected_chunk_index:
                continue
            if (
                expected_document_id is not None
                and result.document_id != expected_document_id
            ):
                continue

            try:
                relevant_score = float(result.similarity_score)
            except (TypeError, ValueError) as exc:
                raise RetrievalError(
                    "Retrieval evaluation received an invalid score"
                ) from exc
            if not math.isfinite(relevant_score):
                raise RetrievalError(
                    "Retrieval evaluation received an invalid score"
                )
            return rank, relevant_score

        return None, None

    @staticmethod
    def _hit_rate(
        case_results: Sequence[RetrievalEvaluationCaseResultDTO],
        *,
        cutoff: int,
    ) -> float:
        hits = sum(
            result.relevant_rank is not None
            and result.relevant_rank <= cutoff
            for result in case_results
        )
        return hits / len(case_results)

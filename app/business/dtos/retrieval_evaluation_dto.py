from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCaseDTO:
    query: str
    expected_chunk_index: int
    expected_document_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCaseResultDTO:
    query: str
    expected_chunk_index: int
    expected_document_id: str | None
    relevant_rank: int | None
    relevant_score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReportDTO:
    total_cases: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    average_relevant_score: float | None
    cases: tuple[RetrievalEvaluationCaseResultDTO, ...]

    @property
    def mean_reciprocal_rank(self) -> float:
        """Descriptive alias for callers that avoid metric abbreviations."""

        return self.mrr

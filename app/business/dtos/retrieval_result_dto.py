from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    content: str
    page_number: int | None
    similarity_score: float

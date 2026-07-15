from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextSourceDTO:
    source_number: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    content: str
    similarity_score: float

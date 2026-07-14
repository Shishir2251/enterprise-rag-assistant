from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[UUID] | None = None


class RetrievalResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    content: str
    page_number: int | None
    similarity_score: float

    model_config = {"from_attributes": True}


class RetrievalSearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[RetrievalResultResponse]

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.presentation.schemas.retrieval_schema import RetrievalSearchRequest


class ContextBuildRequest(RetrievalSearchRequest):
    pass


class ContextSourceResponse(BaseModel):
    source_number: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class ContextBuildResponse(BaseModel):
    query: str
    context: str
    sources: list[ContextSourceResponse]
    llm_status: Literal["not_configured"]

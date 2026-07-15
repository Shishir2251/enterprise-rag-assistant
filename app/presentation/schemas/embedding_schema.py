from typing import Literal

from pydantic import BaseModel


class DocumentEmbeddingResponse(BaseModel):
    document_id: str
    embedded_chunks: int
    status: Literal["completed"]
    embedding_provider: str
    embedding_model: str


class DocumentEmbeddingResetResponse(BaseModel):
    document_id: str
    cleared_chunks: int
    status: Literal["cleared"]

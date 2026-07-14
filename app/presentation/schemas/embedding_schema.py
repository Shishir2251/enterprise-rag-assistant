from typing import Literal

from pydantic import BaseModel


class DocumentEmbeddingResponse(BaseModel):
    document_id: str
    embedded_chunks: int
    status: Literal["embedded", "already_embedded"]

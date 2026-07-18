from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatSessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageCreateRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[UUID] = Field(default_factory=list)


class CitationResponse(BaseModel):
    source_number: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    content: str
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class ChatTurnResponse(BaseModel):
    session_id: str
    user_message_id: str
    assistant_message_id: str | None
    status: Literal["completed", "llm_not_configured"]
    answer: str | None
    citations: list[CitationResponse]

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_missing_citations(cls, value: Any) -> Any:
        return [] if value is None else value


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageResponse]

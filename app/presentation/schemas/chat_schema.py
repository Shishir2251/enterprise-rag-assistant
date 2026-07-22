from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.business.dtos.chat_turn_dto import MAX_CHAT_DOCUMENT_IDS


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
    top_k: int | None = Field(default=None, ge=1)
    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_CHAT_DOCUMENT_IDS,
    )

    @field_validator("message", mode="before")
    @classmethod
    def trim_message(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("document_ids")
    @classmethod
    def deduplicate_document_ids(cls, value: list[UUID]) -> list[UUID]:
        return list(dict.fromkeys(value))


class GroundedChatRequest(ChatMessageCreateRequest):
    conversation_id: UUID | None = None
    document_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=MAX_CHAT_DOCUMENT_IDS,
    )


class CitationResponse(BaseModel):
    source_number: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class ChatTurnResponse(BaseModel):
    session_id: str
    user_message_id: str
    assistant_message_id: str | None
    status: Literal["completed", "llm_not_configured"]
    answer: str | None
    citations: list[CitationResponse]
    llm_provider: str | None = None
    llm_model: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    status: Literal["pending", "completed", "failed", "fallback"] = (
        "completed"
    )
    llm_provider: str | None = None
    llm_model: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_missing_citations(cls, value: Any) -> Any:
        return [] if value is None else value


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageResponse]


class GroundedChatResponse(BaseModel):
    conversation_id: str
    message_id: str | None
    answer: str | None
    status: Literal["completed", "llm_not_configured"]
    llm_provider: str | None = None
    llm_model: str | None = None
    citations: list[CitationResponse]


class ConversationDetailResponse(ChatSessionResponse):
    messages: list[ChatMessageResponse]

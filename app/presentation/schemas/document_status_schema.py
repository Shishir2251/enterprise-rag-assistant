from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.presentation.schemas.document_schema import sanitize_document_error


class DocumentStatusResponse(BaseModel):
    document_id: str = Field(validation_alias="id")
    status: str
    progress: int
    current_step: str | None
    error_message: str | None
    retry_count: int
    queued_task_id: str | None = Field(
        default=None,
        validation_alias="task_id",
    )
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("error_message")
    @classmethod
    def hide_unsafe_error_details(cls, value: str | None) -> str | None:
        return sanitize_document_error(value)


class DocumentRetryResponse(BaseModel):
    document_id: str = Field(validation_alias="id")
    status: str
    queued_task_id: str = Field(validation_alias="task_id")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentReindexResponse(BaseModel):
    document_id: str = Field(validation_alias="id")
    status: Literal["queued"]

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

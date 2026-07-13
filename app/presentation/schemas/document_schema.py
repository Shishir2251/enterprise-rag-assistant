from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator


class DocumentResponse(BaseModel):
    id: str
    original_name: str
    mime_type: str
    file_size: int
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("error_message")
    @classmethod
    def hide_unsafe_error_details(cls, value: str | None) -> str | None:
        if value is None:
            return None

        safe_messages = {
            "Document processing failed",
            "No chunks were generated from the document",
            "No readable text was extracted from the document",
            "Uploaded file is unavailable",
        }
        if value in safe_messages or value.startswith(
            "No text extractor registered for extension:"
        ):
            return value
        return "Document processing failed"


class DocumentProcessResponse(BaseModel):
    status: Literal["completed"]
    error_message: str | None = None

    model_config = {"from_attributes": True}

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base


class ChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessageStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_chat_messages_role",
        ),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed', 'fallback')",
            name="ck_chat_messages_status",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=ChatMessageStatus.COMPLETED.value,
        nullable=False,
    )
    llm_provider: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    llm_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    session = relationship(
        "ChatSessionModel",
        back_populates="messages",
    )

    def __init__(self, **kwargs: Any) -> None:
        # SQLAlchemy column defaults are normally applied during INSERT. Set
        # the lifecycle default eagerly as well so service tests and in-memory
        # repositories observe the same state as persisted rows.
        kwargs.setdefault("status", ChatMessageStatus.COMPLETED.value)
        super().__init__(**kwargs)

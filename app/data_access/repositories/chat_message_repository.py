from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.data_access.interfaces.chat_message_repository_interface import (
    IChatMessageRepository,
)
from app.data_access.models.chat_message_model import (
    ChatMessageModel,
    ChatMessageRole,
    ChatMessageStatus,
)
from app.data_access.models.chat_session_model import ChatSessionModel


class ChatMessageRepository(IChatMessageRepository):

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, message: ChatMessageModel) -> ChatMessageModel:
        try:
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            return message
        except Exception:
            self.db.rollback()
            raise

    def list_by_session(
        self,
        session_id: str,
        owner_id: str | None = None,
    ) -> list[ChatMessageModel]:
        statement = select(ChatMessageModel).where(
            ChatMessageModel.session_id == session_id
        )
        if owner_id is not None:
            statement = statement.join(
                ChatSessionModel,
                ChatSessionModel.id == ChatMessageModel.session_id,
            ).where(ChatSessionModel.owner_id == owner_id)
        statement = statement.order_by(
            ChatMessageModel.created_at.asc(),
            ChatMessageModel.id.asc(),
        )
        return list(self.db.scalars(statement).all())

    def list_recent(
        self,
        session_id: str,
        owner_id: str,
        *,
        max_messages: int,
    ) -> list[ChatMessageModel]:
        if max_messages < 0:
            raise ValueError(
                "max_messages must be greater than or equal to zero"
            )
        if max_messages == 0:
            return []

        statement = (
            select(ChatMessageModel)
            .join(
                ChatSessionModel,
                ChatSessionModel.id == ChatMessageModel.session_id,
            )
            .where(
                ChatMessageModel.session_id == session_id,
                ChatSessionModel.owner_id == owner_id,
                or_(
                    ChatMessageModel.role == ChatMessageRole.USER.value,
                    and_(
                        ChatMessageModel.role
                        == ChatMessageRole.ASSISTANT.value,
                        ChatMessageModel.status.in_(
                            (
                                ChatMessageStatus.COMPLETED.value,
                                ChatMessageStatus.FALLBACK.value,
                            )
                        ),
                    ),
                ),
            )
            .order_by(
                ChatMessageModel.created_at.desc(),
                ChatMessageModel.id.desc(),
            )
            .limit(max_messages)
        )
        newest_first = list(self.db.scalars(statement).all())
        newest_first.reverse()
        return newest_first

    def finalize(
        self,
        message: ChatMessageModel,
        *,
        content: str,
        status: ChatMessageStatus | str,
        citations: list[dict[str, Any]] | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> ChatMessageModel:
        normalized_status = (
            status.value if isinstance(status, ChatMessageStatus) else status
        )
        if normalized_status not in {
            ChatMessageStatus.COMPLETED.value,
            ChatMessageStatus.FALLBACK.value,
        }:
            raise ValueError(
                "Assistant messages may only be finalized as completed or fallback"
            )
        message.content = content
        message.status = normalized_status
        message.citations = citations
        message.llm_provider = llm_provider
        message.llm_model = llm_model
        return self._save(message)

    def mark_failed(self, message: ChatMessageModel) -> ChatMessageModel:
        message.status = ChatMessageStatus.FAILED.value
        message.citations = None
        message.llm_provider = None
        message.llm_model = None
        return self._save(message)

    def _save(self, message: ChatMessageModel) -> ChatMessageModel:
        try:
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            return message
        except Exception:
            self.db.rollback()
            raise

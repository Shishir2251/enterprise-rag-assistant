from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_access.interfaces.chat_message_repository_interface import (
    IChatMessageRepository,
)
from app.data_access.models.chat_message_model import ChatMessageModel


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
    ) -> list[ChatMessageModel]:
        statement = (
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(
                ChatMessageModel.created_at.asc(),
                ChatMessageModel.id.asc(),
            )
        )
        return list(self.db.scalars(statement).all())


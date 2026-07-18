from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_access.interfaces.chat_session_repository_interface import (
    IChatSessionRepository,
)
from app.data_access.models.chat_session_model import ChatSessionModel


class ChatSessionRepository(IChatSessionRepository):

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, session: ChatSessionModel) -> ChatSessionModel:
        try:
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            return session
        except Exception:
            self.db.rollback()
            raise

    def get_by_id(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel | None:
        statement = select(ChatSessionModel).where(
            ChatSessionModel.id == session_id,
            ChatSessionModel.owner_id == owner_id,
        )
        return self.db.scalar(statement)

    def list_by_owner(self, owner_id: str) -> list[ChatSessionModel]:
        statement = (
            select(ChatSessionModel)
            .where(ChatSessionModel.owner_id == owner_id)
            .order_by(ChatSessionModel.updated_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def touch(self, session: ChatSessionModel) -> ChatSessionModel:
        session.updated_at = datetime.utcnow()
        try:
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            return session
        except Exception:
            self.db.rollback()
            raise


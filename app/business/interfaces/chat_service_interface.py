from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.data_access.models.chat_message_model import ChatMessageModel
from app.data_access.models.chat_session_model import ChatSessionModel


class IChatService(ABC):

    @abstractmethod
    def create_session(
        self,
        owner_id: str,
        title: str | None = None,
    ) -> ChatSessionModel:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self, owner_id: str) -> list[ChatSessionModel]:
        raise NotImplementedError

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        owner_id: str,
    ) -> list[ChatMessageModel]:
        raise NotImplementedError

    @abstractmethod
    def send_message(
        self,
        session_id: str,
        owner_id: str,
        message: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatTurnDTO:
        raise NotImplementedError


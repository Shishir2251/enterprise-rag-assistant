from abc import ABC, abstractmethod

from app.data_access.models.chat_message_model import ChatMessageModel


class IChatMessageRepository(ABC):

    @abstractmethod
    def create(self, message: ChatMessageModel) -> ChatMessageModel:
        raise NotImplementedError

    @abstractmethod
    def list_by_session(
        self,
        session_id: str,
    ) -> list[ChatMessageModel]:
        raise NotImplementedError


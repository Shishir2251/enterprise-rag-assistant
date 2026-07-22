from abc import ABC, abstractmethod

from app.data_access.models.chat_session_model import ChatSessionModel


class IChatSessionRepository(ABC):

    @abstractmethod
    def create(self, session: ChatSessionModel) -> ChatSessionModel:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel | None:
        raise NotImplementedError

    @abstractmethod
    def list_by_owner(self, owner_id: str) -> list[ChatSessionModel]:
        raise NotImplementedError

    @abstractmethod
    def update_title(
        self,
        session_id: str,
        owner_id: str,
        title: str,
    ) -> ChatSessionModel | None:
        raise NotImplementedError

    @abstractmethod
    def touch(self, session: ChatSessionModel) -> ChatSessionModel:
        raise NotImplementedError

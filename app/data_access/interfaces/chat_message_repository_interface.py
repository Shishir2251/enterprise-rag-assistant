from abc import ABC, abstractmethod

from app.data_access.models.chat_message_model import (
    ChatMessageModel,
    ChatMessageStatus,
)


class IChatMessageRepository(ABC):

    @abstractmethod
    def create(self, message: ChatMessageModel) -> ChatMessageModel:
        raise NotImplementedError

    @abstractmethod
    def list_by_session(
        self,
        session_id: str,
        owner_id: str | None = None,
    ) -> list[ChatMessageModel]:
        raise NotImplementedError

    @abstractmethod
    def list_recent(
        self,
        session_id: str,
        owner_id: str,
        *,
        max_messages: int,
    ) -> list[ChatMessageModel]:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        message: ChatMessageModel,
        *,
        content: str,
        status: ChatMessageStatus | str,
        citations: list[dict] | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> ChatMessageModel:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, message: ChatMessageModel) -> ChatMessageModel:
        raise NotImplementedError

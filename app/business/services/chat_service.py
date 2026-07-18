from collections.abc import Sequence
from dataclasses import asdict

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.business.dtos.llm_dto import LLMMessageDTO
from app.business.interfaces.chat_service_interface import IChatService
from app.business.interfaces.context_builder_interface import IContextBuilder
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.core.exceptions import LLMError, NotFoundError, ValidationError
from app.data_access.interfaces.chat_message_repository_interface import (
    IChatMessageRepository,
)
from app.data_access.interfaces.chat_session_repository_interface import (
    IChatSessionRepository,
)
from app.data_access.models.chat_message_model import (
    ChatMessageModel,
    ChatMessageRole,
)
from app.data_access.models.chat_session_model import ChatSessionModel


class ChatService(IChatService):

    DEFAULT_SESSION_TITLE = "New conversation"
    MAX_MESSAGE_LENGTH = 2000

    def __init__(
        self,
        session_repository: IChatSessionRepository,
        message_repository: IChatMessageRepository,
        retrieval_service: IRetrievalService,
        context_builder: IContextBuilder,
        llm_provider: ILLMProvider,
    ) -> None:
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_provider = llm_provider

    def create_session(
        self,
        owner_id: str,
        title: str | None = None,
    ) -> ChatSessionModel:
        normalized_title = (
            title.strip() if title is not None else self.DEFAULT_SESSION_TITLE
        )
        if not normalized_title:
            raise ValidationError("Chat session title must not be empty")
        if len(normalized_title) > 200:
            raise ValidationError(
                "Chat session title must not exceed 200 characters"
            )

        return self.session_repository.create(
            ChatSessionModel(
                owner_id=owner_id,
                title=normalized_title,
            )
        )

    def list_sessions(self, owner_id: str) -> list[ChatSessionModel]:
        return self.session_repository.list_by_owner(owner_id)

    def get_history(
        self,
        session_id: str,
        owner_id: str,
    ) -> list[ChatMessageModel]:
        self._get_owned_session(session_id, owner_id)
        return self.message_repository.list_by_session(session_id)

    def send_message(
        self,
        session_id: str,
        owner_id: str,
        message: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatTurnDTO:
        session = self._get_owned_session(session_id, owner_id)
        normalized_message = message.strip()
        if not 2 <= len(normalized_message) <= self.MAX_MESSAGE_LENGTH:
            raise ValidationError(
                "Chat message must contain between 2 and 2000 characters"
            )

        previous_messages = self.message_repository.list_by_session(session_id)
        user_message = self.message_repository.create(
            ChatMessageModel(
                session_id=session_id,
                role=ChatMessageRole.USER.value,
                content=normalized_message,
                citations=None,
            )
        )
        self.session_repository.touch(session)

        retrieval_results = self.retrieval_service.search(
            query=normalized_message,
            owner_id=owner_id,
            top_k=top_k,
            document_ids=document_ids,
        )
        context, sources = self.context_builder.build_context(
            retrieval_results
        )
        llm_result = self.llm_provider.generate_answer(
            query=normalized_message,
            context=context,
            conversation_history=[
                LLMMessageDTO(role=item.role, content=item.content)
                for item in previous_messages
            ],
        )

        assistant_message_id: str | None = None
        answer = self._validated_answer(
            status=llm_result.status,
            answer=llm_result.answer,
        )
        if answer is not None:
            assistant_message = self.message_repository.create(
                ChatMessageModel(
                    session_id=session_id,
                    role=ChatMessageRole.ASSISTANT.value,
                    content=answer,
                    citations=[asdict(source) for source in sources],
                )
            )
            assistant_message_id = assistant_message.id
            self.session_repository.touch(session)

        return ChatTurnDTO(
            session_id=session_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message_id,
            status=llm_result.status,
            answer=answer,
            citations=tuple(sources),
        )

    def _get_owned_session(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel:
        session = self.session_repository.get_by_id(
            session_id=session_id,
            owner_id=owner_id,
        )
        if session is None:
            raise NotFoundError("Chat session not found")
        return session

    @staticmethod
    def _validated_answer(
        status: str,
        answer: str | None,
    ) -> str | None:
        if status == "llm_not_configured":
            if answer is not None:
                raise LLMError(
                    "LLM provider returned an invalid disabled response"
                )
            return None
        if status != "completed":
            raise LLMError("LLM provider returned an unsupported status")

        normalized_answer = answer.strip() if answer is not None else ""
        if not normalized_answer:
            raise LLMError(
                "LLM provider completed without returning an answer"
            )
        return normalized_answer


from collections.abc import Sequence
from dataclasses import asdict

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.llm_dto import (
    LLMMessageDTO,
    LLMResponseDTO,
)
from app.business.dtos.prompt_dto import PromptDTO
from app.business.interfaces.chat_service_interface import IChatService
from app.business.interfaces.context_builder_interface import IContextBuilder
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.business.interfaces.prompt_builder_interface import IPromptBuilder
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.business.services.citation_parser_service import (
    CitationParserService,
)
from app.business.services.prompt_builder_service import (
    INSUFFICIENT_CONTEXT_FALLBACK,
    PromptBuilderService,
)
from app.core.exceptions import (
    ApplicationError,
    LLMError,
    LLMProviderError,
    NotFoundError,
    ValidationError,
)
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
        prompt_builder: IPromptBuilder | None = None,
        citation_parser: CitationParserService | None = None,
        history_max_messages: int = 10,
    ) -> None:
        if history_max_messages < 0:
            raise ValueError(
                "history_max_messages must be greater than or equal to zero"
            )
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.history_max_messages = history_max_messages
        # Defaults retain compatibility for direct service construction. The
        # application dependency graph injects both collaborators explicitly.
        self.prompt_builder = prompt_builder or PromptBuilderService(
            history_max_messages=history_max_messages
        )
        self.citation_parser = citation_parser or CitationParserService()

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
        conversation_history = self._bounded_conversation_history(
            previous_messages
        )

        if not context.strip() or not sources:
            assistant_message_id = self._persist_assistant_message(
                session=session,
                content=INSUFFICIENT_CONTEXT_FALLBACK,
                citations=(),
            )
            return ChatTurnDTO(
                session_id=session_id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message_id,
                status="completed",
                answer=INSUFFICIENT_CONTEXT_FALLBACK,
                citations=(),
            )

        if not self._provider_is_configured():
            return ChatTurnDTO(
                session_id=session_id,
                user_message_id=user_message.id,
                assistant_message_id=None,
                status="llm_not_configured",
                answer=None,
                citations=tuple(sources),
            )

        prompt = self.prompt_builder.build_grounded_prompt(
            query=normalized_message,
            context=context,
            conversation_history=conversation_history,
        )
        answer, citations = self._generate_answer(
            prompt=prompt,
            supplied_sources=sources,
        )
        assistant_message_id = self._persist_assistant_message(
            session=session,
            content=answer,
            citations=citations,
        )

        return ChatTurnDTO(
            session_id=session_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message_id,
            status="completed",
            answer=answer,
            citations=citations,
        )

    def _bounded_conversation_history(
        self,
        messages: Sequence[ChatMessageModel],
    ) -> tuple[LLMMessageDTO, ...]:
        history = tuple(
            LLMMessageDTO(role=item.role, content=item.content)
            for item in messages
            if item.role in {
                ChatMessageRole.USER.value,
                ChatMessageRole.ASSISTANT.value,
            }
            and item.content.strip()
        )
        if self.history_max_messages == 0:
            return ()
        return history[-self.history_max_messages :]

    def _provider_is_configured(self) -> bool:
        configured = getattr(self.llm_provider, "is_configured", None)
        if isinstance(configured, bool):
            return configured
        provider_name = str(
            getattr(self.llm_provider, "provider_name", "")
        ).strip().lower()
        return provider_name not in {"", "none", "disabled"}

    def _generate_answer(
        self,
        *,
        prompt: PromptDTO,
        supplied_sources: Sequence[ContextSourceDTO],
    ) -> tuple[str, tuple[ContextSourceDTO, ...]]:
        try:
            response = self.llm_provider.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                conversation_history=prompt.conversation_history,
            )
            answer = self._validated_llm_response(response)
            citations = self.citation_parser.parse(
                answer,
                supplied_sources,
            )
            return answer, citations
        except ApplicationError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM provider request failed."
            ) from exc

    def _persist_assistant_message(
        self,
        *,
        session: ChatSessionModel,
        content: str,
        citations: Sequence[ContextSourceDTO],
    ) -> str:
        assistant_message = self.message_repository.create(
            ChatMessageModel(
                session_id=session.id,
                role=ChatMessageRole.ASSISTANT.value,
                content=content,
                citations=[asdict(source) for source in citations],
            )
        )
        self.session_repository.touch(session)
        return assistant_message.id

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
    def _validated_llm_response(response: LLMResponseDTO) -> str:
        if not isinstance(response, LLMResponseDTO):
            raise LLMError("LLM provider returned an invalid response")
        normalized_answer = response.content.strip()
        if not normalized_answer:
            raise LLMError(
                "LLM provider completed without returning an answer"
            )
        return normalized_answer

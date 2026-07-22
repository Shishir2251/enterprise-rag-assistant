import inspect
import logging
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any, TypeVar

from app.business.dtos.chat_turn_dto import (
    MAX_CHAT_DOCUMENT_IDS,
    ChatTurnDTO,
)
from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.llm_dto import LLMMessageDTO, LLMResponseDTO
from app.business.dtos.prompt_dto import PromptDTO
from app.business.interfaces.chat_service_interface import IChatService
from app.business.interfaces.context_builder_interface import IContextBuilder
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.business.interfaces.prompt_builder_interface import IPromptBuilder
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.business.services.citation_parser_service import CitationParserService
from app.business.services.prompt_builder_service import (
    INSUFFICIENT_CONTEXT_FALLBACK,
    PromptBuilderService,
)
from app.core.exceptions import (
    ApplicationError,
    ConflictError,
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
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.chat_message_model import (
    ChatMessageModel,
    ChatMessageRole,
    ChatMessageStatus,
)
from app.data_access.models.chat_session_model import ChatSessionModel
from app.data_access.models.document_model import DocumentStatus


logger = logging.getLogger(__name__)
T = TypeVar("T")


class ChatService(IChatService):

    DEFAULT_SESSION_TITLE = "New conversation"
    MAX_MESSAGE_LENGTH = 2000
    MAX_AUTOMATIC_TITLE_LENGTH = 80

    def __init__(
        self,
        session_repository: IChatSessionRepository,
        message_repository: IChatMessageRepository,
        retrieval_service: IRetrievalService,
        context_builder: IContextBuilder,
        llm_provider: ILLMProvider,
        prompt_builder: IPromptBuilder | None = None,
        citation_parser: CitationParserService | None = None,
        document_repository: IDocumentRepository | None = None,
        history_max_messages: int = 10,
        history_max_characters: int = 6000,
        default_top_k: int = 5,
        maximum_top_k: int = 10,
        no_context_message: str = INSUFFICIENT_CONTEXT_FALLBACK,
    ) -> None:
        if history_max_messages < 0:
            raise ValueError(
                "history_max_messages must be greater than or equal to zero"
            )
        if history_max_characters < 0:
            raise ValueError(
                "history_max_characters must be greater than or equal to zero"
            )
        if maximum_top_k <= 0:
            raise ValueError("maximum_top_k must be greater than zero")
        if not 1 <= default_top_k <= maximum_top_k:
            raise ValueError(
                "default_top_k must be between 1 and maximum_top_k"
            )
        normalized_fallback = no_context_message.strip()
        if not normalized_fallback:
            raise ValueError("no_context_message must not be empty")

        self.session_repository = session_repository
        self.message_repository = message_repository
        self.document_repository = document_repository
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.history_max_messages = history_max_messages
        self.history_max_characters = history_max_characters
        self.default_top_k = default_top_k
        self.maximum_top_k = maximum_top_k
        self.no_context_message = normalized_fallback
        # Defaults retain compatibility for direct service construction. The
        # application dependency graph injects both collaborators explicitly.
        self.prompt_builder = prompt_builder or PromptBuilderService(
            history_max_messages=history_max_messages,
            history_max_characters=history_max_characters,
            no_context_message=normalized_fallback,
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

        session = self._repository_call(
            self.session_repository.create,
            ChatSessionModel(owner_id=owner_id, title=normalized_title),
        )
        logger.info(
            "Chat conversation created",
            extra={
                "session_id": session.id,
                "owner_id": owner_id,
            },
        )
        return session

    def list_sessions(self, owner_id: str) -> list[ChatSessionModel]:
        return self._repository_call(
            self.session_repository.list_by_owner,
            owner_id,
        )

    def get_session(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel:
        return self._get_owned_session(session_id, owner_id)

    def get_history(
        self,
        session_id: str,
        owner_id: str,
    ) -> list[ChatMessageModel]:
        self._get_owned_session(session_id, owner_id)
        messages = self._list_session_messages(session_id, owner_id)
        return [
            message
            for message in messages
            if self._is_valid_history_message(message)
        ]

    async def send_message(
        self,
        session_id: str | None,
        owner_id: str,
        message: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatTurnDTO:
        if not isinstance(message, str):
            raise ValidationError("Chat message must be text")
        normalized_message = " ".join(message.split())
        if not 2 <= len(normalized_message) <= self.MAX_MESSAGE_LENGTH:
            raise ValidationError(
                "Chat message must contain between 2 and 2000 characters"
            )

        result_limit = self.default_top_k if top_k is None else top_k
        if (
            isinstance(result_limit, bool)
            or not isinstance(result_limit, int)
            or not 1 <= result_limit <= self.maximum_top_k
        ):
            raise ValidationError(
                f"top_k must be between 1 and {self.maximum_top_k}"
            )

        if session_id is not None:
            session = self._get_owned_session(session_id, owner_id)
        else:
            session = self.create_session(
                owner_id,
                self._automatic_title(normalized_message),
            )
        selected_document_ids = self._validate_documents(
            document_ids,
            owner_id,
        )

        logger.info(
            "Chat request started",
            extra={
                "session_id": session.id,
                "owner_id": owner_id,
                "document_count": len(selected_document_ids or ()),
                "top_k": result_limit,
            },
        )
        user_message = self._create_message(
            ChatMessageModel(
                session_id=session.id,
                role=ChatMessageRole.USER.value,
                content=normalized_message,
                status=ChatMessageStatus.COMPLETED.value,
                citations=None,
            )
        )
        self._touch_session(session)
        conversation_history = self._load_previous_history(
            session_id=session.id,
            owner_id=owner_id,
            current_message_id=user_message.id,
        )

        retrieval_results = self.retrieval_service.search(
            query=normalized_message,
            owner_id=owner_id,
            top_k=result_limit,
            document_ids=selected_document_ids,
        )
        context, sources = self.context_builder.build_context(retrieval_results)
        logger.info(
            "Chat retrieval completed",
            extra={
                "session_id": session.id,
                "retrieval_result_count": len(retrieval_results),
                "included_source_count": len(sources),
            },
        )

        if not context.strip() or not sources:
            assistant_message = self._persist_assistant_message(
                session=session,
                content=self.no_context_message,
                citations=(),
                status=ChatMessageStatus.FALLBACK,
            )
            logger.info(
                "Chat no-context fallback used",
                extra={"session_id": session.id},
            )
            return ChatTurnDTO(
                session_id=session.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                status="completed",
                answer=self.no_context_message,
                citations=(),
            )

        selected_provider_name = str(
            getattr(self.llm_provider, "provider_name", "unknown")
        )
        logger.info(
            "Chat provider selected",
            extra={
                "session_id": session.id,
                "llm_provider": selected_provider_name,
            },
        )
        if not self._provider_is_configured():
            return ChatTurnDTO(
                session_id=session.id,
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
        answer, citations, response = await self._generate_answer(
            prompt=prompt,
            supplied_sources=sources,
            session_id=session.id,
        )
        assistant_status = (
            ChatMessageStatus.FALLBACK
            if answer == self.no_context_message
            else ChatMessageStatus.COMPLETED
        )
        if assistant_status is ChatMessageStatus.FALLBACK:
            citations = ()
        assistant_message = self._persist_assistant_message(
            session=session,
            content=answer,
            citations=citations,
            status=assistant_status,
            llm_provider=response.provider,
            llm_model=response.model,
        )
        logger.info(
            "Chat answer persisted",
            extra={
                "session_id": session.id,
                "assistant_message_id": assistant_message.id,
                "citation_count": len(citations),
                "llm_provider": response.provider,
                "llm_model": response.model,
            },
        )

        return ChatTurnDTO(
            session_id=session.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            status="completed",
            answer=answer,
            citations=citations,
            llm_provider=response.provider,
            llm_model=response.model,
        )

    def _load_previous_history(
        self,
        *,
        session_id: str,
        owner_id: str,
        current_message_id: str,
    ) -> tuple[LLMMessageDTO, ...]:
        if self.history_max_messages == 0 or self.history_max_characters == 0:
            return ()

        list_recent = getattr(self.message_repository, "list_recent", None)
        if callable(list_recent):
            messages = self._repository_call(
                list_recent,
                session_id,
                owner_id,
                max_messages=self.history_max_messages + 1,
            )
        else:
            messages = self._list_session_messages(session_id, owner_id)

        return self._bounded_conversation_history(
            [
                item
                for item in messages
                if item.id != current_message_id
            ]
        )

    def _bounded_conversation_history(
        self,
        messages: Sequence[ChatMessageModel],
    ) -> tuple[LLMMessageDTO, ...]:
        valid_history = [
            LLMMessageDTO(role=item.role, content=item.content.strip())
            for item in messages
            if self._is_valid_history_message(item)
        ]
        recent_history = valid_history[-self.history_max_messages :]
        selected: list[LLMMessageDTO] = []
        current_characters = 0
        for item in reversed(recent_history):
            item_length = len(item.content)
            if current_characters + item_length > self.history_max_characters:
                continue
            selected.append(item)
            current_characters += item_length
        selected.reverse()
        return tuple(selected)

    @staticmethod
    def _is_valid_history_message(message: ChatMessageModel) -> bool:
        if message.role not in {
            ChatMessageRole.USER.value,
            ChatMessageRole.ASSISTANT.value,
        }:
            return False
        if not isinstance(message.content, str) or not message.content.strip():
            return False
        if message.role == ChatMessageRole.USER.value:
            return True
        status = getattr(message, "status", None)
        return status in {
            None,
            ChatMessageStatus.COMPLETED.value,
            ChatMessageStatus.FALLBACK.value,
        }

    def _validate_documents(
        self,
        document_ids: Sequence[str] | None,
        owner_id: str,
    ) -> list[str] | None:
        if isinstance(document_ids, (str, bytes)):
            raise ValidationError("document_ids must be a list")
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for raw_document_id in document_ids or ():
            document_id = str(raw_document_id).strip()
            if not document_id:
                raise ValidationError("Document IDs must not be empty")
            if document_id not in seen:
                seen.add(document_id)
                normalized_ids.append(document_id)

        if len(normalized_ids) > MAX_CHAT_DOCUMENT_IDS:
            raise ValidationError(
                "A chat request may select at most "
                f"{MAX_CHAT_DOCUMENT_IDS} documents"
            )

        # Direct service construction predates explicit document validation.
        # The application dependency graph always supplies this repository.
        if self.document_repository is None:
            return normalized_ids or None

        list_by_ids = getattr(self.document_repository, "list_by_ids", None)
        if callable(list_by_ids):
            documents = self._repository_call(
                list_by_ids,
                normalized_ids,
                owner_id,
            )
            documents_by_id = {document.id: document for document in documents}
            ordered_documents = [
                documents_by_id.get(document_id)
                for document_id in normalized_ids
            ]
            if any(document is None for document in ordered_documents):
                # The owner-scoped batch query intentionally makes unknown
                # and cross-owner document IDs indistinguishable.
                raise NotFoundError("Document not found")
            for document in ordered_documents:
                if (
                    document is not None
                    and document.status
                    not in DocumentStatus.process_complete_values()
                ):
                    raise ConflictError("Document is not ready for retrieval")
            return normalized_ids or None

        # Compatibility for legacy direct-service test doubles. Production
        # repositories provide the owner-scoped batch method above.
        for document_id in normalized_ids:
            document = self._repository_call(
                self.document_repository.get_by_id,
                document_id=document_id,
                owner_id=owner_id,
            )
            if document is None:
                # Owner-scoped lookup intentionally makes unknown and
                # cross-owner document IDs indistinguishable.
                raise NotFoundError("Document not found")
            if document.status not in DocumentStatus.process_complete_values():
                raise ConflictError("Document is not ready for retrieval")
        return normalized_ids or None

    def _provider_is_configured(self) -> bool:
        configured = getattr(self.llm_provider, "is_configured", None)
        if isinstance(configured, bool):
            return configured
        provider_name = str(
            getattr(self.llm_provider, "provider_name", "")
        ).strip().lower()
        return provider_name not in {"", "none", "disabled"}

    async def _generate_answer(
        self,
        *,
        prompt: PromptDTO,
        supplied_sources: Sequence[ContextSourceDTO],
        session_id: str,
    ) -> tuple[str, tuple[ContextSourceDTO, ...], LLMResponseDTO]:
        provider_name = str(
            getattr(self.llm_provider, "provider_name", "unknown")
        )
        started_at = perf_counter()
        try:
            generated = self.llm_provider.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                conversation_history=prompt.conversation_history,
            )
            response = await generated if inspect.isawaitable(generated) else generated
            answer = self._validated_llm_response(response)
            citations = self.citation_parser.parse(answer, supplied_sources)
            logger.info(
                "Chat provider completed",
                extra={
                    "session_id": session_id,
                    "llm_provider": provider_name,
                    "provider_latency_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )
            return answer, citations, response
        except ApplicationError as exc:
            provider_latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )
            logger.warning(
                "Chat provider failed",
                extra={
                    "session_id": session_id,
                    "llm_provider": provider_name,
                    "failure_category": type(exc).__name__,
                    "provider_latency_ms": provider_latency_ms,
                },
            )
            raise
        except Exception as exc:
            provider_latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )
            logger.warning(
                "Chat provider failed",
                extra={
                    "session_id": session_id,
                    "llm_provider": provider_name,
                    "failure_category": type(exc).__name__,
                    "provider_latency_ms": provider_latency_ms,
                },
            )
            raise LLMProviderError("LLM provider request failed.") from exc

    def _persist_assistant_message(
        self,
        *,
        session: ChatSessionModel,
        content: str,
        citations: Sequence[ContextSourceDTO],
        status: ChatMessageStatus,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> ChatMessageModel:
        assistant_message = self._create_message(
            ChatMessageModel(
                session_id=session.id,
                role=ChatMessageRole.ASSISTANT.value,
                content=content,
                status=status.value,
                citations=[
                    self._citation_metadata(source) for source in citations
                ],
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        )
        self._touch_session(session)
        return assistant_message

    @staticmethod
    def _citation_metadata(source: ContextSourceDTO) -> dict[str, object]:
        return {
            "source_number": source.source_number,
            "chunk_id": source.chunk_id,
            "document_id": source.document_id,
            "document_name": source.document_name,
            "page_number": source.page_number,
            "chunk_index": source.chunk_index,
            "similarity_score": source.similarity_score,
        }

    def _get_owned_session(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel:
        session = self._repository_call(
            self.session_repository.get_by_id,
            session_id=session_id,
            owner_id=owner_id,
        )
        if session is None:
            raise NotFoundError("Chat session not found")
        return session

    def _list_session_messages(
        self,
        session_id: str,
        owner_id: str,
    ) -> list[ChatMessageModel]:
        try:
            return self.message_repository.list_by_session(session_id, owner_id)
        except TypeError:
            # Compatibility for small in-memory repositories used by existing
            # direct service tests. Ownership was already checked above.
            return self._repository_call(
                self.message_repository.list_by_session,
                session_id,
            )
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(
                "Chat persistence is unavailable."
            ) from exc

    def _create_message(self, message: ChatMessageModel) -> ChatMessageModel:
        return self._repository_call(self.message_repository.create, message)

    def _touch_session(self, session: ChatSessionModel) -> None:
        self._repository_call(self.session_repository.touch, session)

    @classmethod
    def _automatic_title(cls, message: str) -> str:
        if len(message) <= cls.MAX_AUTOMATIC_TITLE_LENGTH:
            return message
        prefix_length = cls.MAX_AUTOMATIC_TITLE_LENGTH - 3
        return f"{message[:prefix_length].rstrip()}..."

    @staticmethod
    def _repository_call(
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        try:
            return operation(*args, **kwargs)
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError("Chat persistence is unavailable.") from exc

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

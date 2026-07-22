import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.dialects import postgresql

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.llm_dto import LLMResponseDTO
from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.services.chat_service import ChatService
from app.business.services.context_builder_service import ContextBuilderService
from app.core.exceptions import ConfigurationError, NotFoundError
from app.data_access.models.chat_message_model import ChatMessageRole
from app.data_access.models.chat_session_model import ChatSessionModel
from app.data_access.repositories.chat_session_repository import (
    ChatSessionRepository,
)
from app.infrastructure.llm.llm_provider_factory import create_llm_provider
from app.infrastructure.llm.no_llm_provider import NoLLMProvider
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import get_chat_service


NOW = datetime(2026, 7, 18, 12, 0, 0)


def make_session(
    session_id: str = "session-id",
    owner_id: str = "owner-id",
) -> ChatSessionModel:
    return ChatSessionModel(
        id=session_id,
        owner_id=owner_id,
        title="RAG questions",
        created_at=NOW,
        updated_at=NOW,
    )


def make_retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-id",
        document_id="document-id",
        document_name="architecture.pdf",
        page_number=2,
        chunk_index=3,
        content="FastAPI and PostgreSQL are used.",
        similarity_score=0.88,
    )


class FakeSessionRepository:
    def __init__(self, session: ChatSessionModel | None = None) -> None:
        self.sessions = [session] if session is not None else []
        self.touched: list[str] = []

    def create(self, session: ChatSessionModel) -> ChatSessionModel:
        session.id = session.id or f"session-{len(self.sessions) + 1}"
        session.created_at = session.created_at or NOW
        session.updated_at = session.updated_at or NOW
        self.sessions.append(session)
        return session

    def get_by_id(
        self,
        session_id: str,
        owner_id: str,
    ) -> ChatSessionModel | None:
        return next(
            (
                session
                for session in self.sessions
                if session.id == session_id and session.owner_id == owner_id
            ),
            None,
        )

    def list_by_owner(self, owner_id: str) -> list[ChatSessionModel]:
        return [
            session
            for session in self.sessions
            if session.owner_id == owner_id
        ]

    def touch(self, session: ChatSessionModel) -> ChatSessionModel:
        self.touched.append(session.id)
        return session


class FakeMessageRepository:
    def __init__(self) -> None:
        self.messages = []

    def create(self, message):
        message.id = message.id or f"message-{len(self.messages) + 1}"
        message.created_at = message.created_at or NOW
        self.messages.append(message)
        return message

    def list_by_session(self, session_id: str):
        return [
            message
            for message in self.messages
            if message.session_id == session_id
        ]


class FakeRetrievalService:
    def __init__(self, results=None) -> None:
        self.results = (
            [make_retrieval_result()] if results is None else results
        )
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class AnsweringLLMProvider:
    provider_name = "test"
    is_configured = True

    def __init__(
        self,
        answer: str = "A grounded answer. [SOURCE 1]",
    ) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> LLMResponseDTO:
        self.calls.append(kwargs)
        return LLMResponseDTO(
            content=self.answer,
            provider="test",
            model="test-model",
        )


class ChatServiceTests(unittest.TestCase):
    def make_service(self, llm_provider=None):
        session_repository = FakeSessionRepository(make_session())
        message_repository = FakeMessageRepository()
        retrieval_service = FakeRetrievalService()
        provider = llm_provider or NoLLMProvider()
        service = ChatService(
            session_repository=session_repository,
            message_repository=message_repository,
            retrieval_service=retrieval_service,
            context_builder=ContextBuilderService(),
            llm_provider=provider,
        )
        return (
            service,
            session_repository,
            message_repository,
            retrieval_service,
            provider,
        )

    def test_disabled_llm_persists_only_user_message(self) -> None:
        (
            service,
            session_repository,
            message_repository,
            retrieval_service,
            _,
        ) = self.make_service()

        result = asyncio.run(
            service.send_message(
                session_id="session-id",
                owner_id="owner-id",
                message="What technologies are used?",
                top_k=4,
                document_ids=["document-id"],
            )
        )

        self.assertEqual(result.status, "llm_not_configured")
        self.assertIsNone(result.answer)
        self.assertIsNone(result.assistant_message_id)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].document_name, "architecture.pdf")
        self.assertEqual(len(message_repository.messages), 1)
        self.assertEqual(
            message_repository.messages[0].role,
            ChatMessageRole.USER.value,
        )
        self.assertIsNone(message_repository.messages[0].citations)
        self.assertEqual(session_repository.touched, ["session-id"])
        self.assertEqual(
            retrieval_service.calls,
            [
                {
                    "query": "What technologies are used?",
                    "owner_id": "owner-id",
                    "top_k": 4,
                    "document_ids": ["document-id"],
                }
            ],
        )

    def test_assistant_is_persisted_only_when_provider_returns_answer(
        self,
    ) -> None:
        provider = AnsweringLLMProvider()
        service, _, message_repository, _, _ = self.make_service(provider)

        result = asyncio.run(
            service.send_message(
                "session-id",
                "owner-id",
                "What technologies are used?",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "A grounded answer. [SOURCE 1]")
        self.assertEqual(result.assistant_message_id, "message-2")
        self.assertEqual(
            [message.role for message in message_repository.messages],
            ["user", "assistant"],
        )
        assistant = message_repository.messages[1]
        self.assertEqual(assistant.content, "A grounded answer. [SOURCE 1]")
        self.assertEqual(
            assistant.citations[0]["document_name"],
            "architecture.pdf",
        )

    def test_previous_history_is_passed_without_duplicate_current_message(
        self,
    ) -> None:
        provider = AnsweringLLMProvider()
        service, _, message_repository, _, _ = self.make_service(provider)
        message_repository.create(
            SimpleNamespace(
                id=None,
                session_id="session-id",
                role="user",
                content="Earlier question",
                citations=None,
                created_at=NOW,
            )
        )
        message_repository.create(
            SimpleNamespace(
                id=None,
                session_id="session-id",
                role="assistant",
                content="Earlier answer",
                citations=[],
                created_at=NOW,
            )
        )

        asyncio.run(
            service.send_message(
                "session-id",
                "owner-id",
                "Current question",
            )
        )

        history = provider.calls[0]["conversation_history"]
        self.assertEqual(
            [(item.role, item.content) for item in history],
            [
                ("user", "Earlier question"),
                ("assistant", "Earlier answer"),
            ],
        )

    def test_other_owner_cannot_read_or_write_session(self) -> None:
        service, _, message_repository, retrieval_service, _ = (
            self.make_service()
        )

        with self.assertRaises(NotFoundError):
            service.get_history("session-id", "other-owner")
        with self.assertRaises(NotFoundError):
            asyncio.run(
                service.send_message(
                    "session-id",
                    "other-owner",
                    "Unauthorized question",
                )
            )

        self.assertEqual(message_repository.messages, [])
        self.assertEqual(retrieval_service.calls, [])

    def test_user_message_remains_persisted_if_retrieval_fails(self) -> None:
        service, _, message_repository, retrieval_service, _ = (
            self.make_service()
        )
        retrieval_service.search = Mock(
            side_effect=RuntimeError("retrieval unavailable")
        )

        with self.assertRaises(RuntimeError):
            asyncio.run(
                service.send_message(
                    "session-id",
                    "owner-id",
                    "Persist this question",
                )
            )

        self.assertEqual(len(message_repository.messages), 1)
        self.assertEqual(message_repository.messages[0].role, "user")

    def test_history_returns_persisted_messages_in_repository_order(
        self,
    ) -> None:
        service, _, message_repository, _, _ = self.make_service()
        for role, content in (
            ("user", "First question"),
            ("assistant", "First answer"),
        ):
            message_repository.create(
                SimpleNamespace(
                    id=None,
                    session_id="session-id",
                    role=role,
                    content=content,
                    citations=None,
                    created_at=NOW,
                )
            )

        history = service.get_history("session-id", "owner-id")

        self.assertEqual(
            [(message.role, message.content) for message in history],
            [
                ("user", "First question"),
                ("assistant", "First answer"),
            ],
        )


class LLMProviderFactoryTests(unittest.TestCase):
    def test_none_provider_requires_no_api_key_or_network_client(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER="NoNe",
            OPENAI_API_KEY=SecretStr(""),
        )

        provider = create_llm_provider(config)
        result = provider.generate_answer(
            query="Question",
            context="Grounded context",
            conversation_history=[],
        )

        self.assertIsInstance(provider, NoLLMProvider)
        self.assertEqual(result.status, "llm_not_configured")
        self.assertIsNone(result.answer)

    def test_unknown_provider_is_rejected_without_fallback_answer(self) -> None:
        config = SimpleNamespace(LLM_PROVIDER="fake-answer")

        with self.assertRaises(ConfigurationError):
            create_llm_provider(config)


class ChatSessionRepositoryTests(unittest.TestCase):
    def test_session_lookup_is_owner_scoped(self) -> None:
        db = Mock()
        repository = ChatSessionRepository(db)

        repository.get_by_id("session-id", "owner-id")

        statement = db.scalar.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("chat_sessions.id =", sql)
        self.assertIn("chat_sessions.owner_id =", sql)
        self.assertEqual(compiled.params["id_1"], "session-id")
        self.assertEqual(compiled.params["owner_id_1"], "owner-id")


class ChatEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_send_message_requires_authentication(self) -> None:
        response = TestClient(app).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={"message": "What technologies are used?"},
        )

        self.assertEqual(response.status_code, 401)

    def test_disabled_llm_response_has_null_answer_and_citations(self) -> None:
        chat_service = Mock()
        chat_service.send_message.return_value = ChatTurnDTO(
            session_id="session-id",
            user_message_id="user-message-id",
            assistant_message_id=None,
            status="llm_not_configured",
            answer=None,
            citations=(
                ContextSourceDTO(
                    source_number=1,
                    chunk_id="chunk-id",
                    document_id="document-id",
                    document_name="architecture.pdf",
                    page_number=2,
                    chunk_index=3,
                    content="FastAPI and PostgreSQL are used.",
                    similarity_score=0.88,
                ),
            ),
        )
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_chat_service] = lambda: chat_service

        response = TestClient(app).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={
                "message": "What technologies are used?",
                "document_ids": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "llm_not_configured")
        self.assertIsNone(body["answer"])
        self.assertIsNone(body["assistant_message_id"])
        self.assertEqual(body["citations"][0]["document_name"], "architecture.pdf")
        self.assertNotIn("embedding", body["citations"][0])
        chat_service.send_message.assert_called_once_with(
            session_id="session-id",
            owner_id="owner-id",
            message="What technologies are used?",
            top_k=None,
            document_ids=None,
        )

    def test_history_normalizes_missing_citations_to_empty_list(self) -> None:
        chat_service = Mock()
        chat_service.get_history.return_value = [
            SimpleNamespace(
                id="message-id",
                role="user",
                content="Persisted question",
                citations=None,
                created_at=NOW,
            )
        ]
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_chat_service] = lambda: chat_service

        response = TestClient(app).get(
            "/api/v1/chat/sessions/session-id/messages"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["messages"][0]["citations"], [])


if __name__ == "__main__":
    unittest.main()

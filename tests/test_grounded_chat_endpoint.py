import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.services.prompt_builder_service import (
    INSUFFICIENT_CONTEXT_FALLBACK,
)
from app.core.exceptions import LLMProviderError, LLMTimeoutError
from app.main import app
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import get_chat_service


class GroundedChatEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _authenticated_client(self, chat_service: Mock) -> TestClient:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="owner-id"
        )
        app.dependency_overrides[get_chat_service] = lambda: chat_service
        return TestClient(app)

    def test_completed_answer_returns_persisted_id_and_safe_citations(
        self,
    ) -> None:
        chat_service = Mock()
        chat_service.send_message.return_value = ChatTurnDTO(
            session_id="session-id",
            user_message_id="user-message-id",
            assistant_message_id="assistant-message-id",
            status="completed",
            answer="The policy requires approval. [SOURCE 1]",
            citations=(
                ContextSourceDTO(
                    source_number=1,
                    chunk_id="chunk-id",
                    document_id="document-id",
                    document_name="policy.pdf",
                    page_number=4,
                    chunk_index=2,
                    content="Approval is required.",
                    similarity_score=0.91,
                ),
            ),
        )
        response = self._authenticated_client(chat_service).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={"message": "What does the policy require?"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["assistant_message_id"], "assistant-message-id")
        self.assertEqual(body["citations"][0]["source_number"], 1)
        serialized = response.text.lower()
        for forbidden in (
            "embedding",
            "stored_name",
            "file_path",
            "openai_api_key",
            "jwt_secret",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_no_context_fallback_is_completed_without_citations(self) -> None:
        chat_service = Mock()
        chat_service.send_message.return_value = ChatTurnDTO(
            session_id="session-id",
            user_message_id="user-message-id",
            assistant_message_id="assistant-message-id",
            status="completed",
            answer=INSUFFICIENT_CONTEXT_FALLBACK,
            citations=(),
        )
        response = self._authenticated_client(chat_service).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={"message": "What is not covered?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session_id": "session-id",
                "user_message_id": "user-message-id",
                "assistant_message_id": "assistant-message-id",
                "status": "completed",
                "answer": INSUFFICIENT_CONTEXT_FALLBACK,
                "citations": [],
            },
        )

    def test_provider_error_response_is_sanitized(self) -> None:
        chat_service = Mock()
        unsafe_error = RuntimeError(
            "sk-proj-secret C:\\internal\\documents\\stored-file.txt"
        )
        safe_error = LLMProviderError("LLM provider request failed.")
        safe_error.__cause__ = unsafe_error
        chat_service.send_message.side_effect = safe_error

        response = self._authenticated_client(chat_service).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={"message": "Trigger a provider failure"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "LLM provider request failed."},
        )
        self.assertNotIn("sk-proj-secret", response.text)
        self.assertNotIn("stored-file", response.text)

    def test_provider_timeout_response_is_sanitized(self) -> None:
        chat_service = Mock()
        chat_service.send_message.side_effect = LLMTimeoutError(
            "LLM provider timed out."
        )

        response = self._authenticated_client(chat_service).post(
            "/api/v1/chat/sessions/session-id/messages",
            json={"message": "Trigger a provider timeout"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "LLM provider timed out."},
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import APIConnectionError
from openai import APIStatusError
from openai import APITimeoutError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError
from pydantic import SecretStr

from app.business.dtos.llm_dto import LLMMessageDTO, LLMResponseDTO
from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.infrastructure.llm.llm_provider_factory import create_llm_provider
from app.infrastructure.llm.fake_llm_provider import FakeLLMProvider
from app.infrastructure.llm.no_llm_provider import NoLLMProvider
from app.infrastructure.llm.openai_llm_provider import OpenAILLMProvider


LOGGER_NAME = "app.infrastructure.llm.openai_llm_provider"
TEST_API_KEY = "sk-proj-test-secret-never-log"


def make_provider(client: Mock) -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key=TEST_API_KEY,
        model_name="gpt-4.1-mini",
        temperature=0.1,
        max_output_tokens=1200,
        timeout_seconds=30,
        client=client,
    )


class OpenAILLMProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.client.responses.create = AsyncMock()
        self.provider = make_provider(self.client)

    async def test_successful_response_maps_dto_and_sends_grounded_request(
        self,
    ) -> None:
        self.client.responses.create.return_value = SimpleNamespace(
            output_text="  A grounded answer [SOURCE 1].  ",
            status="completed",
            model="gpt-4.1-mini-2025-04-14",
            usage=SimpleNamespace(input_tokens=87, output_tokens=14),
        )
        history = [
            LLMMessageDTO(role="user", content="Earlier question"),
            LLMMessageDTO(role="assistant", content="Earlier answer"),
        ]

        result = await self.provider.generate(
            system_prompt="  Strict grounding system prompt.  ",
            user_prompt="  DOCUMENT CONTEXT and current question.  ",
            conversation_history=history,
        )

        self.assertIsInstance(result, LLMResponseDTO)
        self.assertEqual(result.content, "A grounded answer [SOURCE 1].")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-4.1-mini-2025-04-14")
        self.assertEqual(result.input_tokens, 87)
        self.assertEqual(result.output_tokens, 14)
        self.assertEqual(result.finish_reason, "completed")
        self.client.responses.create.assert_called_once_with(
            model="gpt-4.1-mini",
            instructions="Strict grounding system prompt.",
            input=[
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {
                    "role": "user",
                    "content": "DOCUMENT CONTEXT and current question.",
                },
            ],
            temperature=0.1,
            max_output_tokens=1200,
            timeout=30.0,
            store=False,
        )

    async def test_empty_or_malformed_output_is_rejected(self) -> None:
        responses = (
            SimpleNamespace(
                output_text="",
                status="completed",
                model="gpt-4.1-mini",
                usage=None,
            ),
            SimpleNamespace(
                output_text="  \n\t ",
                status="completed",
                model="gpt-4.1-mini",
                usage=None,
            ),
            SimpleNamespace(
                output_text=["not", "text"],
                status="completed",
                model="gpt-4.1-mini",
                usage=None,
            ),
            SimpleNamespace(
                status="completed",
                model="gpt-4.1-mini",
                usage=None,
            ),
        )

        for response in responses:
            with self.subTest(response=response):
                self.client.responses.create.return_value = response
                with self.assertRaisesRegex(
                    LLMProviderError,
                    "LLM provider returned an invalid response",
                ):
                    await self.provider.generate(
                        system_prompt="Grounding instructions",
                        user_prompt="Grounded prompt",
                        conversation_history=[],
                    )

    async def test_non_completed_response_status_is_rejected(self) -> None:
        self.client.responses.create.return_value = SimpleNamespace(
            output_text="partial output",
            status="incomplete",
            model="gpt-4.1-mini",
            usage=None,
        )

        with self.assertRaisesRegex(
            LLMProviderError,
            "LLM provider request failed",
        ):
            await self.provider.generate(
                system_prompt="Grounding instructions",
                user_prompt="Grounded prompt",
                conversation_history=[],
            )

    async def test_authentication_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        raw_error = f"Incorrect API key provided: {TEST_API_KEY}"
        response = httpx.Response(
            status_code=401,
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        )
        self.client.responses.create.side_effect = OpenAIAuthenticationError(
            raw_error,
            response=response,
            body={"error": {"message": raw_error}},
        )

        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            with self.assertRaises(LLMConfigurationError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider is not configured.")
        self._assert_no_sensitive_text(caught.exception, logs.output, raw_error)

    async def test_timeout_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        self.client.responses.create.side_effect = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com/v1/responses")
        )

        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            with self.assertRaises(LLMTimeoutError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider timed out.")
        self._assert_no_sensitive_text(caught.exception, logs.output)

    async def test_rate_limit_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        raw_error = f"Quota exceeded for key {TEST_API_KEY}"
        response = httpx.Response(
            status_code=429,
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        )
        self.client.responses.create.side_effect = RateLimitError(
            raw_error,
            response=response,
            body={"error": {"message": raw_error}},
        )

        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            with self.assertRaises(LLMProviderError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider request failed.")
        self._assert_no_sensitive_text(caught.exception, logs.output, raw_error)

    async def test_network_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        raw_error = f"Socket failure while using {TEST_API_KEY}"
        self.client.responses.create.side_effect = APIConnectionError(
            message=raw_error,
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            with self.assertRaises(LLMProviderError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider request failed.")
        self._assert_no_sensitive_text(caught.exception, logs.output, raw_error)

    async def test_api_status_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        raw_error = f"Invalid model for key {TEST_API_KEY}"
        response = httpx.Response(
            status_code=404,
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        )
        self.client.responses.create.side_effect = APIStatusError(
            raw_error,
            response=response,
            body={"error": {"message": raw_error}},
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            with self.assertRaises(LLMProviderError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider request failed.")
        self._assert_no_sensitive_text(caught.exception, logs.output, raw_error)

    async def test_unexpected_error_is_sanitized_and_secret_is_not_logged(
        self,
    ) -> None:
        raw_error = f"Unexpected upstream failure containing {TEST_API_KEY}"
        self.client.responses.create.side_effect = RuntimeError(raw_error)

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            with self.assertRaises(LLMProviderError) as caught:
                await self.provider.generate(
                    system_prompt="Grounding instructions",
                    user_prompt="Grounded prompt",
                    conversation_history=[],
                )

        self.assertEqual(str(caught.exception), "LLM provider request failed.")
        self._assert_no_sensitive_text(caught.exception, logs.output, raw_error)

    def test_constructor_rejects_blank_and_placeholder_keys_without_client(
        self,
    ) -> None:
        invalid_keys = (
            "",
            "   ",
            "your_openai_api_key",
            "YOUR_API_KEY",
            "replace_me",
            "change-me",
            "<openai-api-key>",
        )

        with patch(
            "app.infrastructure.llm.openai_llm_provider.import_module"
        ) as import_sdk:
            for api_key in invalid_keys:
                with self.subTest(api_key=api_key):
                    with self.assertRaisesRegex(
                        LLMConfigurationError,
                        "LLM provider is not configured",
                    ):
                        OpenAILLMProvider(
                            api_key=api_key,
                            model_name="gpt-4.1-mini",
                            temperature=0.1,
                            max_output_tokens=1200,
                            timeout_seconds=30,
                        )

        import_sdk.assert_not_called()

    def _assert_no_sensitive_text(
        self,
        exception: Exception,
        log_output: list[str],
        raw_error: str | None = None,
    ) -> None:
        exposed_text = "\n".join([str(exception), *log_output])
        self.assertNotIn(TEST_API_KEY, exposed_text)
        if raw_error is not None:
            self.assertNotIn(raw_error, exposed_text)


class LLMProviderFactoryTests(unittest.TestCase):
    def test_disabled_provider_needs_no_key_or_openai_construction(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER="DiSaBlEd",
            OPENAI_API_KEY=None,
        )

        with patch("builtins.__import__", wraps=__import__) as importer:
            provider = create_llm_provider(config)

        self.assertIsInstance(provider, NoLLMProvider)
        self.assertFalse(provider.is_configured)
        self._assert_openai_not_imported(importer)

    def test_legacy_none_alias_returns_disabled_provider(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER="NoNe",
            OPENAI_API_KEY=SecretStr(""),
        )

        with patch("builtins.__import__", wraps=__import__) as importer:
            provider = create_llm_provider(config)

        self.assertIsInstance(provider, NoLLMProvider)
        self._assert_openai_not_imported(importer)

    def test_fake_provider_needs_no_key_or_openai_import(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER=" FaKe ",
            OPENAI_API_KEY=None,
            FAKE_LLM_MODEL="fake-test-model",
            CHAT_NO_CONTEXT_MESSAGE="No matching evidence.",
        )

        with patch("builtins.__import__", wraps=__import__) as importer:
            provider = create_llm_provider(config)

        self.assertIsInstance(provider, FakeLLMProvider)
        self.assertEqual(provider.model_name, "fake-test-model")
        self._assert_openai_not_imported(importer)

    def test_openai_provider_is_constructed_from_configuration(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER="OpEnAI",
            OPENAI_API_KEY=SecretStr(TEST_API_KEY),
            LLM_MODEL="gpt-4.1-mini",
            LLM_TEMPERATURE=0.1,
            LLM_MAX_OUTPUT_TOKENS=1200,
            LLM_TIMEOUT_SECONDS=30,
        )
        constructed_provider = Mock()

        with patch(
            "app.infrastructure.llm.openai_llm_provider.OpenAILLMProvider",
            return_value=constructed_provider,
        ) as provider_type:
            provider = create_llm_provider(config)

        self.assertIs(provider, constructed_provider)
        provider_type.assert_called_once_with(
            api_key=TEST_API_KEY,
            model_name="gpt-4.1-mini",
            temperature=0.1,
            max_output_tokens=1200,
            timeout_seconds=30,
        )

    def test_openai_provider_requires_an_api_key(self) -> None:
        config = SimpleNamespace(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY=None,
            LLM_MODEL="gpt-4.1-mini",
            LLM_TEMPERATURE=0.1,
            LLM_MAX_OUTPUT_TOKENS=1200,
            LLM_TIMEOUT_SECONDS=30,
        )

        with self.assertRaisesRegex(
            LLMConfigurationError,
            "LLM provider is not configured",
        ):
            create_llm_provider(config)

    def test_unsupported_provider_raises_configuration_error(self) -> None:
        config = SimpleNamespace(LLM_PROVIDER="local-fallback")

        with self.assertRaisesRegex(
            ConfigurationError,
            "Expected 'disabled', 'fake', or 'openai'",
        ):
            create_llm_provider(config)

    def _assert_openai_not_imported(self, importer: Mock) -> None:
        imported_names = {
            call.args[0]
            for call in importer.call_args_list
            if call.args and isinstance(call.args[0], str)
        }
        self.assertFalse(
            any(
                name == "openai"
                or name.startswith("openai.")
                or name == "app.infrastructure.llm.openai_llm_provider"
                for name in imported_names
            ),
            imported_names,
        )


class LLMSettingsTests(unittest.TestCase):
    def test_fake_defaults_need_no_openai_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = Settings(
                _env_file=None,
                DATABASE_URL="postgresql://user:password@localhost/database",
                JWT_SECRET_KEY="development-secret",
            )

        self.assertEqual(config.LLM_PROVIDER, "fake")
        self.assertEqual(config.LLM_MODEL, "gpt-4.1-mini")
        self.assertEqual(config.OPENAI_CHAT_MODEL, "gpt-4.1-mini")
        self.assertEqual(config.FAKE_LLM_MODEL, "fake-grounded-llm-v1")
        self.assertEqual(config.LLM_TEMPERATURE, 0.0)
        self.assertEqual(config.LLM_MAX_OUTPUT_TOKENS, 800)
        self.assertEqual(config.LLM_TIMEOUT_SECONDS, 45)
        self.assertEqual(config.MAX_CONTEXT_CHARACTERS, 12000)
        self.assertEqual(config.CHAT_HISTORY_MAX_MESSAGES, 10)
        self.assertEqual(config.CHAT_HISTORY_MAX_CHARACTERS, 6000)
        self.assertEqual(config.CHAT_CONTEXT_MAX_CHARACTERS, 12000)
        self.assertEqual(config.CHAT_DEFAULT_TOP_K, 5)
        self.assertEqual(config.CHAT_MAX_TOP_K, 10)
        self.assertEqual(
            config.CHAT_NO_CONTEXT_MESSAGE,
            "I could not find enough information in the selected documents.",
        )
        self.assertIsNone(config.OPENAI_API_KEY)

    def test_provider_is_normalized_and_openai_model_alias_is_supported(
        self,
    ) -> None:
        config = Settings(
            _env_file=None,
            DATABASE_URL="postgresql://user:password@localhost/database",
            JWT_SECRET_KEY="development-secret",
            LLM_PROVIDER=" OpEnAI ",
            OPENAI_CHAT_MODEL=" gpt-test-model ",
        )

        self.assertEqual(config.LLM_PROVIDER, "openai")
        self.assertEqual(config.LLM_MODEL, "gpt-test-model")
        self.assertEqual(config.OPENAI_CHAT_MODEL, "gpt-test-model")


if __name__ == "__main__":
    unittest.main()

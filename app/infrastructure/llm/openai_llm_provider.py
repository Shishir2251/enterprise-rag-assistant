import logging
from collections.abc import Sequence

from openai import APIConnectionError
from openai import APIStatusError
from openai import APITimeoutError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import OpenAI
from openai import OpenAIError
from openai import RateLimitError

from app.business.dtos.llm_dto import LLMMessageDTO, LLMResponseDTO
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMTimeoutError,
)


logger = logging.getLogger(__name__)


class OpenAILLMProvider(ILLMProvider):
    """Generate provider-neutral LLM responses with OpenAI Responses API."""

    _PLACEHOLDER_PREFIXES = (
        "<",
        "change_",
        "change-",
        "replace_",
        "replace-",
        "your_",
        "your-",
    )
    _PLACEHOLDER_VALUES = {
        "changeme",
        "openai_api_key",
        "placeholder",
        "replace_me",
        "sk-placeholder",
        "your_api_key",
        "your_openai_api_key",
    }

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: int,
        client: OpenAI | None = None,
    ) -> None:
        normalized_api_key = api_key.strip() if isinstance(api_key, str) else ""
        normalized_model_name = (
            model_name.strip() if isinstance(model_name, str) else ""
        )
        lowered_api_key = normalized_api_key.lower()

        if (
            not normalized_api_key
            or lowered_api_key in self._PLACEHOLDER_VALUES
            or lowered_api_key.startswith(self._PLACEHOLDER_PREFIXES)
        ):
            raise LLMConfigurationError("LLM provider is not configured.")
        if not normalized_model_name:
            raise LLMConfigurationError("LLM provider is not configured.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= temperature <= 2.0
        ):
            raise LLMConfigurationError("LLM provider is not configured.")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise LLMConfigurationError("LLM provider is not configured.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0
        ):
            raise LLMConfigurationError("LLM provider is not configured.")

        self._model_name = normalized_model_name
        self._temperature = float(temperature)
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._client = (
            client
            if client is not None
            else OpenAI(
                api_key=normalized_api_key,
                timeout=float(timeout_seconds),
            )
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Sequence[LLMMessageDTO],
    ) -> LLMResponseDTO:
        normalized_system_prompt = (
            system_prompt.strip() if isinstance(system_prompt, str) else ""
        )
        normalized_user_prompt = (
            user_prompt.strip() if isinstance(user_prompt, str) else ""
        )
        if not normalized_system_prompt or not normalized_user_prompt:
            raise LLMProviderError("LLM provider request failed.")

        input_messages = self._build_input_messages(
            conversation_history=conversation_history,
            user_prompt=normalized_user_prompt,
        )

        try:
            response = self._client.responses.create(
                model=self._model_name,
                instructions=normalized_system_prompt,
                input=input_messages,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                timeout=float(self._timeout_seconds),
                store=False,
            )
        except OpenAIAuthenticationError as exc:
            logger.warning("OpenAI LLM authentication failed")
            raise LLMConfigurationError(
                "LLM provider is not configured."
            ) from exc
        except APITimeoutError as exc:
            logger.warning("OpenAI LLM request timed out")
            raise LLMTimeoutError("LLM provider timed out.") from exc
        except RateLimitError as exc:
            logger.warning("OpenAI LLM request was rate limited")
            raise LLMProviderError("LLM provider request failed.") from exc
        except (APIConnectionError, APIStatusError, OpenAIError) as exc:
            logger.error(
                "OpenAI LLM request failed (%s)",
                type(exc).__name__,
            )
            raise LLMProviderError("LLM provider request failed.") from exc
        except Exception as exc:
            logger.error(
                "OpenAI LLM request failed (%s)",
                type(exc).__name__,
            )
            raise LLMProviderError("LLM provider request failed.") from exc

        return self._normalize_response(response)

    @staticmethod
    def _build_input_messages(
        *,
        conversation_history: Sequence[LLMMessageDTO],
        user_prompt: str,
    ) -> list[dict[str, str]]:
        input_messages: list[dict[str, str]] = []
        for message in conversation_history:
            if message.role not in {"user", "assistant"}:
                raise LLMProviderError("LLM provider request failed.")
            normalized_content = (
                message.content.strip()
                if isinstance(message.content, str)
                else ""
            )
            if not normalized_content:
                raise LLMProviderError("LLM provider request failed.")
            input_messages.append(
                {
                    "role": message.role,
                    "content": normalized_content,
                }
            )

        input_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return input_messages

    def _normalize_response(self, response: object) -> LLMResponseDTO:
        try:
            output_text = getattr(response, "output_text", None)
            if not isinstance(output_text, str) or not output_text.strip():
                raise LLMProviderError(
                    "LLM provider returned an invalid response."
                )

            response_status = getattr(response, "status", None)
            finish_reason: str | None = None
            if isinstance(response_status, str) and response_status.strip():
                finish_reason = response_status.strip()
                if finish_reason != "completed":
                    raise LLMProviderError(
                        "LLM provider request failed."
                    )

            response_model = getattr(response, "model", None)
            model = (
                response_model.strip()
                if isinstance(response_model, str) and response_model.strip()
                else self._model_name
            )
            usage = getattr(response, "usage", None)

            return LLMResponseDTO(
                content=output_text.strip(),
                provider=self.provider_name,
                model=model,
                input_tokens=self._optional_token_count(
                    getattr(usage, "input_tokens", None)
                ),
                output_tokens=self._optional_token_count(
                    getattr(usage, "output_tokens", None)
                ),
                finish_reason=finish_reason,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.error(
                "OpenAI LLM response normalization failed (%s)",
                type(exc).__name__,
            )
            raise LLMProviderError(
                "LLM provider returned an invalid response."
            ) from exc

    @staticmethod
    def _optional_token_count(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        return None

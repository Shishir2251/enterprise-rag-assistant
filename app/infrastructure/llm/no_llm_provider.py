from collections.abc import Sequence

from app.business.dtos.llm_dto import (
    LLMGenerationResult,
    LLMMessageDTO,
    LLMResponseDTO,
)
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.exceptions import LLMConfigurationError


class NoLLMProvider(ILLMProvider):
    """Disabled provider that never performs a network request."""

    @property
    def provider_name(self) -> str:
        return "disabled"

    @property
    def is_configured(self) -> bool:
        return False

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Sequence[LLMMessageDTO],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponseDTO:
        del (
            system_prompt,
            user_prompt,
            conversation_history,
            max_output_tokens,
            temperature,
        )
        raise LLMConfigurationError("LLM provider is not configured.")

    def generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Sequence[LLMMessageDTO],
    ) -> LLMGenerationResult:
        del query, context, conversation_history
        return LLMGenerationResult(
            status="llm_not_configured",
            answer=None,
        )

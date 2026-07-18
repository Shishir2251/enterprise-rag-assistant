from collections.abc import Sequence

from app.business.dtos.llm_dto import LLMGenerationResult, LLMMessageDTO
from app.business.interfaces.llm_provider_interface import ILLMProvider


class NoLLMProvider(ILLMProvider):
    """Disabled provider that never performs a network request."""

    @property
    def provider_name(self) -> str:
        return "none"

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


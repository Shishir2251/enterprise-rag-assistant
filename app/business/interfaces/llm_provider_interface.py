from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.llm_dto import LLMMessageDTO, LLMResponseDTO


class ILLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Sequence[LLMMessageDTO],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponseDTO:
        raise NotImplementedError

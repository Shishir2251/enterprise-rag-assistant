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
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Sequence[LLMMessageDTO],
    ) -> LLMResponseDTO:
        raise NotImplementedError

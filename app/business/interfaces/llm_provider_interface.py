from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.llm_dto import LLMGenerationResult, LLMMessageDTO


class ILLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Sequence[LLMMessageDTO],
    ) -> LLMGenerationResult:
        raise NotImplementedError


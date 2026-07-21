from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.business.dtos.llm_dto import LLMMessageDTO
from app.business.dtos.prompt_dto import PromptDTO


class IPromptBuilder(ABC):

    @abstractmethod
    def build_grounded_prompt(
        self,
        *,
        query: str,
        context: str,
        conversation_history: Sequence[LLMMessageDTO],
    ) -> PromptDTO:
        raise NotImplementedError

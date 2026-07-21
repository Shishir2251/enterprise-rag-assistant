from dataclasses import dataclass

from app.business.dtos.llm_dto import LLMMessageDTO


@dataclass(frozen=True, slots=True)
class PromptDTO:
    system_prompt: str
    user_prompt: str
    conversation_history: tuple[LLMMessageDTO, ...] = ()

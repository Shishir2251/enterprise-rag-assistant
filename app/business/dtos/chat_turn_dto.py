from dataclasses import dataclass

from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.llm_dto import LLMStatus


@dataclass(frozen=True, slots=True)
class ChatTurnDTO:
    session_id: str
    user_message_id: str
    assistant_message_id: str | None
    status: LLMStatus
    answer: str | None
    citations: tuple[ContextSourceDTO, ...]


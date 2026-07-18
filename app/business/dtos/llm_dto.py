from dataclasses import dataclass
from typing import Literal


LLMStatus = Literal["completed", "llm_not_configured"]


@dataclass(frozen=True, slots=True)
class LLMMessageDTO:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMGenerationResult:
    status: LLMStatus
    answer: str | None


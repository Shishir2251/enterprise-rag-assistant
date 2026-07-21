from dataclasses import dataclass
from typing import Literal


LLMStatus = Literal["completed", "llm_not_configured"]


@dataclass(frozen=True, slots=True)
class LLMMessageDTO:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponseDTO:
    content: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LLMGenerationResult:
    """Legacy orchestration result retained for disabled-mode compatibility."""

    status: LLMStatus
    answer: str | None

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.business.dtos.llm_dto import LLMMessageDTO, LLMResponseDTO
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.exceptions import LLMConfigurationError, LLMProviderError


DEFAULT_FAKE_LLM_MODEL = "fake-grounded-llm-v1"
DEFAULT_NO_CONTEXT_MESSAGE = (
    "I could not find enough information in the selected documents."
)

_SOURCE_START = (
    r"^\[SOURCE\s+[1-9]\d*\][ \t]*\r?\n"
    r"Document:[^\r\n]+\r?\n"
    r"Page:[^\r\n]+\r?\n"
    r"Chunk:[^\r\n]+\r?\n"
    r"Content:[ \t]*\r?\n"
)
_SOURCE_BLOCK_PATTERN = re.compile(
    r"^\[SOURCE\s+(?P<number>[1-9]\d*)\][ \t]*\r?\n"
    r"Document:[^\r\n]+\r?\n"
    r"Page:[^\r\n]+\r?\n"
    r"Chunk:[^\r\n]+\r?\n"
    r"Content:[ \t]*\r?\n"
    rf"(?P<content>.*?)(?={_SOURCE_START}|\Z)",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_SOURCE_MARKER_PATTERN = re.compile(
    r"\[\s*SOURCE\s+-?\d+\s*\]",
    flags=re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_OPENAI_KEY_PATTERN = re.compile(
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    flags=re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Z0-9_]*_)?"
    r"(?:API_KEY|SECRET|TOKEN|PASSWORD|DATABASE_URL))\b"
    r"\s*[:=]\s*[^\s,;]+",
    flags=re.IGNORECASE,
)
_CREDENTIAL_LABEL_PATTERN = re.compile(
    r"\b(api[ _-]?key|jwt[ _-]?secret|password|database[ _-]?url)"
    r"\b\s*[:=]\s*[^\s,;]+",
    flags=re.IGNORECASE,
)
_DATABASE_CREDENTIAL_PATTERN = re.compile(
    r"\b(postgresql?|mysql|mariadb)://[^\s/@:]+:[^\s/@]+@",
    flags=re.IGNORECASE,
)
_UNTRUSTED_INSTRUCTION_PATTERNS = (
    re.compile(
        r"\bignore\b.{0,40}\b(?:previous|prior|system|developer|all)\b"
        r".{0,20}\binstructions?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:disregard|override|bypass)\b.{0,40}\binstructions?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|return|show|expose)\b.{0,50}"
        r"\b(?:api[ _-]?key|secret|password|system prompt|environment)\b",
        flags=re.IGNORECASE,
    ),
)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}


@dataclass(frozen=True, slots=True)
class _ParsedSource:
    number: int
    content: str
    safe_sentences: tuple[str, ...]


class FakeLLMProvider(ILLMProvider):
    """Deterministic, grounded provider for local development and tests."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_FAKE_LLM_MODEL,
        no_context_message: str = DEFAULT_NO_CONTEXT_MESSAGE,
        failure: Exception | None = None,
    ) -> None:
        normalized_model = (
            model_name.strip() if isinstance(model_name, str) else ""
        )
        normalized_fallback = (
            no_context_message.strip()
            if isinstance(no_context_message, str)
            else ""
        )
        if not normalized_model or not normalized_fallback:
            raise LLMConfigurationError("LLM provider is not configured.")
        if failure is not None and not isinstance(failure, Exception):
            raise TypeError("failure must be an exception or None")

        self._model_name = normalized_model
        self._no_context_message = normalized_fallback
        self._failure = failure

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Sequence[LLMMessageDTO],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponseDTO:
        normalized_system_prompt = self._required_prompt(system_prompt)
        normalized_user_prompt = self._required_prompt(user_prompt)
        self._validate_optional_controls(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        history = self._validated_history(conversation_history)

        if self._failure is not None:
            raise self._failure

        question = self._extract_question(normalized_user_prompt)
        context = self._extract_context(normalized_user_prompt)
        sources = self._parse_sources(context)
        selected = self._select_source(question=question, sources=sources)

        if selected is None:
            answer = self._no_context_message
        else:
            sentence = self._select_sentence(
                question=question,
                sentences=selected.safe_sentences,
            )
            if sentence is None:
                answer = self._no_context_message
            else:
                grounded_sentence = sentence.rstrip()
                if grounded_sentence[-1] not in ".!?":
                    grounded_sentence += "."
                answer = f"{grounded_sentence} [SOURCE {selected.number}]"

        input_text = " ".join(
            [
                normalized_system_prompt,
                *(message.content for message in history),
                normalized_user_prompt,
            ]
        )
        return LLMResponseDTO(
            content=answer,
            provider=self.provider_name,
            model=self.model_name,
            input_tokens=self._token_count(input_text),
            output_tokens=self._token_count(answer),
            finish_reason="completed",
        )

    @staticmethod
    def _required_prompt(value: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise LLMProviderError("LLM provider request failed.")
        return normalized

    @staticmethod
    def _validate_optional_controls(
        *,
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> None:
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise LLMProviderError("LLM provider request failed.")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= temperature <= 2.0
        ):
            raise LLMProviderError("LLM provider request failed.")

    @staticmethod
    def _validated_history(
        conversation_history: Sequence[LLMMessageDTO],
    ) -> tuple[LLMMessageDTO, ...]:
        history = tuple(conversation_history)
        for message in history:
            if (
                not isinstance(message, LLMMessageDTO)
                or message.role not in {"user", "assistant"}
                or not isinstance(message.content, str)
                or not message.content.strip()
            ):
                raise LLMProviderError("LLM provider request failed.")
        return history

    @classmethod
    def _extract_question(cls, user_prompt: str) -> str:
        patterns = (
            re.compile(
                r"<current_question>\s*(.*?)\s*</current_question>",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"(?:^|\n)QUESTION:\s*(.*?)\s*\Z",
                flags=re.IGNORECASE | re.DOTALL,
            ),
        )
        for pattern in patterns:
            match = pattern.search(user_prompt)
            if match is not None and match.group(1).strip():
                return match.group(1).strip()
        raise LLMProviderError("LLM provider request failed.")

    @staticmethod
    def _extract_context(user_prompt: str) -> str:
        patterns = (
            re.compile(
                r"<retrieved_context(?:\s+[^>]*)?>\s*(.*?)\s*"
                r"</retrieved_context>",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<BEGIN_DOCUMENT_CONTEXT>\s*(.*?)\s*"
                r"<END_DOCUMENT_CONTEXT>",
                flags=re.IGNORECASE | re.DOTALL,
            ),
        )
        for pattern in patterns:
            match = pattern.search(user_prompt)
            if match is not None:
                return match.group(1).strip()
        return ""

    @classmethod
    def _parse_sources(cls, context: str) -> tuple[_ParsedSource, ...]:
        sources: list[_ParsedSource] = []
        expected_number = 1
        for match in _SOURCE_BLOCK_PATTERN.finditer(context):
            source_number = int(match.group("number"))
            if source_number != expected_number:
                continue
            raw_content = match.group("content").strip()
            safe_sentences = cls._safe_sentences(raw_content)
            sources.append(
                _ParsedSource(
                    number=source_number,
                    content=raw_content,
                    safe_sentences=safe_sentences,
                )
            )
            expected_number += 1
        return tuple(sources)

    @classmethod
    def _select_source(
        cls,
        *,
        question: str,
        sources: Sequence[_ParsedSource],
    ) -> _ParsedSource | None:
        question_terms = cls._terms(question)
        if not question_terms:
            return None

        best_source: _ParsedSource | None = None
        best_score = (0, 0.0)
        for source in sources:
            safe_content = " ".join(source.safe_sentences)
            source_terms = cls._terms(safe_content)
            overlap = question_terms.intersection(source_terms)
            score = (
                len(overlap),
                len(overlap) / len(question_terms),
            )
            if score > best_score:
                best_source = source
                best_score = score
        return best_source

    @classmethod
    def _select_sentence(
        cls,
        *,
        question: str,
        sentences: Sequence[str],
    ) -> str | None:
        question_terms = cls._terms(question)
        best_sentence: str | None = None
        best_score = (0, 0.0)
        for sentence in sentences:
            sentence_terms = cls._terms(sentence)
            overlap = question_terms.intersection(sentence_terms)
            score = (
                len(overlap),
                len(overlap) / max(len(sentence_terms), 1),
            )
            if score > best_score:
                best_sentence = sentence
                best_score = score
        return best_sentence

    @classmethod
    def _safe_sentences(cls, content: str) -> tuple[str, ...]:
        sentences: list[str] = []
        for candidate in _SENTENCE_BOUNDARY_PATTERN.split(content):
            normalized = " ".join(candidate.split())
            if not normalized or cls._looks_like_untrusted_instruction(
                normalized
            ):
                continue
            normalized = cls._redact_secrets(normalized)
            normalized = _SOURCE_MARKER_PATTERN.sub("", normalized)
            normalized = " ".join(normalized.split()).strip()
            if normalized:
                sentences.append(normalized)
        return tuple(sentences)

    @staticmethod
    def _looks_like_untrusted_instruction(text: str) -> bool:
        return any(
            pattern.search(text) is not None
            for pattern in _UNTRUSTED_INSTRUCTION_PATTERNS
        )

    @staticmethod
    def _redact_secrets(text: str) -> str:
        redacted = _OPENAI_KEY_PATTERN.sub("[REDACTED]", text)
        redacted = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
            lambda match: f"{match.group(1)}=[REDACTED]",
            redacted,
        )
        redacted = _CREDENTIAL_LABEL_PATTERN.sub(
            lambda match: f"{match.group(1)}: [REDACTED]",
            redacted,
        )
        return _DATABASE_CREDENTIAL_PATTERN.sub(
            lambda match: f"{match.group(1)}://[REDACTED]@",
            redacted,
        )

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        terms: set[str] = set()
        for raw_token in _TOKEN_PATTERN.findall(text.casefold()):
            if raw_token in _STOP_WORDS:
                continue
            terms.add(raw_token)
            stem = cls._stem(raw_token)
            if stem and stem not in _STOP_WORDS:
                terms.add(stem)
        return terms

    @staticmethod
    def _stem(token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(token) > 4 and token.endswith("ed"):
            stem = token[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    @staticmethod
    def _token_count(text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))

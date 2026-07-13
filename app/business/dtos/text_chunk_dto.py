from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    character_count: int
    page_number: int | None = None
    section_title: str | None = None
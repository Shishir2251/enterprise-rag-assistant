from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int | None
    content: str


@dataclass(frozen=True)
class ExtractedDocument:
    pages: list[ExtractedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            page.content.strip()
            for page in self.pages
            if page.content.strip()
        )
import re
from collections.abc import Sequence

from app.business.dtos.context_source_dto import ContextSourceDTO


class CitationParserService:
    """Map answer markers only to source metadata supplied to the model."""

    SOURCE_MARKER_PATTERN = re.compile(
        r"\[\s*SOURCE\s+([1-9][0-9]*)\s*\]",
        flags=re.IGNORECASE,
    )

    def parse(
        self,
        answer: str,
        supplied_sources: Sequence[ContextSourceDTO],
    ) -> tuple[ContextSourceDTO, ...]:
        sources_by_number = {
            source.source_number: source for source in supplied_sources
        }
        seen: set[int] = set()
        citations: list[ContextSourceDTO] = []

        for match in self.SOURCE_MARKER_PATTERN.finditer(answer):
            source_number = int(match.group(1))
            source = sources_by_number.get(source_number)
            if source is None or source_number in seen:
                continue
            seen.add(source_number)
            citations.append(source)

        return tuple(citations)

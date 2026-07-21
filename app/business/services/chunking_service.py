from app.business.dtos.extracted_text_dto import ExtractedDocument
from app.business.dtos.text_chunk_dto import TextChunk
from app.business.interfaces.chunking_service_interface import (
    IChunkingService,
)


class ChunkingService(IChunkingService):

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(
        self,
        document: ExtractedDocument,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in document.pages:
            page_text = self._normalize_text(page.content)

            if not page_text:
                continue

            start = 0
            text_length = len(page_text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)

                chunk_text = page_text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        TextChunk(
                            index=chunk_index,
                            content=chunk_text,
                            character_count=len(chunk_text),
                            page_number=page.page_number,
                        )
                    )
                    chunk_index += 1

                if end >= text_length:
                    break

                start = end - self.chunk_overlap

        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        # PostgreSQL text values cannot contain the NUL character. Some PDF
        # extractors preserve embedded NULs from malformed font mappings, so
        # remove them at the text-normalization boundary before persistence.
        text = text.replace("\x00", "")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)

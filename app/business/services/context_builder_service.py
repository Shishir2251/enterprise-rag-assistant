from collections.abc import Sequence

from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.interfaces.context_builder_interface import IContextBuilder


class ContextBuilderService(IContextBuilder):

    def __init__(self, max_context_characters: int = 12000) -> None:
        if max_context_characters <= 0:
            raise ValueError("max_context_characters must be greater than zero")
        self.max_context_characters = max_context_characters

    def build_context(
        self,
        retrieval_results: Sequence[RetrievalResult],
    ) -> tuple[str, list[ContextSourceDTO]]:
        sources: list[ContextSourceDTO] = []
        context_blocks: list[str] = []
        current_length = 0

        for result in retrieval_results:
            if not result.content.strip():
                continue

            source = ContextSourceDTO(
                source_number=len(sources) + 1,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_name,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                content=result.content,
                similarity_score=result.similarity_score,
            )
            block = self._format_source(source)
            separator_length = 2 if context_blocks else 0
            next_length = current_length + separator_length + len(block)
            if next_length > self.max_context_characters:
                break

            sources.append(source)
            context_blocks.append(block)
            current_length = next_length

        return "\n\n".join(context_blocks), sources

    @staticmethod
    def _format_source(source: ContextSourceDTO) -> str:
        page_number = (
            str(source.page_number)
            if source.page_number is not None
            else "N/A"
        )
        return (
            f"[SOURCE {source.source_number}]\n"
            f"Document: {source.document_name}\n"
            f"Page: {page_number}\n"
            f"Chunk: {source.chunk_index}\n"
            f"Content:\n{source.content}"
        )

from collections.abc import Sequence

from app.business.dtos.context_source_dto import ContextSourceDTO
from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.interfaces.context_builder_interface import IContextBuilder


class ContextBuilderService(IContextBuilder):

    def build_context(
        self,
        retrieval_results: Sequence[RetrievalResult],
    ) -> tuple[str, list[ContextSourceDTO]]:
        sources = [
            ContextSourceDTO(
                source_number=index,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_name,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                content=result.content,
                similarity_score=result.similarity_score,
            )
            for index, result in enumerate(retrieval_results, start=1)
        ]

        context_blocks = [self._format_source(source) for source in sources]
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

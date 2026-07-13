from abc import ABC, abstractmethod

from app.business.dtos.extracted_text_dto import ExtractedDocument
from app.business.dtos.text_chunk_dto import TextChunk


class IChunkingService(ABC):

    @abstractmethod
    def create_chunks(
        self,
        document: ExtractedDocument,
    ) -> list[TextChunk]:
        raise NotImplementedError
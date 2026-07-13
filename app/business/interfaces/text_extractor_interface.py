from abc import ABC, abstractmethod
from pathlib import Path

from app.business.dtos.extracted_text_dto import ExtractedDocument


class ITextExtractor(ABC):

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        raise NotImplementedError
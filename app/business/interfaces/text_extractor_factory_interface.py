from abc import ABC, abstractmethod
from pathlib import Path

from app.business.interfaces.text_extractor_interface import ITextExtractor


class ITextExtractorFactory(ABC):

    @abstractmethod
    def get_extractor(self, file_path: Path) -> ITextExtractor:
        raise NotImplementedError

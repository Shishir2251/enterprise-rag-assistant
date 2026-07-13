from pathlib import Path

from app.business.interfaces.text_extractor_factory_interface import (
    ITextExtractorFactory,
)
from app.business.interfaces.text_extractor_interface import ITextExtractor
from app.core.exceptions import ValidationError
from app.infrastructure.text_extraction.docx_text_extractor import (
    DocxTextExtractor,
)
from app.infrastructure.text_extraction.pdf_text_extractor import (
    PdfTextExtractor,
)
from app.infrastructure.text_extraction.txt_text_extractor import (
    TxtTextExtractor,
)


class TextExtractorFactory(ITextExtractorFactory):

    def __init__(self) -> None:
        self._extractors: dict[str, ITextExtractor] = {
            ".pdf": PdfTextExtractor(),
            ".docx": DocxTextExtractor(),
            ".txt": TxtTextExtractor(),
        }

    def get_extractor(self, file_path: Path) -> ITextExtractor:
        extension = file_path.suffix.lower()

        extractor = self._extractors.get(extension)

        if extractor is None:
            raise ValidationError(
                f"No text extractor registered for extension: {extension}"
            )

        return extractor

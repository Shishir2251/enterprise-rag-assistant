from pathlib import Path

from app.business.dtos.extracted_text_dto import (
    ExtractedDocument,
    ExtractedPage,
)
from app.business.interfaces.text_extractor_interface import ITextExtractor


class TxtTextExtractor(ITextExtractor):

    def extract(self, file_path: Path) -> ExtractedDocument:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1")

        pages = []

        if text.strip():
            pages.append(
                ExtractedPage(
                    page_number=None,
                    content=text.strip(),
                )
            )

        return ExtractedDocument(pages=pages)
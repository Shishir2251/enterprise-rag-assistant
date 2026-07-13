from pathlib import Path

from pypdf import PdfReader

from app.business.dtos.extracted_text_dto import (
    ExtractedDocument,
    ExtractedPage,
)
from app.business.interfaces.text_extractor_interface import ITextExtractor


class PdfTextExtractor(ITextExtractor):

    def extract(self, file_path: Path) -> ExtractedDocument:
        reader = PdfReader(str(file_path))

        pages: list[ExtractedPage] = []

        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append(
                    ExtractedPage(
                        page_number=index,
                        content=text.strip(),
                    )
                )

        return ExtractedDocument(pages=pages)
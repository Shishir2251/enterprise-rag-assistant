from typing import BinaryIO, Protocol


class IUploadedFile(Protocol):
    """Framework-neutral shape required by the upload use case."""

    filename: str | None
    content_type: str | None
    file: BinaryIO

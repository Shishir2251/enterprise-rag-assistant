import logging
import zipfile
from pathlib import Path

from app.business.interfaces.document_service_interface import IDocumentService
from app.business.interfaces.file_storage_interface import IFileStorage
from app.business.interfaces.uploaded_file_interface import IUploadedFile
from app.core.config import settings
from app.core.exceptions import NotFoundError, PayloadTooLargeError, ValidationError
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentModel


logger = logging.getLogger(__name__)


class DocumentService(IDocumentService):

    ALLOWED_MIME_TYPES = {
        ".pdf": {"application/pdf"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        },
        ".txt": {"text/plain"},
    }

    def __init__(
        self,
        document_repository: IDocumentRepository,
        file_storage: IFileStorage,
    ):
        self.document_repository = document_repository
        self.file_storage = file_storage

    def upload(
        self,
        file: IUploadedFile,
        owner_id: str,
    ) -> DocumentModel:
        original_name, extension, mime_type, file_size = self._validate_upload(
            file
        )

        stored_name, file_path = self.file_storage.save(
            file=file,
            owner_id=owner_id,
            extension=extension,
        )

        document = DocumentModel(
            owner_id=owner_id,
            original_name=original_name,
            stored_name=stored_name,
            file_path=file_path,
            mime_type=mime_type,
            file_size=file_size,
        )

        try:
            return self.document_repository.create(document)

        except Exception:
            self.file_storage.delete(file_path)
            raise

    def list_documents(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        return self.document_repository.list_by_owner(owner_id)

    def get_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel:
        document = self.document_repository.get_by_id(
            document_id=document_id,
            owner_id=owner_id,
        )

        if not document:
            raise NotFoundError("Document not found")

        return document

    def delete_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> None:
        document = self.get_document(document_id, owner_id)

        self.document_repository.delete(document)
        try:
            self.file_storage.delete(document.file_path)
        except Exception:
            logger.exception(
                "Document record deleted but file cleanup failed",
                extra={"document_id": document.id},
            )
            raise

    def _validate_upload(
        self,
        file: IUploadedFile,
    ) -> tuple[str, str, str, int]:
        raw_filename = (file.filename or "").strip()
        if not raw_filename:
            raise ValidationError("Filename is required")

        original_name = Path(raw_filename.replace("\\", "/")).name
        if not original_name or original_name in {".", ".."}:
            raise ValidationError("Filename is invalid")
        if len(original_name) > 255:
            raise ValidationError("Filename must not exceed 255 characters")

        extension = Path(original_name).suffix.lower()
        allowed_mime_types = self.ALLOWED_MIME_TYPES.get(extension)
        if allowed_mime_types is None:
            raise ValidationError("Only PDF, DOCX and TXT files are allowed")

        mime_type = (file.content_type or "").split(";", 1)[0].strip().lower()
        if mime_type not in allowed_mime_types:
            raise ValidationError(
                f"MIME type does not match the {extension} extension"
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size == 0:
            raise ValidationError("Uploaded file must not be empty")
        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_size_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise PayloadTooLargeError(
                f"Maximum file size is {max_size_mb} MB"
            )

        self._validate_file_signature(file, extension)
        return original_name, extension, mime_type, file_size

    @staticmethod
    def _validate_file_signature(
        file: IUploadedFile,
        extension: str,
    ) -> None:
        try:
            file.file.seek(0)
            if extension == ".pdf":
                if file.file.read(5) != b"%PDF-":
                    raise ValidationError("File content is not a valid PDF")
            elif extension == ".docx":
                try:
                    with zipfile.ZipFile(file.file) as archive:
                        names = set(archive.namelist())
                        required = {"[Content_Types].xml", "word/document.xml"}
                        if not required.issubset(names):
                            raise ValidationError(
                                "File content is not a valid DOCX document"
                            )
                except zipfile.BadZipFile as exc:
                    raise ValidationError(
                        "File content is not a valid DOCX document"
                    ) from exc
            else:
                sample = file.file.read(8192)
                if b"\x00" in sample:
                    raise ValidationError("File content is not valid text")
        finally:
            file.file.seek(0)

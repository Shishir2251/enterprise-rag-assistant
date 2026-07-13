from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.business.interfaces.document_service_interface import IDocumentService
from app.business.interfaces.file_storage_interface import IFileStorage
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentModel


class DocumentService(IDocumentService):

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    MAX_FILE_SIZE = 10 * 1024 * 1024

    def __init__(
        self,
        document_repository: IDocumentRepository,
        file_storage: IFileStorage,
    ):
        self.document_repository = document_repository
        self.file_storage = file_storage

    def upload(
        self,
        file: UploadFile,
        owner_id: str,
    ) -> DocumentModel:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        extension = Path(file.filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF, DOCX and TXT files are allowed",
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Maximum file size is 10 MB",
            )

        stored_name, file_path = self.file_storage.save(
            file=file,
            owner_id=owner_id,
        )

        document = DocumentModel(
            owner_id=owner_id,
            original_name=file.filename,
            stored_name=stored_name,
            file_path=file_path,
            mime_type=file.content_type or "application/octet-stream",
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        return document

    def delete_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> None:
        document = self.get_document(document_id, owner_id)

        self.file_storage.delete(document.file_path)
        self.document_repository.delete(document)
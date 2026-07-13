from sqlalchemy.orm import Session
from fastapi import Depends

from app.business.services.auth_service import AuthService
from app.data_access.repositories.user_repository import UserRepository
from app.business.services.document_service import DocumentService
from app.data_access.repositories.document_repository import DocumentRepository
from app.infrastructure.database.session import get_db
from app.infrastructure.file_storage.local_storage_provider import (
    LocalStorageProvider,
)


def get_auth_service(db: Session) -> AuthService:
    user_repository = UserRepository(db)
    return AuthService(user_repository)

def get_document_service(
    db: Session = Depends(get_db),
) -> DocumentService:
    repository = DocumentRepository(db)
    storage = LocalStorageProvider()

    return DocumentService(
        document_repository=repository,
        file_storage=storage,
    )
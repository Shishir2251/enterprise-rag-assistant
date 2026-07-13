from fastapi import Depends
from sqlalchemy.orm import Session

from app.business.interfaces.auth_service_interface import IAuthService
from app.business.interfaces.document_service_interface import IDocumentService
from app.business.interfaces.file_storage_interface import IFileStorage
from app.business.interfaces.ingestion_service_interface import IIngestionService
from app.business.services.auth_service import AuthService
from app.business.services.chunking_service import ChunkingService
from app.business.services.document_service import DocumentService
from app.business.services.ingestion_service import IngestionService
from app.core.config import settings
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.data_access.repositories.document_repository import DocumentRepository
from app.data_access.repositories.user_repository import UserRepository
from app.infrastructure.database.session import get_db
from app.infrastructure.file_storage.local_storage_provider import (
    LocalStorageProvider,
)
from app.infrastructure.text_extraction.text_extractor_factory import (
    TextExtractorFactory,
)


def get_user_repository(
    db: Session = Depends(get_db),
) -> IUserRepository:
    return UserRepository(db)


def get_document_repository(
    db: Session = Depends(get_db),
) -> IDocumentRepository:
    return DocumentRepository(db)


def get_chunk_repository(
    db: Session = Depends(get_db),
) -> IDocumentChunkRepository:
    return DocumentChunkRepository(db)


def get_file_storage() -> IFileStorage:
    return LocalStorageProvider()


def get_auth_service(
    user_repository: IUserRepository = Depends(get_user_repository),
) -> IAuthService:
    return AuthService(user_repository)


def get_document_service(
    repository: IDocumentRepository = Depends(get_document_repository),
    storage: IFileStorage = Depends(get_file_storage),
) -> IDocumentService:
    return DocumentService(
        document_repository=repository,
        file_storage=storage,
    )


def get_ingestion_service(
    document_repository: IDocumentRepository = Depends(
        get_document_repository
    ),
    chunk_repository: IDocumentChunkRepository = Depends(
        get_chunk_repository
    ),
) -> IIngestionService:
    return IngestionService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        chunking_service=ChunkingService(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        ),
        extractor_factory=TextExtractorFactory(),
    )

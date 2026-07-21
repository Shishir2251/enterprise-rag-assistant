from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.business.interfaces.embedding_service_interface import (
    IEmbeddingService,
)
from app.business.interfaces.ingestion_service_interface import (
    IIngestionService,
)
from app.business.services.chunking_service import ChunkingService
from app.business.services.embedding_service import EmbeddingService
from app.business.services.ingestion_service import IngestionService
from app.core.config import settings
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.data_access.repositories.document_repository import DocumentRepository
from app.infrastructure.database.session import SessionLocal
from app.infrastructure.embeddings.embedding_provider_factory import (
    create_embedding_provider,
)
from app.infrastructure.text_extraction.text_extractor_factory import (
    TextExtractorFactory,
)


@dataclass(frozen=True, slots=True)
class DocumentTaskDependencies:
    session: Session
    document_repository: IDocumentRepository
    chunk_repository: IDocumentChunkRepository
    ingestion_service: IIngestionService
    embedding_service: IEmbeddingService


@contextmanager
def get_document_task_dependencies() -> Iterator[DocumentTaskDependencies]:
    session = SessionLocal()
    try:
        document_repository = DocumentRepository(session)
        chunk_repository = DocumentChunkRepository(session)
        embedding_provider = create_embedding_provider(settings)

        yield DocumentTaskDependencies(
            session=session,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_service=IngestionService(
                document_repository=document_repository,
                chunk_repository=chunk_repository,
                chunking_service=ChunkingService(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                ),
                extractor_factory=TextExtractorFactory(),
            ),
            embedding_service=EmbeddingService(
                document_repository=document_repository,
                chunk_repository=chunk_repository,
                embedding_provider=embedding_provider,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
            ),
        )
    finally:
        session.close()


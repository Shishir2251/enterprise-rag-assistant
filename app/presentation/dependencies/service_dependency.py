from fastapi import Depends
from sqlalchemy.orm import Session

from app.business.interfaces.auth_service_interface import IAuthService
from app.business.interfaces.chat_service_interface import IChatService
from app.business.interfaces.context_builder_interface import IContextBuilder
from app.business.interfaces.document_service_interface import IDocumentService
from app.business.interfaces.embedding_service_interface import (
    IEmbeddingService,
)
from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.business.interfaces.file_storage_interface import IFileStorage
from app.business.interfaces.ingestion_service_interface import IIngestionService
from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.business.services.auth_service import AuthService
from app.business.services.chat_service import ChatService
from app.business.services.chunking_service import ChunkingService
from app.business.services.context_builder_service import ContextBuilderService
from app.business.services.document_service import DocumentService
from app.business.services.embedding_service import EmbeddingService
from app.business.services.ingestion_service import IngestionService
from app.business.services.retrieval_service import RetrievalService
from app.core.config import settings
from app.data_access.interfaces.chat_message_repository_interface import (
    IChatMessageRepository,
)
from app.data_access.interfaces.chat_session_repository_interface import (
    IChatSessionRepository,
)
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.interfaces.vector_repository_interface import (
    IVectorRepository,
)
from app.data_access.repositories.chat_message_repository import (
    ChatMessageRepository,
)
from app.data_access.repositories.chat_session_repository import (
    ChatSessionRepository,
)
from app.data_access.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.data_access.repositories.document_repository import DocumentRepository
from app.data_access.repositories.pgvector_repository import PgVectorRepository
from app.data_access.repositories.user_repository import UserRepository
from app.infrastructure.database.session import get_db
from app.infrastructure.embeddings.embedding_provider_factory import (
    create_embedding_provider,
)
from app.infrastructure.file_storage.local_storage_provider import (
    LocalStorageProvider,
)
from app.infrastructure.llm.llm_provider_factory import create_llm_provider
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


def get_vector_repository(
    db: Session = Depends(get_db),
) -> IVectorRepository:
    return PgVectorRepository(db)


def get_chat_session_repository(
    db: Session = Depends(get_db),
) -> IChatSessionRepository:
    return ChatSessionRepository(db)


def get_chat_message_repository(
    db: Session = Depends(get_db),
) -> IChatMessageRepository:
    return ChatMessageRepository(db)


def get_file_storage() -> IFileStorage:
    return LocalStorageProvider()


def get_embedding_provider() -> IEmbeddingProvider:
    return create_embedding_provider(settings)


def get_llm_provider() -> ILLMProvider:
    return create_llm_provider(settings)


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


def get_embedding_service(
    document_repository: IDocumentRepository = Depends(
        get_document_repository
    ),
    chunk_repository: IDocumentChunkRepository = Depends(
        get_chunk_repository
    ),
    provider: IEmbeddingProvider = Depends(get_embedding_provider),
) -> IEmbeddingService:
    return EmbeddingService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        embedding_provider=provider,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


def get_retrieval_service(
    vector_repository: IVectorRepository = Depends(get_vector_repository),
    provider: IEmbeddingProvider = Depends(get_embedding_provider),
) -> IRetrievalService:
    return RetrievalService(
        vector_repository=vector_repository,
        embedding_provider=provider,
        default_top_k=settings.RETRIEVAL_TOP_K,
        minimum_score=settings.RETRIEVAL_MIN_SCORE,
    )


def get_context_builder_service() -> IContextBuilder:
    return ContextBuilderService()


def get_chat_service(
    session_repository: IChatSessionRepository = Depends(
        get_chat_session_repository
    ),
    message_repository: IChatMessageRepository = Depends(
        get_chat_message_repository
    ),
    retrieval_service: IRetrievalService = Depends(get_retrieval_service),
    context_builder: IContextBuilder = Depends(get_context_builder_service),
    llm_provider: ILLMProvider = Depends(get_llm_provider),
) -> IChatService:
    return ChatService(
        session_repository=session_repository,
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        llm_provider=llm_provider,
    )

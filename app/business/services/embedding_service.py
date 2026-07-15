from datetime import datetime, timezone

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.business.interfaces.embedding_service_interface import (
    IEmbeddingService,
)
from app.core.exceptions import ConflictError, EmbeddingError, NotFoundError
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentStatus


class EmbeddingService(IEmbeddingService):

    def __init__(
        self,
        document_repository: IDocumentRepository,
        chunk_repository: IDocumentChunkRepository,
        embedding_provider: IEmbeddingProvider,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero")

        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.embedding_provider = embedding_provider
        self.batch_size = batch_size

    def embed_document(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        document = self.document_repository.get_by_id(
            document_id=document_id,
            owner_id=owner_id,
        )
        if document is None:
            raise NotFoundError("Document not found")
        if document.status != DocumentStatus.COMPLETED.value:
            raise ConflictError(
                "Document processing must complete before embedding"
            )

        chunks = self.chunk_repository.list_without_embeddings(document_id)
        if not chunks:
            return 0

        embedded_count = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self.embedding_provider.embed_texts(
                [chunk.content for chunk in batch]
            )
            self._validate_vectors(vectors, expected_count=len(batch))

            for chunk, vector in zip(batch, vectors, strict=True):
                chunk.embedding = vector

            self.chunk_repository.save_embeddings(
                chunks=batch,
                model_name=self.embedding_provider.model_name,
                embedded_at=datetime.now(timezone.utc),
            )
            embedded_count += len(batch)

        return embedded_count

    def clear_document_embeddings(
        self,
        document_id: str,
        owner_id: str,
    ) -> int:
        document = self.document_repository.get_by_id(
            document_id=document_id,
            owner_id=owner_id,
        )
        if document is None:
            raise NotFoundError("Document not found")

        return self.chunk_repository.clear_embeddings(document_id)

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                "Embedding count does not match the chunk count"
            )

        expected_dimension = self.embedding_provider.dimensions
        if any(len(vector) != expected_dimension for vector in vectors):
            raise EmbeddingError(
                "Embedding vector dimension does not match configuration"
            )

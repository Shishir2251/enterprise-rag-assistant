import logging
from collections.abc import Sequence

from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.core.exceptions import (
    ApplicationError,
    EmbeddingError,
    RetrievalError,
    ValidationError,
)
from app.data_access.interfaces.vector_repository_interface import (
    IVectorRepository,
)


logger = logging.getLogger(__name__)


class RetrievalService(IRetrievalService):

    MAX_TOP_K = 20

    def __init__(
        self,
        vector_repository: IVectorRepository,
        embedding_provider: IEmbeddingProvider,
        default_top_k: int,
        minimum_score: float,
    ) -> None:
        if not 1 <= default_top_k <= self.MAX_TOP_K:
            raise ValueError("Default retrieval top_k must be between 1 and 20")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("Minimum retrieval score must be between 0 and 1")

        self.vector_repository = vector_repository
        self.embedding_provider = embedding_provider
        self.default_top_k = default_top_k
        self.minimum_score = minimum_score

    def search(
        self,
        query: str,
        owner_id: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValidationError("Search query must not be empty")
        if len(normalized_query) < 2 or len(normalized_query) > 2000:
            raise ValidationError(
                "Search query must contain between 2 and 2000 characters"
            )

        result_limit = self.default_top_k if top_k is None else top_k
        if not 1 <= result_limit <= self.MAX_TOP_K:
            raise ValidationError("top_k must be between 1 and 20")

        try:
            query_embedding = self.embedding_provider.embed_query(
                normalized_query
            )
            if len(query_embedding) != self.embedding_provider.dimensions:
                raise EmbeddingError(
                    "Query embedding dimension does not match configuration"
                )

            return self.vector_repository.similarity_search(
                query_embedding=query_embedding,
                owner_id=owner_id,
                top_k=result_limit,
                minimum_score=self.minimum_score,
                document_ids=document_ids,
            )
        except ApplicationError:
            raise
        except Exception as exc:
            logger.exception("Vector retrieval failed")
            raise RetrievalError("Retrieval service is unavailable") from exc

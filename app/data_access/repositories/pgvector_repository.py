from collections.abc import Sequence

from sqlalchemy import literal, select
from sqlalchemy.orm import Session

from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.data_access.interfaces.vector_repository_interface import (
    IVectorRepository,
)
from app.data_access.models.document_chunk_model import DocumentChunkModel
from app.data_access.models.document_model import (
    DocumentModel,
    DocumentStatus,
)


class PgVectorRepository(IVectorRepository):

    def __init__(self, db: Session) -> None:
        self.db = db

    def similarity_search(
        self,
        query_embedding: list[float],
        owner_id: str,
        top_k: int,
        minimum_score: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievalResult]:
        if document_ids is not None and not document_ids:
            return []

        cosine_distance = DocumentChunkModel.embedding.cosine_distance(
            query_embedding
        )
        similarity_score = (
            literal(1.0) - cosine_distance
        ).label("similarity_score")

        statement = (
            select(
                DocumentChunkModel.id.label("chunk_id"),
                DocumentChunkModel.document_id,
                DocumentModel.original_name.label("document_name"),
                DocumentChunkModel.chunk_index,
                DocumentChunkModel.content,
                DocumentChunkModel.page_number,
                similarity_score,
            )
            .join(
                DocumentModel,
                DocumentModel.id == DocumentChunkModel.document_id,
            )
            .where(
                DocumentModel.owner_id == owner_id,
                DocumentModel.status == DocumentStatus.COMPLETED.value,
                DocumentChunkModel.embedding.is_not(None),
                similarity_score >= minimum_score,
            )
            .order_by(cosine_distance.asc())
            .limit(top_k)
        )

        if document_ids is not None:
            statement = statement.where(
                DocumentChunkModel.document_id.in_(list(document_ids))
            )

        rows = self.db.execute(statement).mappings().all()
        return [
            RetrievalResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                page_number=row["page_number"],
                similarity_score=float(row["similarity_score"]),
            )
            for row in rows
        ]

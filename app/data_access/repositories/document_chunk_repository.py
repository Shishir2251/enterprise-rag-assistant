from datetime import datetime

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.models.document_chunk_model import DocumentChunkModel


class DocumentChunkRepository(IDocumentChunkRepository):

    def __init__(self, db: Session):
        self.db = db

    def create_many(
        self,
        chunks: list[DocumentChunkModel],
    ) -> list[DocumentChunkModel]:
        if not chunks:
            return []

        try:
            self.db.add_all(chunks)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        for chunk in chunks:
            self.db.refresh(chunk)

        return chunks

    def list_by_document(
        self,
        document_id: str,
    ) -> list[DocumentChunkModel]:
        statement = (
            select(DocumentChunkModel)
            .where(DocumentChunkModel.document_id == document_id)
            .order_by(DocumentChunkModel.chunk_index.asc())
        )

        return list(self.db.scalars(statement).all())

    def delete_by_document(
        self,
        document_id: str,
    ) -> None:
        statement = delete(DocumentChunkModel).where(
            DocumentChunkModel.document_id == document_id
        )

        try:
            self.db.execute(statement)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def replace_by_document(
        self,
        document_id: str,
        chunks: list[DocumentChunkModel],
    ) -> list[DocumentChunkModel]:
        statement = delete(DocumentChunkModel).where(
            DocumentChunkModel.document_id == document_id
        )

        try:
            self.db.execute(statement)
            self.db.add_all(chunks)
            self.db.commit()

            for chunk in chunks:
                self.db.refresh(chunk)

            return chunks
        except Exception:
            self.db.rollback()
            raise

    def list_without_embeddings(
        self,
        document_id: str,
    ) -> list[DocumentChunkModel]:
        statement = (
            select(DocumentChunkModel)
            .where(
                DocumentChunkModel.document_id == document_id,
                DocumentChunkModel.embedding.is_(None),
            )
            .order_by(DocumentChunkModel.chunk_index.asc())
        )
        return list(self.db.scalars(statement).all())

    def list_stale_embeddings(
        self,
        document_id: str,
        model_name: str,
        provider_name: str,
    ) -> list[DocumentChunkModel]:
        statement = (
            select(DocumentChunkModel)
            .where(
                DocumentChunkModel.document_id == document_id,
                or_(
                    DocumentChunkModel.embedding.is_(None),
                    DocumentChunkModel.embedding_model.is_(None),
                    DocumentChunkModel.embedding_model != model_name,
                    DocumentChunkModel.embedding_provider.is_(None),
                    DocumentChunkModel.embedding_provider != provider_name,
                ),
            )
            .order_by(DocumentChunkModel.chunk_index.asc())
        )
        return list(self.db.scalars(statement).all())

    def list_stale_document_ids(
        self,
        model_name: str,
        provider_name: str,
    ) -> list[str]:
        statement = (
            select(DocumentChunkModel.document_id)
            .where(
                or_(
                    DocumentChunkModel.embedding.is_(None),
                    DocumentChunkModel.embedding_model.is_(None),
                    DocumentChunkModel.embedding_model != model_name,
                    DocumentChunkModel.embedding_provider.is_(None),
                    DocumentChunkModel.embedding_provider != provider_name,
                )
            )
            .distinct()
            .order_by(DocumentChunkModel.document_id.asc())
        )
        return list(self.db.scalars(statement).all())

    def save_embeddings(
        self,
        chunks: list[DocumentChunkModel],
        model_name: str,
        provider_name: str,
        embedded_at: datetime,
    ) -> None:
        if not chunks:
            return

        try:
            for chunk in chunks:
                chunk.embedding_model = model_name
                chunk.embedding_provider = provider_name
                chunk.embedded_at = embedded_at
            self.db.add_all(chunks)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def clear_embeddings(
        self,
        document_id: str,
    ) -> int:
        statement = (
            update(DocumentChunkModel)
            .where(
                DocumentChunkModel.document_id == document_id,
                or_(
                    DocumentChunkModel.embedding.is_not(None),
                    DocumentChunkModel.embedding_model.is_not(None),
                    DocumentChunkModel.embedding_provider.is_not(None),
                    DocumentChunkModel.embedded_at.is_not(None),
                ),
            )
            .values(
                embedding=None,
                embedding_model=None,
                embedding_provider=None,
                embedded_at=None,
            )
            .execution_options(synchronize_session=False)
        )

        try:
            result = self.db.execute(statement)
            self.db.commit()
            return int(result.rowcount or 0)
        except Exception:
            self.db.rollback()
            raise

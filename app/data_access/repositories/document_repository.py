from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentModel


class DocumentRepository(IDocumentRepository):

    def __init__(self, db: Session):
        self.db = db

    def create(self, document: DocumentModel) -> DocumentModel:
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_by_id(
        self,
        document_id: str,
        owner_id: str,
    ) -> DocumentModel | None:
        statement = select(DocumentModel).where(
            DocumentModel.id == document_id,
            DocumentModel.owner_id == owner_id,
        )

        return self.db.scalar(statement)

    def list_by_owner(
        self,
        owner_id: str,
    ) -> list[DocumentModel]:
        statement = (
            select(DocumentModel)
            .where(DocumentModel.owner_id == owner_id)
            .order_by(DocumentModel.created_at.desc())
        )

        return list(self.db.scalars(statement).all())

    def delete(self, document: DocumentModel) -> None:
        self.db.delete(document)
        self.db.commit()
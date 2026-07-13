from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.models.user_model import UserModel


class UserRepository(IUserRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> UserModel | None:
        return self.db.scalar(
            select(UserModel).where(UserModel.email == email)
        )

    def get_by_id(self, user_id: str) -> UserModel | None:
        return self.db.scalar(
            select(UserModel).where(UserModel.id == user_id)
        )

    def create(self, user: UserModel) -> UserModel:
        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except IntegrityError as exc:
            self.db.rollback()
            raise ConflictError("Email already registered") from exc
        except Exception:
            self.db.rollback()
            raise

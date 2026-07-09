from sqlalchemy.orm import Session

from app.business.services.auth_service import AuthService
from app.data_access.repositories.user_repository import UserRepository


def get_auth_service(db: Session) -> AuthService:
    user_repository = UserRepository(db)
    return AuthService(user_repository)
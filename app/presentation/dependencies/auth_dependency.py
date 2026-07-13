from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import decode_access_token
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.service_dependency import get_user_repository


security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    user_repository: IUserRepository = Depends(get_user_repository),
) -> UserModel:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Authentication credentials are required")

    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise AuthenticationError("Invalid or expired token") from exc

    user = user_repository.get_by_id(user_id)
    if user is None:
        raise AuthenticationError("Invalid or expired token")
    if not user.is_active:
        raise AuthorizationError("Inactive user")

    return user

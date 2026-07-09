from fastapi import HTTPException, status

from app.business.interfaces.auth_service_interface import IAuthService
from app.core.security import hash_password, verify_password, create_access_token
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.models.user_model import UserModel
from app.presentation.schemas.auth_schema import RegisterRequest, LoginRequest, AuthResponse, UserResponse


class AuthService(IAuthService):
    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository

    def register(self, payload: RegisterRequest) -> UserResponse:
        existing_user = self.user_repository.get_by_email(payload.email)

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        user = UserModel(
            full_name=payload.full_name,
            email=payload.email,
            hashed_password=hash_password(payload.password),
        )

        created_user = self.user_repository.create(user)

        return UserResponse.model_validate(created_user)

    def login(self, payload: LoginRequest) -> AuthResponse:
        user = self.user_repository.get_by_email(payload.email)

        if not user or not verify_password(payload.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        access_token = create_access_token(subject=user.id)

        return AuthResponse(access_token=access_token)
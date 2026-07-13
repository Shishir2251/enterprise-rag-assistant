from app.business.interfaces.auth_service_interface import IAuthService
from app.core.exceptions import AuthenticationError, ConflictError, ValidationError
from app.core.security import create_access_token, hash_password, verify_password
from app.data_access.interfaces.user_repository_interface import IUserRepository
from app.data_access.models.user_model import UserModel


class AuthService(IAuthService):
    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository

    def register(
        self,
        full_name: str,
        email: str,
        password: str,
    ) -> UserModel:
        normalized_email = email.strip().lower()
        if len(password.encode("utf-8")) > 72:
            raise ValidationError("Password must not exceed 72 bytes")

        existing_user = self.user_repository.get_by_email(normalized_email)

        if existing_user:
            raise ConflictError("Email already registered")

        user = UserModel(
            full_name=full_name.strip(),
            email=normalized_email,
            hashed_password=hash_password(password),
        )

        return self.user_repository.create(user)

    def login(self, email: str, password: str) -> str:
        user = self.user_repository.get_by_email(email.strip().lower())

        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("Invalid email or password")

        return create_access_token(subject=user.id)

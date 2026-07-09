from abc import ABC, abstractmethod

from app.presentation.schemas.auth_schema import RegisterRequest, LoginRequest, AuthResponse, UserResponse


class IAuthService(ABC):

    @abstractmethod
    def register(self, payload: RegisterRequest) -> UserResponse:
        pass

    @abstractmethod
    def login(self, payload: LoginRequest) -> AuthResponse:
        pass
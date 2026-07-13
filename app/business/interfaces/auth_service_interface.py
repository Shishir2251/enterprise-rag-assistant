from abc import ABC, abstractmethod

from app.data_access.models.user_model import UserModel


class IAuthService(ABC):

    @abstractmethod
    def register(
        self,
        full_name: str,
        email: str,
        password: str,
    ) -> UserModel:
        raise NotImplementedError

    @abstractmethod
    def login(self, email: str, password: str) -> str:
        raise NotImplementedError

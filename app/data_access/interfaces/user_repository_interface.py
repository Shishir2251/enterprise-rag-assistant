from abc import ABC, abstractmethod
from app.data_access.models.user_model import UserModel


class IUserRepository(ABC):

    @abstractmethod
    def get_by_email(self, email: str) -> UserModel | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, user_id: str) -> UserModel | None:
        raise NotImplementedError

    @abstractmethod
    def create(self, user: UserModel) -> UserModel:
        raise NotImplementedError

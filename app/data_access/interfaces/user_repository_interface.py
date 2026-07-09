from abc import ABC, abstractmethod
from typing import Optional

from app.data_access.models.user_model import UserModel


class IUserRepository(ABC):

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[UserModel]:
        pass

    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[UserModel]:
        pass

    @abstractmethod
    def create(self, user: UserModel) -> UserModel:
        pass
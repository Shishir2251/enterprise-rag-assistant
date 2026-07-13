from abc import ABC, abstractmethod

from fastapi import UploadFile


class IFileStorage(ABC):

    @abstractmethod
    def save(
        self,
        file: UploadFile,
        owner_id: str,
    ) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, file_path: str) -> None:
        raise NotImplementedError
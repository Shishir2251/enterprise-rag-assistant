from abc import ABC, abstractmethod

from app.business.interfaces.uploaded_file_interface import IUploadedFile


class IFileStorage(ABC):

    @abstractmethod
    def save(
        self,
        file: IUploadedFile,
        owner_id: str,
        extension: str,
    ) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, file_path: str) -> None:
        raise NotImplementedError

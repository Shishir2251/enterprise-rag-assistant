import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.business.interfaces.file_storage_interface import IFileStorage
from app.core.config import settings


class LocalStorageProvider(IFileStorage):

    def __init__(self) -> None:
        self.upload_root = Path(settings.UPLOAD_DIR)

    def save(
        self,
        file: UploadFile,
        owner_id: str,
    ) -> tuple[str, str]:
        user_directory = self.upload_root / owner_id
        user_directory.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "").suffix.lower()
        stored_name = f"{uuid.uuid4()}{suffix}"
        file_path = user_directory / stored_name

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return stored_name, str(file_path)

    def delete(self, file_path: str) -> None:
        path = Path(file_path)

        if path.exists() and path.is_file():
            path.unlink()
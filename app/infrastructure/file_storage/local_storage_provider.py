import shutil
import uuid
from pathlib import Path

from app.business.interfaces.file_storage_interface import IFileStorage
from app.business.interfaces.uploaded_file_interface import IUploadedFile
from app.core.config import settings


class LocalStorageProvider(IFileStorage):

    ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}

    def __init__(self) -> None:
        self.upload_root = Path(settings.UPLOAD_DIR).resolve()

    def save(
        self,
        file: IUploadedFile,
        owner_id: str,
        extension: str,
    ) -> tuple[str, str]:
        user_directory = self._resolve_inside_root(self.upload_root / owner_id)
        user_directory.mkdir(parents=True, exist_ok=True)

        suffix = extension.lower()
        if suffix not in self.ALLOWED_SUFFIXES:
            raise ValueError("Unsupported storage file extension")

        stored_name = f"{uuid.uuid4()}{suffix}"
        file_path = user_directory / stored_name

        try:
            with file_path.open("xb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise

        return stored_name, str(file_path)

    def delete(self, file_path: str) -> None:
        path = self._resolve_inside_root(Path(file_path))

        if path.exists() and path.is_file():
            path.unlink()

    def _resolve_inside_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.upload_root):
            raise ValueError("Storage path is outside the upload directory")
        return resolved

"""File storage behind a small interface. Local disk by default; S3 can be added later.

Files are keyed by drive id: a stored file's key looks like "<drive_id>/video.mov".
Writes are streamed in chunks so multi-GB uploads never sit in memory.
"""
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from backend import config

CHUNK_SIZE = 1024 * 1024  # 1 MiB


class Storage(ABC):
    """Where uploaded files live. Keys are opaque strings returned by save()."""

    @abstractmethod
    def save(self, drive_id: str, name: str, source: BinaryIO) -> str:
        """Stream `source` (a binary file object) into storage as `name` for a drive; return the key."""

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Open a stored file for reading (binary)."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove one stored file. Missing keys are ignored."""

    @abstractmethod
    def delete_drive(self, drive_id: str) -> None:
        """Remove everything stored for a drive. Missing drives are ignored."""

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A local filesystem path for tools like ffmpeg. Remote backends should
        download to a temp file here."""


class LocalDiskStorage(Storage):
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root if root is not None else config.storage_dir())

    def _resolve(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        return path

    def save(self, drive_id: str, name: str, source: BinaryIO) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", drive_id):
            raise ValueError(f"invalid drive id: {drive_id!r}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name.startswith("."):
            raise ValueError(f"invalid file name: {name!r}")
        key = f"{drive_id}/{name}"
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            shutil.copyfileobj(source, out, CHUNK_SIZE)
        return key

    def open(self, key: str) -> BinaryIO:
        return open(self._resolve(key), "rb")

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def delete_drive(self, drive_id: str) -> None:
        shutil.rmtree(self._resolve(drive_id), ignore_errors=True)

    def local_path(self, key: str) -> Path:
        return self._resolve(key)


def get_storage() -> Storage:
    """The configured storage backend (FastAPI dependency; override in tests)."""
    return LocalDiskStorage()

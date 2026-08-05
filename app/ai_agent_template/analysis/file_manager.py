"""File registry for run_python's downloadable outputs (charts/data files).

Trimmed from the reference's `file_manager.py`: `detect_file_type`/
`validate_file_size`/`validate_file_type` only supported the Data Analysis
REST tab's file-upload validation, which this template doesn't have — the
chat `run_python` tool never accepts an uploaded file, only produces output
files. Also fixed a portability bug on the way over: the reference hardcodes a
POSIX-only `/tmp/...` storage path; this project's own working conventions
are Windows/PowerShell, so this uses `tempfile.gettempdir()` instead.
"""

from __future__ import annotations

import logging
import mimetypes
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_agent_template.analysis.types import DownloadFile

logger = logging.getLogger(__name__)


class FileRegistry:
    """Registry for downloadable files with automatic cleanup."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "ai_agent_template_analysis"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, "DownloadFile"] = {}
        self._load_existing_files()
        logger.info("FileRegistry initialized: %s", self.storage_dir)

    def _load_existing_files(self) -> None:
        """Load metadata for existing files from storage directory.

        This allows the registry to work across restarts by reading file IDs
        from the filesystem. Metadata like original filename is lost, but the
        files remain downloadable by their file_id.
        """
        from ai_agent_template.analysis.types import DownloadFile

        for file_path in self.storage_dir.glob("*"):
            if file_path.is_file():
                try:
                    file_id = file_path.name
                    stat = file_path.stat()
                    self._files[file_id] = DownloadFile(
                        file_id=file_id,
                        filename=file_id,  # original filename unknown, use file_id
                        content=b"",  # lazy load on get_file()
                        mime_type="application/octet-stream",
                        created_at=stat.st_mtime,
                    )
                except Exception as e:
                    logger.warning("Failed to load file %s: %s", file_path, e)

    def register_file(self, filename: str, content: bytes, mime_type: str | None = None) -> str:
        """Register a file for download and return its file_id."""
        from ai_agent_template.analysis.types import DownloadFile

        file_id = self._generate_file_id()
        if not mime_type:
            mime_type = self._detect_mime_type(filename)

        file_path = self.storage_dir / file_id
        file_path.write_bytes(content)

        self._files[file_id] = DownloadFile(
            file_id=file_id,
            filename=filename,
            content=content,
            mime_type=mime_type,
            created_at=time.time(),
        )

        logger.info("Registered file: %s -> %s (%d bytes)", filename, file_id, len(content))
        return file_id

    def get_file(self, file_id: str) -> "DownloadFile | None":
        """Retrieve a file by its file_id."""
        file = self._files.get(file_id)
        if not file:
            logger.warning("File not found: %s", file_id)
            return None

        # Read from disk (in case in-memory was cleared, e.g. after restart)
        file_path = self.storage_dir / file_id
        if file_path.exists():
            file.content = file_path.read_bytes()
        return file

    def cleanup_old_files(self, max_age_seconds: int = 3600) -> int:
        """Remove files older than max_age_seconds. Returns count deleted."""
        now = time.time()
        to_delete = [
            file_id
            for file_id, file in self._files.items()
            if (now - file.created_at) > max_age_seconds
        ]

        deleted = 0
        for file_id in to_delete:
            try:
                file_path = self.storage_dir / file_id
                if file_path.exists():
                    file_path.unlink()
                del self._files[file_id]
                deleted += 1
            except Exception as e:
                logger.error("Failed to delete file %s: %s", file_id, e)

        if deleted:
            logger.info("Cleaned up %d old files", deleted)
        return deleted

    @staticmethod
    def _generate_file_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _detect_mime_type(filename: str) -> str:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"


# Global registry instance
_file_registry: FileRegistry | None = None


def get_file_registry() -> FileRegistry:
    """Get or create the global file registry."""
    global _file_registry
    if _file_registry is None:
        _file_registry = FileRegistry()
    return _file_registry

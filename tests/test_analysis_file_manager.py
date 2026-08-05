"""Tests for FileRegistry — real filesystem (tmp_path), no AWS calls."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ai_agent_template.analysis.file_manager import FileRegistry


def test_register_and_get_file(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    file_id = registry.register_file("chart.plotly.json", b'{"data": []}')

    file = registry.get_file(file_id)
    assert file is not None
    assert file.filename == "chart.plotly.json"
    assert file.content == b'{"data": []}'
    assert file.mime_type == "application/json"


def test_get_unknown_file_returns_none(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    assert registry.get_file("does-not-exist") is None


def test_registry_survives_reload_from_disk(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    file_id = registry.register_file("data.csv", b"a,b\n1,2\n")

    # Simulate a process restart — a fresh registry over the same directory.
    reloaded = FileRegistry(storage_dir=tmp_path)
    file = reloaded.get_file(file_id)
    assert file is not None
    assert file.content == b"a,b\n1,2\n"


def test_cleanup_old_files_removes_expired_only(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    old_id = registry.register_file("old.txt", b"old")
    new_id = registry.register_file("new.txt", b"new")

    # Backdate the "old" file's registered timestamp past the cutoff.
    registry._files[old_id].created_at = time.time() - 7200

    deleted = registry.cleanup_old_files(max_age_seconds=3600)
    assert deleted == 1
    assert registry.get_file(old_id) is None
    assert registry.get_file(new_id) is not None


def test_default_storage_dir_is_portable_not_hardcoded_unix_path():
    registry = FileRegistry()
    # Must not hardcode a POSIX-only /tmp path — this project's own working
    # conventions are Windows/PowerShell, and tempfile.gettempdir() resolves
    # correctly on both.
    assert registry.storage_dir.exists()
    assert str(registry.storage_dir).startswith(str(Path(tempfile.gettempdir())))

"""Tests for CodeExecutionService — fake backend dict injected directly, no
boto3/AWS calls. Focus: the self-contained start->execute->stop lifecycle
(the whole reason `sandbox_cache` reuse wasn't ported) always stops the
sandbox, even on error or timeout."""

from __future__ import annotations

import asyncio

import pytest

from ai_agent_template.analysis.code_execution_service import (
    CodeExecutionService,
    get_code_execution_service,
)
from ai_agent_template.config import CodeExecutionConfig


def _service(backend: dict, **overrides) -> CodeExecutionService:
    return CodeExecutionService(
        interpreter_id="fake-interpreter",
        region="us-east-1",
        exec_timeout_seconds=overrides.get("exec_timeout_seconds", 90),
        sandbox_idle_timeout_seconds=300,
        backend=backend,
    )


def _fake_backend(*, execute_code=None, download_file=None):
    stop_calls: list[str] = []

    async def start_code_interpreter_session(**kwargs) -> dict:
        return {"session_id": "sess-1"}

    async def stop_code_interpreter_session(**kwargs) -> dict:
        stop_calls.append(kwargs["session_id"])
        return {}

    async def default_execute_code(**kwargs) -> dict:
        return {"output": "42", "files_created": [], "error": None}

    async def default_download_file(**kwargs) -> dict:
        return {"content": b"bytes-for-" + kwargs["file_path"].encode()}

    backend = {
        "start_code_interpreter_session": start_code_interpreter_session,
        "stop_code_interpreter_session": stop_code_interpreter_session,
        "execute_code": execute_code or default_execute_code,
        "download_file": download_file or default_download_file,
    }
    return backend, stop_calls


async def test_run_returns_output_and_stops_sandbox():
    backend, stop_calls = _fake_backend()
    outcome = await _service(backend).run("print(42)")
    assert outcome.ok
    assert outcome.output == "42"
    assert stop_calls == ["sess-1"]


async def test_run_downloads_produced_files():
    async def execute_code(**kwargs) -> dict:
        return {"output": "", "files_created": ["chart.plotly.json"], "error": None}

    backend, stop_calls = _fake_backend(execute_code=execute_code)
    outcome = await _service(backend).run("...")
    assert outcome.ok
    assert outcome.files == [("chart.plotly.json", b"bytes-for-chart.plotly.json")]
    assert stop_calls == ["sess-1"]


async def test_run_execution_error_still_stops_sandbox():
    async def execute_code(**kwargs) -> dict:
        return {"output": "", "files_created": [], "error": "NameError: x undefined"}

    backend, stop_calls = _fake_backend(execute_code=execute_code)
    outcome = await _service(backend).run("bad code")
    assert not outcome.ok
    assert "NameError" in outcome.error
    assert stop_calls == ["sess-1"]


async def test_run_timeout_returns_clean_error_and_stops_sandbox():
    async def slow_execute_code(**kwargs) -> dict:
        await asyncio.sleep(1)
        return {"output": "", "files_created": [], "error": None}

    backend, stop_calls = _fake_backend(execute_code=slow_execute_code)
    outcome = await _service(backend, exec_timeout_seconds=0.05).run("while True: pass")
    assert not outcome.ok
    assert "time" in outcome.error.lower()
    assert stop_calls == ["sess-1"]


async def test_run_one_bad_download_does_not_abort_the_rest():
    calls = {"n": 0}

    async def execute_code(**kwargs) -> dict:
        return {"output": "", "files_created": ["bad.txt", "good.txt"], "error": None}

    async def download_file(**kwargs) -> dict:
        calls["n"] += 1
        if kwargs["file_path"] == "bad.txt":
            raise RuntimeError("download failed")
        return {"content": b"good-bytes"}

    backend, _ = _fake_backend(execute_code=execute_code, download_file=download_file)
    outcome = await _service(backend).run("...")
    assert outcome.ok
    assert outcome.files == [("good.txt", b"good-bytes")]


async def test_get_code_execution_service_returns_none_when_unconfigured():
    assert get_code_execution_service(CodeExecutionConfig(), bedrock_region="us-east-1") is None

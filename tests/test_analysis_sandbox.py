"""Structural tests for CodeSandbox — fake backend dict, no boto3/AWS calls."""

from __future__ import annotations

import pytest

from ai_agent_template.analysis.sandbox import CodeSandbox


def _fake_backend(**overrides):
    async def start_code_interpreter_session(**kwargs) -> dict:
        return {"session_id": "sess-1"}

    async def stop_code_interpreter_session(**kwargs) -> dict:
        return {}

    async def execute_code(**kwargs) -> dict:
        return {"output": "hello", "files_created": [], "error": None}

    async def download_file(**kwargs) -> dict:
        return {"content": b"file-bytes"}

    backend = {
        "start_code_interpreter_session": start_code_interpreter_session,
        "stop_code_interpreter_session": stop_code_interpreter_session,
        "execute_code": execute_code,
        "download_file": download_file,
    }
    backend.update(overrides)
    return backend


async def test_missing_backend_op_rejected():
    with pytest.raises(ValueError, match="Missing required AgentCore backend operations"):
        CodeSandbox({"start_code_interpreter_session": None})


async def test_start_sets_sandbox_id():
    sandbox = CodeSandbox(_fake_backend())
    sid = await sandbox.start()
    assert sid == "sess-1"
    assert sandbox.sandbox_id == "sess-1"


async def test_stop_clears_sandbox_id():
    sandbox = CodeSandbox(_fake_backend())
    await sandbox.start()
    await sandbox.stop()
    assert sandbox.sandbox_id is None


async def test_execute_code_without_start_raises():
    sandbox = CodeSandbox(_fake_backend())
    with pytest.raises(RuntimeError, match="No active session"):
        await sandbox.execute_code("print(1)")


async def test_execute_code_returns_result():
    sandbox = CodeSandbox(_fake_backend())
    await sandbox.start()
    result = await sandbox.execute_code("print('hello')")
    assert result.output == "hello"
    assert result.error is None


async def test_execute_code_backend_exception_becomes_error_result():
    async def failing_execute(**kwargs):
        raise RuntimeError("boom")

    sandbox = CodeSandbox(_fake_backend(execute_code=failing_execute))
    await sandbox.start()
    result = await sandbox.execute_code("print(1)")
    assert result.error is not None
    assert "boom" in result.error


async def test_download_file_returns_bytes():
    sandbox = CodeSandbox(_fake_backend())
    await sandbox.start()
    content = await sandbox.download_file("chart.plotly.json")
    assert content == b"file-bytes"


async def test_download_file_without_start_raises():
    sandbox = CodeSandbox(_fake_backend())
    with pytest.raises(RuntimeError, match="No active session"):
        await sandbox.download_file("x")


async def test_context_manager_starts_and_stops():
    backend = _fake_backend()
    async with CodeSandbox(backend) as sandbox:
        assert sandbox.sandbox_id == "sess-1"
    assert sandbox.sandbox_id is None

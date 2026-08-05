"""Boto3-backed `backend` dict consumed by CodeSandbox.

Calls the AWS Bedrock AgentCore Code Interpreter API directly via boto3 —
NOT MCP. Trimmed to the four operations the chat `run_python` tool actually
exercises (`start`/`stop`/`execute_code`/`download_file`); the reference's
`upload_file`/`execute_command`/file-inspection support only the Data
Analysis REST tab (file uploads), which this template doesn't have.

Sandbox filesystem notes (carried over from the reference's own probe):
  - Python cwd inside the sandbox is a fixed working directory
  - readFiles/listFiles operate on that same working directory
  - Absolute paths (e.g. /mnt/data/) are rejected with a "path traversal" error
  - Use relative paths (bare filename, no leading /) for all file operations —
    Python code should reference produced files as just the filename

`identifier`/`region` are explicit params, not read from `os.environ` here —
this project centralizes every env read in config.py's load_config();
the caller (code_execution_service.py) threads CodeExecutionConfig through.
No region default: an unset region is the caller's problem to surface, not
this module's to paper over with a real deployment's region.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)

# Files present in a fresh session (never treated as "created by code execution")
_SANDBOX_BASELINE = frozenset({
    "log", "node_modules", "nodejs-js-execution", "nodejs-ts-execution",
    "package-lock.json", "package.json", "run", ".ipython",
})


def _make_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)


# ── Stream / event collectors ─────────────────────────────────────────────────

def _collect_stdout_stderr(stream) -> tuple[str, str, int]:
    """Drain an InvokeCodeInterpreter EventStream. Return (stdout, stderr, exit_code)."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code = 0

    for event in stream:
        result = event.get("result")
        if not result:
            continue
        sc = result.get("structuredContent") or {}
        if sc.get("stdout"):
            stdout_parts.append(sc["stdout"])
        if sc.get("stderr"):
            stderr_parts.append(sc["stderr"])
        if sc.get("exitCode") is not None:
            exit_code = sc["exitCode"]

    return "".join(stdout_parts), "".join(stderr_parts), exit_code


def _collect_bytes(stream) -> bytes:
    """Drain a readFiles EventStream. Return file content as bytes.

    readFiles returns blocks of type 'resource' with a nested 'resource' dict:
        {'type': 'resource', 'resource': {'uri': 'file:///name', 'mimeType': ..., 'text': ...}}
    Binary files use a 'blob'/'data' field; text files use 'text'.
    """
    for event in stream:
        result = event.get("result")
        if not result:
            continue
        for block in result.get("content", []):
            # Top-level binary data
            if block.get("data"):
                raw = block["data"]
                return raw if isinstance(raw, bytes) else bytes(raw)
            # Nested resource dict
            resource = block.get("resource") or {}
            if resource.get("blob"):
                raw = resource["blob"]
                return raw if isinstance(raw, bytes) else bytes(raw)
            if resource.get("data"):
                raw = resource["data"]
                return raw if isinstance(raw, bytes) else bytes(raw)
            if resource.get("text"):
                return resource["text"].encode()
            # Top-level text fallback
            if block.get("text"):
                return block["text"].encode()
    return b""


def _collect_file_names(stream) -> list[str]:
    """Drain a listFiles EventStream. Return bare filenames (relative, no leading /).

    listFiles returns 'resource_link' blocks:
        {'type': 'resource_link', 'uri': 'file:///name', 'name': 'name', 'description': 'File'}
    """
    names: list[str] = []
    for event in stream:
        result = event.get("result")
        if not result or result.get("isError"):
            continue
        for block in result.get("content", []):
            name = block.get("name", "")
            desc = block.get("description", "")
            # Only include regular files, not directories
            if name and desc == "File":
                names.append(name)
    return names


# ── Sync helpers (run inside asyncio.to_thread) ───────────────────────────────

def _sync_start(client, identifier: str, timeout_seconds: int) -> dict:
    resp = client.start_code_interpreter_session(
        codeInterpreterIdentifier=identifier,
        name="aiAgentTemplateAnalysis",
        sessionTimeoutSeconds=timeout_seconds,
    )
    return {"session_id": resp["sessionId"]}


def _sync_stop(client, identifier: str, session_id: str) -> dict:
    client.stop_code_interpreter_session(
        codeInterpreterIdentifier=identifier,
        sessionId=session_id,
    )
    return {}


def _sync_list_files(client, identifier: str, session_id: str) -> list[str]:
    """Return bare filenames of all regular files in the sandbox cwd."""
    try:
        resp = client.invoke_code_interpreter(
            codeInterpreterIdentifier=identifier,
            sessionId=session_id,
            name="listFiles",
            arguments={"directoryPath": ""},
        )
        return _collect_file_names(resp["stream"])
    except Exception as exc:
        logger.warning("listFiles failed: %s", exc)
        return []


def _sync_execute_code(client, identifier: str, session_id: str, code: str) -> dict:
    # Snapshot existing files so we can diff after execution
    before = set(_sync_list_files(client, identifier, session_id)) | _SANDBOX_BASELINE

    exec_resp = client.invoke_code_interpreter(
        codeInterpreterIdentifier=identifier,
        sessionId=session_id,
        name="executeCode",
        arguments={"code": code, "language": "python"},
    )
    stdout, stderr, exit_code = _collect_stdout_stderr(exec_resp["stream"])

    # Detect files created by the code (new files not present before)
    after = _sync_list_files(client, identifier, session_id)
    files_created = [f for f in after if f not in before]

    return {
        "output": stdout,
        "files_created": files_created,
        "error": stderr if exit_code != 0 else None,
    }


def _sync_download(client, identifier: str, session_id: str, file_path: str) -> dict:
    # file_path is a bare filename (returned in files_created)
    resp = client.invoke_code_interpreter(
        codeInterpreterIdentifier=identifier,
        sessionId=session_id,
        name="readFiles",
        arguments={"paths": [file_path]},
    )
    content = _collect_bytes(resp["stream"])
    return {"content": content}


# ── Public factory ────────────────────────────────────────────────────────────

def build_agentcore_backend(identifier: str, region: str) -> dict[str, Any]:
    """Return a `backend` dict of async callables backed by direct boto3 AgentCore calls.

    Matches the interface expected by CodeSandbox:
      - start_code_interpreter_session(timeout_seconds=...) → {"session_id": ...}
      - stop_code_interpreter_session(session_id=...)       → {}
      - execute_code(session_id, code)                      → {"output", "files_created", "error"}
      - download_file(session_id, file_path)                → {"content": bytes}

    Note: file_path values are bare filenames. Python code executed in the
    sandbox must reference produced files the same way.
    """

    async def start_code_interpreter_session(**kwargs) -> dict:
        timeout = int(kwargs.get("timeout_seconds", 1800))
        client = _make_client(region)
        return await asyncio.to_thread(_sync_start, client, identifier, timeout)

    async def stop_code_interpreter_session(**kwargs) -> dict:
        session_id = kwargs["session_id"]
        client = _make_client(region)
        return await asyncio.to_thread(_sync_stop, client, identifier, session_id)

    async def execute_code(**kwargs) -> dict:
        session_id = kwargs["session_id"]
        code = kwargs["code"]
        client = _make_client(region)
        return await asyncio.to_thread(_sync_execute_code, client, identifier, session_id, code)

    async def download_file(**kwargs) -> dict:
        session_id = kwargs["session_id"]
        file_path = kwargs["file_path"]
        client = _make_client(region)
        return await asyncio.to_thread(_sync_download, client, identifier, session_id, file_path)

    return {
        "start_code_interpreter_session": start_code_interpreter_session,
        "stop_code_interpreter_session": stop_code_interpreter_session,
        "execute_code": execute_code,
        "download_file": download_file,
    }

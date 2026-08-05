"""Code sandbox (AgentCore code interpreter) over a boto3-backed backend.

Trimmed from the reference's `CodeSandbox`: `upload_file`/`inspect_file`/
`execute_command` only support the Data Analysis REST tab (file uploads +
pandas-based CSV/Excel/JSON inspection), which this template doesn't have —
the chat `run_python` tool only ever starts a sandbox, executes code, and
downloads any produced files.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_agent_template.analysis.types import CodeExecutionResult

logger = logging.getLogger(__name__)


class CodeSandbox:
    """Manages an AgentCore code sandbox over a boto3-backed backend.

    Wraps a dict of AgentCore operation callables (NOT MCP — see
    agentcore_code_client) to provide a clean interface for starting/stopping
    a session, executing Python code, and downloading generated files.
    """

    def __init__(self, backend: dict[str, Any]) -> None:
        """Initialize with the AgentCore backend callables.

        Args:
            backend: Dictionary mapping AgentCore operation names to async
                callables. Required: start_code_interpreter_session,
                stop_code_interpreter_session, execute_code, download_file.
        """
        self.backend = backend
        self.sandbox_id: str | None = None
        self._validate_tools()

    def _validate_tools(self) -> None:
        required = {
            "start_code_interpreter_session",
            "stop_code_interpreter_session",
            "execute_code",
            "download_file",
        }
        missing = required - set(self.backend.keys())
        if missing:
            raise ValueError(f"Missing required AgentCore backend operations: {missing}")

    async def start(self, timeout_seconds: int = 600) -> str:
        """Start a new code interpreter session.

        Args:
            timeout_seconds: Sandbox idle timeout in seconds.

        Returns:
            Session ID

        Raises:
            RuntimeError: If session start fails
        """
        if self.sandbox_id:
            logger.warning("Session already started: %s", self.sandbox_id)
            return self.sandbox_id

        try:
            result = await self.backend["start_code_interpreter_session"](
                timeout_seconds=timeout_seconds
            )
            self.sandbox_id = result.get("session_id")
            if not self.sandbox_id:
                raise RuntimeError("No session_id returned from start_code_interpreter_session")
            logger.info("Started code interpreter session: %s", self.sandbox_id)
            return self.sandbox_id
        except Exception as e:
            logger.error("Failed to start code interpreter session: %s", e)
            raise RuntimeError(f"Session start failed: {e}") from e

    async def stop(self) -> None:
        """Stop the current session and clean up resources."""
        if not self.sandbox_id:
            logger.warning("No active session to stop")
            return

        try:
            await self.backend["stop_code_interpreter_session"](session_id=self.sandbox_id)
            logger.info("Stopped code interpreter session: %s", self.sandbox_id)
        except Exception as e:
            logger.error("Failed to stop session %s: %s", self.sandbox_id, e)
        finally:
            self.sandbox_id = None

    async def execute_code(self, code: str) -> CodeExecutionResult:
        """Execute Python code in the sandbox.

        Never raises for ordinary execution errors — they come back on
        `CodeExecutionResult.error` so callers can branch on it.
        """
        from ai_agent_template.analysis.types import CodeExecutionResult

        if not self.sandbox_id:
            raise RuntimeError("No active session")

        try:
            result = await self.backend["execute_code"](session_id=self.sandbox_id, code=code)
            return CodeExecutionResult(
                output=result.get("output", ""),
                files_created=result.get("files_created", []),
                error=result.get("error"),
            )
        except Exception as e:
            logger.error("Failed to execute code: %s", e)
            return CodeExecutionResult(output="", error=str(e))

    async def download_file(self, file_path: str) -> bytes:
        """Download a file from the sandbox.

        Args:
            file_path: Bare filename in the sandbox (from files_created).

        Raises:
            RuntimeError: If download fails
        """
        if not self.sandbox_id:
            raise RuntimeError("No active session")

        try:
            result = await self.backend["download_file"](
                session_id=self.sandbox_id, file_path=file_path
            )
            content = result.get("content")
            if content is None:
                raise RuntimeError(f"No content returned for {file_path}")
            return content if isinstance(content, bytes) else content.encode()
        except Exception as e:
            logger.error("Failed to download file %s: %s", file_path, e)
            raise RuntimeError(f"Download failed: {e}") from e

    async def __aenter__(self) -> CodeSandbox:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

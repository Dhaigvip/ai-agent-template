"""CodeExecutionService — the server-owned code-execution engine for `run_python`.

Composes the boto3 pieces that talk to AgentCore's Code Interpreter:

  • boto3 transport ....... agentcore_code_client.build_agentcore_backend()
  • sandbox lifecycle ..... sandbox.CodeSandbox (start/execute/download/stop)

The LLM never sees start/stop — the server owns the sandbox. Trimmed from the
reference: no `sandbox_cache` reuse-across-calls layer. The reference kept
one because it also served a Data Analysis REST tab where follow-up queries
benefit from reusing the same sandbox (skip re-uploading files); this
template has no file-upload feature and `code_exec_tool.py`'s own docstring
already commits to "one complete analysis per call" — so `run()` here does
the whole start -> execute -> download -> stop lifecycle in one call, with
no `sandbox_id` handed back to reuse. Simpler, and matches what CLAUDE.md
asks for this task: "self-contained per call... leak-free by construction."

`execute()` wraps the run in a wall-clock timeout
(`CodeExecutionConfig.exec_timeout_seconds`); on timeout the sandbox is still
stopped by `run()`'s `finally`, so a runaway can't keep billing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ai_agent_template.analysis.agentcore_code_client import build_agentcore_backend
from ai_agent_template.analysis.sandbox import CodeSandbox
from ai_agent_template.analysis.types import ExecutionOutcome
from ai_agent_template.config import CodeExecutionConfig

logger = logging.getLogger(__name__)


class CodeExecutionService:
    """Server-owned code execution: start a sandbox, run code, stop it.

    Stateless apart from its cached boto3 backend dict, built lazily and
    reused across calls (each call still gets a fresh `CodeSandbox`/session —
    only the boto3-callable dict itself is cheap to keep around).
    """

    def __init__(
        self,
        interpreter_id: str,
        region: str,
        exec_timeout_seconds: int,
        sandbox_idle_timeout_seconds: int,
        backend: dict[str, Any] | None = None,
    ) -> None:
        self._interpreter_id = interpreter_id
        self._region = region
        self._exec_timeout_seconds = exec_timeout_seconds
        self._sandbox_idle_timeout_seconds = sandbox_idle_timeout_seconds
        self._backend = backend

    def _get_backend(self) -> dict[str, Any]:
        if self._backend is None:
            self._backend = build_agentcore_backend(self._interpreter_id, self._region)
        return self._backend

    async def run(self, code: str) -> ExecutionOutcome:
        """Start a fresh sandbox, execute `code`, stop the sandbox. Never
        raises — every failure path (start, execute, timeout, stop) comes
        back as `ExecutionOutcome.error` so callers branch on `.ok`."""
        sandbox = CodeSandbox(self._get_backend())
        try:
            await sandbox.start(timeout_seconds=self._sandbox_idle_timeout_seconds)
        except Exception as err:
            return ExecutionOutcome(output="", error=str(err))

        try:
            return await self._execute(sandbox, code)
        finally:
            # Self-contained lifecycle: stop the sandbox immediately so it
            # never lingers/bills, regardless of how execution went.
            await sandbox.stop()

    async def _execute(self, sandbox: CodeSandbox, code: str) -> ExecutionOutcome:
        try:
            result = await asyncio.wait_for(
                sandbox.execute_code(code), timeout=self._exec_timeout_seconds
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "execute: exceeded %ds wall-clock — sandbox %s will be stopped",
                self._exec_timeout_seconds,
                sandbox.sandbox_id,
            )
            return ExecutionOutcome(
                output="",
                error=(
                    f"Execution exceeded the {self._exec_timeout_seconds}s time "
                    "limit and was stopped. Simplify the analysis or reduce the data size."
                ),
            )

        files: list[tuple[str, bytes]] = []
        if not result.error and result.files_created:
            files = await self._download(sandbox, result.files_created)

        return ExecutionOutcome(
            output=result.output,
            error=result.error,
            files=files,
            files_created=result.files_created,
        )

    async def _download(
        self, sandbox: CodeSandbox, names: list[str]
    ) -> list[tuple[str, bytes]]:
        """Pull each produced file's bytes out of the sandbox. One bad file is
        logged and skipped so it never aborts the rest."""
        out: list[tuple[str, bytes]] = []
        for name in names:
            try:
                content = await sandbox.download_file(name)
                out.append((name.split("/")[-1], content))  # label with bare filename
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to download sandbox file %s: %s", name, e)
        return out


# ── Shared singleton ──────────────────────────────────────────────────────────

_service: Optional["CodeExecutionService"] = None


def get_code_execution_service(
    config: CodeExecutionConfig, *, bedrock_region: str
) -> CodeExecutionService | None:
    """Return the process-wide CodeExecutionService, or None if
    AGENTCORE_CODE_INTERPRETER_ID unset.

    Same id-presence gating pattern as `knowledge_base.kb_client.get_kb_client()`.
    Callers must guard:
        service = get_code_execution_service(config.code_execution, bedrock_region=config.bedrock.region)
        if service:
            tools.append(code_exec_tool(service))
    """
    global _service

    if not config.interpreter_id:
        return None

    if _service is None:
        region = config.region or bedrock_region
        _service = CodeExecutionService(
            interpreter_id=config.interpreter_id,
            region=region,
            exec_timeout_seconds=config.exec_timeout_seconds,
            sandbox_idle_timeout_seconds=config.sandbox_idle_timeout_seconds,
        )
        logger.info(
            "CodeExecutionService initialized: interpreter_id=%s region=%s",
            config.interpreter_id,
            region,
        )

    return _service

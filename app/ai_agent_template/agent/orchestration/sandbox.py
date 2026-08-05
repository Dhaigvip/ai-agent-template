"""RestrictedPython exec harness for the code-orchestration tool.

Ported concept from LangChain deepagents' Interpreter middleware
(docs.langchain.com/oss/*/deepagents/interpreters) — NOT the library itself.
This module knows nothing about MCP, this project's tools, or LangChain — it
only knows how to run restricted Python source against a `tools` object
supplied by the caller (tool_bridge.py).

RestrictedPython's `compile_restricted` rejects `async def`/`await` outright,
so user code is plain sync module-level statements executed in a worker
thread; `tools.xxx()` calls block that worker thread only (via
run_coroutine_threadsafe under the hood, in tool_bridge.py) while the main
event loop keeps running.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from RestrictedPython import PrintCollector, compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence

logger = logging.getLogger(__name__)

# Match config.py's OrchestrationConfig defaults — kept here too since this
# module is also exercised directly in tests without going through config.
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RESULT_CHARS = 4000


@dataclass
class SandboxOutcome:
    """Result of one sandboxed exec. Never raises out of run_sandboxed — all
    failure modes (compile error, runtime exception, timeout, non-serializable
    result) are represented here so the calling tool can hand a clean message
    back to the model instead of crashing the turn."""

    ok: bool
    result_text: str = ""
    stdout: str = ""
    error: str | None = None


def _build_restricted_globals(tools: Any) -> dict:
    """Base RestrictedPython safe_globals plus the guard hooks it requires
    for subscript access (`d["k"]`), iteration, unpacking, and `print()`
    (RestrictedPython rewrites `print(...)` to go through `_print_`, a
    factory — it does NOT write to real stdout, so contextlib.redirect_stdout
    would capture nothing; see PrintCollector below), plus the two names
    user code is allowed to see: `json` and `tools`."""
    g = dict(safe_globals)
    g["_getiter_"] = default_guarded_getiter
    g["_getitem_"] = default_guarded_getitem
    g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    g["_print_"] = PrintCollector
    g["json"] = json
    g["tools"] = tools
    return g


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"


def _exec_sandboxed(code: str, tools: Any, max_result_chars: int) -> SandboxOutcome:
    """Synchronous by design — runs inside asyncio.to_thread (see run_sandboxed).

    Convention: user code assigns a `result` variable; that becomes the tool
    output (JSON-serialized — strictly: dict/list/str/int/float/bool/None
    only, no permissive stringify-fallback, so a malformed result fails loud
    with a clear message rather than silently embedding a Python repr).
    Anything printed via `print(...)` is captured separately (RestrictedPython
    routes print through its own `_print_` collector, not real stdout).
    There is no bare "last expression" support.
    """
    try:
        byte_code = compile_restricted(code, filename="<orchestration>", mode="exec")
    except SyntaxError as err:
        return SandboxOutcome(ok=False, error=f"Code did not compile: {err}")

    restricted_globals = _build_restricted_globals(tools)
    local_vars: dict = {}
    try:
        exec(byte_code, restricted_globals, local_vars)
    except Exception as err:  # noqa: BLE001 — surfaced to the model, never crashes the agent
        printed = local_vars["_print"]() if "_print" in local_vars else ""
        return SandboxOutcome(
            ok=False,
            stdout=_truncate(printed, max_result_chars),
            error=f"{type(err).__name__}: {err}",
        )

    printed = local_vars["_print"]() if "_print" in local_vars else ""
    stdout = _truncate(printed, max_result_chars)

    if "result" not in local_vars:
        return SandboxOutcome(
            ok=False,
            stdout=stdout,
            error="Code finished without setting a `result` variable — assign the value you want returned to `result`.",
        )

    try:
        result_text = json.dumps(local_vars["result"])
    except TypeError as err:
        return SandboxOutcome(
            ok=False,
            stdout=stdout,
            error=(
                f"`result` is not JSON-serializable ({err}). Only dict/list/str/"
                "int/float/bool/None are allowed — convert sets/tuples to lists first."
            ),
        )

    return SandboxOutcome(ok=True, result_text=_truncate(result_text, max_result_chars), stdout=stdout)


async def run_sandboxed(
    code: str,
    tools: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> SandboxOutcome:
    """Execute `code` in a RestrictedPython sandbox on a worker thread.

    `tools` is any object exposing plain (sync) callables — see
    tool_bridge.ToolBridge for the real bound-tool-backed implementation, or
    a fake object in tests. Never raises: every failure path returns
    SandboxOutcome(ok=False, ...).

    `timeout`/`max_result_chars` are explicit params rather than env reads —
    this project centralizes every env read in config.py's load_config();
    the caller (orchestration_tool.py) threads OrchestrationConfig through.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_exec_sandboxed, code, tools, max_result_chars),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("run_sandboxed: execution exceeded %.1fs timeout", timeout)
        return SandboxOutcome(
            ok=False,
            error=(
                f"Execution exceeded the {timeout:.0f}s timeout. Narrow the loop "
                "(fewer items per call) or split the work across multiple calls."
            ),
        )

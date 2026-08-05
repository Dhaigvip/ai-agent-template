"""Capability bridge for sandboxed `tools.xxx()` calls.

Exposes sync-looking callables to the RestrictedPython worker thread while
executing real tool calls on the main asyncio loop.

WARNING — APPROVAL IS NOT ENFORCED HERE THE WAY IT IS FOR NORMAL TOOL CALLS.
This bridge invokes tools DIRECTLY; calls made from inside a sandboxed script
never pass through the model's tool-call path, so `HumanInTheLoopMiddleware`
cannot see them. Left unrestricted, that would make the bridge a genuine
HITL bypass: a script could wrap a mutation and it would execute immediately,
with no human decision, even though the exact same call made directly by the
model would have been intercepted and paused.

This is a real difference from the reference this tool was ported from. There,
`default_policy` allowed ordinary mutations through ungated, because that
production system stages every mutation into a workspace that isn't published
until a human calls a separate commit action — so an unattended mutation
inside the sandbox never actually took effect without a human eventually
approving the publish. This project's demo domain has no staging/publish
split: `risk_classifier.py` requires human approval on every mutation
directly. Porting the looser policy as-is would silently drop that guarantee.

So `default_policy` here is deliberately tighter than the reference:
- hard-deny blocked tools (`risk_classifier.is_blocked_tool`)
- deny anything `risk_classifier.classify_risk` marks as requiring approval
  (i.e. every non-`get_*` tool) — call it directly as a normal tool call
  instead, where HITL can actually see it
- allow read-only (`get_*`) tools through the bridge

That's a real capability reduction versus the reference (no reshape-then-
mutate inside one sandboxed call). If a looser policy is wanted later — e.g.
a one-time per-session "pre-approve mutations for this orchestration run"
grant — that's a follow-up feature, not a default.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from ai_agent_template.agent.middleware.risk_classifier import classify_risk, is_blocked_tool

logger = logging.getLogger(__name__)

DEFAULT_MAX_CALLS = 256
DEFAULT_CALL_TIMEOUT_SECONDS = 15.0

Policy = Callable[[str, dict, "dict[str, Any]"], Awaitable[Any]]
"""(tool_name, args, tools_by_name) -> real tool result (already ainvoke()'d)."""


class OrchestrationPolicyError(Exception):
    """Raised by the bridge when a call is rejected by policy or exceeds the
    call budget. Propagates as a plain exception out of sandboxed `exec()`,
    caught by sandbox._exec_sandboxed's generic handler and turned into a
    SandboxOutcome(ok=False, error=...) the model can read and react to."""


async def default_policy(name: str, args: dict, tools_by_name: dict[str, Any]) -> Any:
    """Per-call gate inside the sandbox. See module docstring for why this is
    tighter than a simple allow-all: it is the only thing standing between a
    sandboxed script and an ungated mutation, since HITL cannot see calls
    made through this bridge."""
    if is_blocked_tool(name):
        raise OrchestrationPolicyError(
            f"'{name}' is hard-blocked — it is never available to you, in or out of this tool."
        )
    if classify_risk(name).requires_approval:
        raise OrchestrationPolicyError(
            f"'{name}' changes data and requires human approval — call it "
            "directly as a normal tool call (outside this sandbox), not from "
            "inside the orchestration sandbox. Only read-only (get_*) tools "
            "are reachable from run_orchestration."
        )
    tool = tools_by_name.get(name)
    if tool is None:
        raise OrchestrationPolicyError(f"'{name}' is not a bound tool.")
    return await tool.ainvoke(args)


def _parse_tool_result(raw: Any) -> Any:
    """Real tool calls (see tool_helpers.mcp_tool_to_langchain) return a
    JSON-text STRING (to_llm_tool_content), not a parsed object. Parse it back
    to a native dict/list here so sandboxed code can index into it directly
    (`mods["tasks"]`) instead of every script re-parsing JSON itself. Falls
    back to the raw value if it isn't valid JSON (e.g. a plain-text tool)."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw


class _CallBudget:
    """Per-run call counter — the real backstop against a runaway loop calling
    tools indefinitely (the sandbox's overall exec timeout bounds wall-clock
    time, not call count)."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.count = 0

    def increment(self, name: str) -> None:
        self.count += 1
        if self.count > self.max_calls:
            raise OrchestrationPolicyError(
                f"Exceeded {self.max_calls} tool calls in one orchestration run "
                f"(hit the limit calling '{name}'). Narrow the loop or split the "
                "work across multiple calls."
            )


class ToolBridge:
    """Sync-looking facade exposed to sandboxed code as the `tools` global.

    Each bound tool name resolves (via __getattr__) to a plain sync callable.
    Calling it blocks the CALLING THREAD — expected to be a RestrictedPython
    worker thread (see sandbox.run_sandboxed's asyncio.to_thread) — until the
    real async tool call resolves on `main_loop`, via
    asyncio.run_coroutine_threadsafe. The tool coroutine itself always runs on
    the main loop; it never runs on the sandbox thread.
    """

    def __init__(
        self,
        tools_by_name: dict[str, Any],
        main_loop: asyncio.AbstractEventLoop,
        *,
        policy: Policy = default_policy,
        max_calls: int = DEFAULT_MAX_CALLS,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        self._tools_by_name = tools_by_name
        self._loop = main_loop
        self._policy = policy
        self._budget = _CallBudget(max_calls)
        self._call_timeout = call_timeout
        self.call_log: list[str] = []  # observability — names called, in order

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # __getattr__ only fires for names not already set in __init__ (the
        # underscore-prefixed instance attributes above), so this never
        # shadows real state.
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._tools_by_name:
            raise AttributeError(
                f"'{name}' is not a bound tool. Available: {sorted(self._tools_by_name)}"
            )

        def _call(**kwargs: Any) -> Any:
            self._budget.increment(name)
            self.call_log.append(name)
            coro = self._policy(name, kwargs, self._tools_by_name)
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            raw = fut.result(timeout=self._call_timeout)
            return _parse_tool_result(raw)

        return _call


def build_tool_bridge(
    tools_by_name: dict[str, Any],
    main_loop: asyncio.AbstractEventLoop,
    **kwargs: Any,
) -> ToolBridge:
    """Build a fresh bridge instance for one orchestration run."""
    return ToolBridge(tools_by_name, main_loop, **kwargs)

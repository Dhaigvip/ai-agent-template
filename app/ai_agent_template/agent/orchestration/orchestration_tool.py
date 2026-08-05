"""Native `run_orchestration` agent tool — sandboxed code-loop over bound tools.

A port of LangChain deepagents' Interpreter *concept* (eval tool + PTC bridge)
as a plain always-bound LangChain tool — NOT the deepagents library, whose
interpreter is JS/QuickJS-only and requires create_deep_agent.

Distinct from `run_python` (analysis/code_exec_tool.py, Task 16): that tool
is a remote AgentCore sandbox for data analysis/charts/math and cannot call
this project's tools at all. This tool is a local, sub-second sandbox whose
entire purpose is calling already-bound tools in a loop.

Gating: `FeatureFlags.orchestration_enabled` is the sole gate — unlike
run_python, this tool has no external AWS resource to additionally check.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import tool

from ai_agent_template.agent.orchestration.sandbox import run_sandboxed
from ai_agent_template.agent.orchestration.tool_bridge import build_tool_bridge
from ai_agent_template.config import OrchestrationConfig

logger = logging.getLogger(__name__)


def orchestration_tool(tools_by_name: dict[str, Any], config: OrchestrationConfig):
    """Return the `run_orchestration` LangChain tool.

    `tools_by_name` is the tool registry assembled in ai_gateway._ensure_graph_built
    (all MCP-sourced tools + always-bound synthetic tools) — captured by
    reference, not copied, so it reflects the graph's final registry at call
    time. Safety does not depend on narrowing this dict: tool_bridge.default_policy
    rejects blocked/mutation tools regardless of what's present here —
    narrowing it would be redundant defense, not the actual gate.
    """

    @tool
    async def run_orchestration(code: str) -> str:
        """The last resort of the MULTI-CALL TASKS ladder in your instructions
        — use it only when the cheaper options there are ruled out.

        What rules them out: calls that DEPEND ON EACH OTHER. You cannot know
        the next call until you have seen the previous result — so you would
        otherwise spend a round trip per step. Typical shapes:
          - chain    : feed one call's output into the next
          - branch   : call A, then call B or C depending on what came back
          - reshape  : turn a read result into a filter/argument for another call
          - loop     : repeat a call per item, with logic between iterations
        Never wrap a single call in it, and never use it for calls that do
        not depend on each other — those belong in one reply, in parallel.

        NOT the same as run_python: this tool has no pandas/numpy/plotly, no
        charts, no math backend. It exists to run a few DEPENDENT tool calls
        in one round trip. For data analysis, statistics, or charts, use
        run_python instead.

        HOW: write plain Python. Call any already-available READ-ONLY tool as
        tools.<tool_name>(**kwargs) — e.g. tools.get_tasks(filter={"active": True}).
        Each call returns the tool's result already parsed as JSON (dict/list)
        when possible. Assign the value you want returned to a `result`
        variable — it must be JSON-native (dict/list/str/int/float/bool/None).

        Example — reshape + branch, two dependent reads, one round trip.
        Note there is no loop over tool calls: each tool is called once, and
        it is the DEPENDENCY between them that needs this tool.
            projects = tools.get_projects(filter={"active": True})
            stale = [p["uid"] for p in projects["projects"] if not p.get("milestoneUid")]
            tasks = tools.get_tasks(filter={"projectUids": stale}) if stale else {"tasks": []}
            result = {"stale_project_count": len(stale), "open_tasks": len(tasks["tasks"])}

        LIMITS: no imports, no file/network access. Only READ-ONLY tools
        (get_*) are reachable from inside this sandbox — mutations
        (create_*/update_*/...) are refused here even though the model can
        call them directly outside this tool, because calls made from inside
        a script bypass human approval entirely. If your plan needs a
        mutation, gather the data you need with run_orchestration first, then
        call the mutation tool directly as a normal, human-reviewable call.
        Runs for a few seconds at most; narrow the loop if it times out.
        """
        main_loop = asyncio.get_running_loop()
        bridge = build_tool_bridge(
            tools_by_name,
            main_loop,
            max_calls=config.max_calls,
            call_timeout=config.call_timeout_seconds,
        )
        outcome = await run_sandboxed(
            code,
            bridge,
            timeout=config.timeout_seconds,
            max_result_chars=config.max_result_chars,
        )

        logger.info(
            "run_orchestration: ok=%s bridged_calls=%d tools=%s",
            outcome.ok, len(bridge.call_log), bridge.call_log,
        )

        if not outcome.ok:
            return f"Orchestration failed: {outcome.error}"

        parts = []
        if outcome.stdout.strip():
            parts.append(outcome.stdout.strip())
        parts.append(outcome.result_text)
        return "\n".join(parts)

    return run_orchestration

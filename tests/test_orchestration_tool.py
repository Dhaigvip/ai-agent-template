"""End-to-end (no AWS, no MCP) tests for the run_orchestration LangChain tool."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from ai_agent_template.agent.orchestration.orchestration_tool import orchestration_tool
from ai_agent_template.config import OrchestrationConfig


async def _fake_get_tasks() -> str:
    return '{"tasks": [{"uid": "t1", "title": "one"}, {"uid": "t2", "title": "two"}]}'


async def _fake_update_task(uid: str) -> str:
    return '{"ok": true}'


def _tools_by_name() -> dict:
    return {
        "get_tasks": StructuredTool.from_function(
            coroutine=_fake_get_tasks, name="get_tasks", description="read"
        ),
        "update_task": StructuredTool.from_function(
            coroutine=_fake_update_task, name="update_task", description="mutate"
        ),
    }


async def test_run_orchestration_executes_dependent_reads():
    run_orchestration = orchestration_tool(_tools_by_name(), OrchestrationConfig())
    code = (
        "tasks = tools.get_tasks()\n"
        "result = {'count': len(tasks['tasks'])}\n"
    )
    output = await run_orchestration.ainvoke({"code": code})
    assert '"count": 2' in output


async def test_run_orchestration_reports_failure_without_raising():
    run_orchestration = orchestration_tool(_tools_by_name(), OrchestrationConfig())
    output = await run_orchestration.ainvoke({"code": "result = 1 / 0"})
    assert output.startswith("Orchestration failed:")
    assert "ZeroDivisionError" in output


async def test_run_orchestration_blocks_mutation_even_with_correct_args():
    run_orchestration = orchestration_tool(_tools_by_name(), OrchestrationConfig())
    code = "result = tools.update_task(uid='t1')\n"
    output = await run_orchestration.ainvoke({"code": code})
    assert output.startswith("Orchestration failed:")
    assert "requires human approval" in output


async def test_run_orchestration_respects_configured_timeout():
    config = OrchestrationConfig(timeout_seconds=0.05)
    run_orchestration = orchestration_tool(_tools_by_name(), config)
    # No tool calls needed — a pure-Python busy loop is enough to blow the
    # tiny configured timeout without depending on real tool latency.
    # (RestrictedPython requires an explicit `_inplacevar_` guard for `+=`,
    # which this harness doesn't provide — same as the reference it's ported
    # from — so the loop uses plain assignment instead.)
    code = "n = 0\nfor i in range(10_000_000):\n    n = n + i\nresult = n\n"
    output = await run_orchestration.ainvoke({"code": code})
    assert output.startswith("Orchestration failed:")
    assert "timeout" in output.lower()

"""Tests for the orchestration tool_bridge — in particular the HITL-safe
default_policy this template re-derives from the reference (see the module
docstring in tool_bridge.py for why it's tighter: mutations must never be
reachable from inside the sandbox, since HumanInTheLoopMiddleware can't see
calls made through the bridge)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import StructuredTool

from ai_agent_template.agent.orchestration.tool_bridge import (
    OrchestrationPolicyError,
    ToolBridge,
    default_policy,
)


async def _fake_get_tasks() -> str:
    return '{"tasks": []}'


async def _fake_update_task(uid: str) -> str:
    return '{"ok": true}'


def _tools_by_name() -> dict:
    get_tasks = StructuredTool.from_function(coroutine=_fake_get_tasks, name="get_tasks", description="read")
    update_task = StructuredTool.from_function(
        coroutine=_fake_update_task, name="update_task", description="mutate"
    )
    return {"get_tasks": get_tasks, "update_task": update_task}


async def test_default_policy_allows_readonly_tool():
    result = await default_policy("get_tasks", {}, _tools_by_name())
    assert result == '{"tasks": []}'


async def test_default_policy_denies_mutation_tool():
    with pytest.raises(OrchestrationPolicyError, match="requires human approval"):
        await default_policy("update_task", {"uid": "t1"}, _tools_by_name())


async def test_default_policy_denies_blocked_tool(monkeypatch):
    monkeypatch.setattr(
        "ai_agent_template.agent.orchestration.tool_bridge.is_blocked_tool",
        lambda name: name == "get_tasks",
    )
    with pytest.raises(OrchestrationPolicyError, match="hard-blocked"):
        await default_policy("get_tasks", {}, _tools_by_name())


async def test_default_policy_denies_unbound_tool():
    with pytest.raises(OrchestrationPolicyError, match="not a bound tool"):
        await default_policy("get_projects", {}, _tools_by_name())


async def test_bridge_call_blocks_worker_thread_until_coroutine_resolves():
    loop = asyncio.get_running_loop()
    bridge = ToolBridge(_tools_by_name(), loop)

    def _worker():
        return bridge.get_tasks()

    result = await asyncio.to_thread(_worker)
    assert result == {"tasks": []}
    assert bridge.call_log == ["get_tasks"]


async def test_bridge_rejects_mutation_from_worker_thread():
    loop = asyncio.get_running_loop()
    bridge = ToolBridge(_tools_by_name(), loop)

    def _worker():
        return bridge.update_task(uid="t1")

    with pytest.raises(Exception, match="requires human approval"):
        await asyncio.to_thread(_worker)


async def test_bridge_enforces_call_budget():
    loop = asyncio.get_running_loop()
    bridge = ToolBridge(_tools_by_name(), loop, max_calls=1)

    def _worker():
        bridge.get_tasks()
        bridge.get_tasks()

    with pytest.raises(Exception, match="Exceeded 1 tool calls"):
        await asyncio.to_thread(_worker)


async def test_bridge_unknown_attribute_raises_attribute_error():
    loop = asyncio.get_running_loop()
    bridge = ToolBridge(_tools_by_name(), loop)
    with pytest.raises(AttributeError):
        bridge.not_a_tool

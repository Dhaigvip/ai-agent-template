"""Structural tests for the RestrictedPython exec harness — no AWS, no MCP."""

from __future__ import annotations

import asyncio

import pytest

from ai_agent_template.agent.orchestration.sandbox import run_sandboxed


class _FakeTools:
    def get_tasks(self, **kwargs):
        return {"tasks": [{"uid": "t1", "title": "one"}]}


async def test_result_variable_returned():
    outcome = await run_sandboxed("result = {'a': 1}", _FakeTools())
    assert outcome.ok
    assert outcome.result_text == '{"a": 1}'


async def test_print_captured_separately_from_result():
    outcome = await run_sandboxed("print('hello')\nresult = 1", _FakeTools())
    assert outcome.ok
    assert outcome.stdout == "hello\n"
    assert outcome.result_text == "1"


async def test_missing_result_variable_fails_clearly():
    outcome = await run_sandboxed("x = 1", _FakeTools())
    assert not outcome.ok
    assert "result" in outcome.error


async def test_non_json_serializable_result_fails_clearly():
    outcome = await run_sandboxed("result = {1, 2, 3}", _FakeTools())
    assert not outcome.ok
    assert "JSON-serializable" in outcome.error


async def test_syntax_error_does_not_crash():
    outcome = await run_sandboxed("def broken(:", _FakeTools())
    assert not outcome.ok
    assert "did not compile" in outcome.error


async def test_runtime_exception_surfaced_not_raised():
    outcome = await run_sandboxed("result = 1 / 0", _FakeTools())
    assert not outcome.ok
    assert "ZeroDivisionError" in outcome.error


async def test_imports_are_blocked():
    outcome = await run_sandboxed("import os\nresult = 1", _FakeTools())
    assert not outcome.ok


async def test_bound_tool_call_reaches_fake_tools():
    outcome = await run_sandboxed("result = tools.get_tasks()", _FakeTools())
    assert outcome.ok
    assert outcome.result_text == '{"tasks": [{"uid": "t1", "title": "one"}]}'


async def test_timeout_returns_clean_error_not_exception():
    class _SlowTools:
        def get_tasks(self, **kwargs):
            import time

            time.sleep(1)
            return {}

    outcome = await run_sandboxed(
        "result = tools.get_tasks()", _SlowTools(), timeout=0.05
    )
    assert not outcome.ok
    assert "timeout" in outcome.error.lower()


async def test_result_and_stdout_truncated():
    outcome = await run_sandboxed(
        "result = 'x' * 100", _FakeTools(), max_result_chars=10
    )
    assert outcome.ok
    assert "truncated" in outcome.result_text

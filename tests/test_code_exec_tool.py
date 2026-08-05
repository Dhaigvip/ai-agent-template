"""End-to-end (no AWS) tests for the run_python LangChain tool."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_agent_template.analysis.code_exec_tool import code_exec_tool
from ai_agent_template.analysis.types import ExecutionOutcome


@dataclass
class _FakeService:
    """Stand-in for CodeExecutionService — returns a canned ExecutionOutcome."""

    outcome: ExecutionOutcome
    raise_error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    async def run(self, code: str) -> ExecutionOutcome:
        self.calls.append(code)
        if self.raise_error:
            raise self.raise_error
        return self.outcome


async def test_run_python_returns_text_output(tmp_path, monkeypatch):
    from ai_agent_template.analysis import file_manager

    monkeypatch.setattr(file_manager, "_file_registry", file_manager.FileRegistry(tmp_path))
    service = _FakeService(ExecutionOutcome(output="mean = 4.5", error=None))
    run_python = code_exec_tool(service)

    output = await run_python.ainvoke({"code": "print('mean = 4.5')"})
    assert output == "mean = 4.5"
    assert service.calls == ["print('mean = 4.5')"]


async def test_run_python_reports_execution_failure_without_raising():
    service = _FakeService(ExecutionOutcome(output="", error="ZeroDivisionError: division by zero"))
    run_python = code_exec_tool(service)

    output = await run_python.ainvoke({"code": "result = 1 / 0"})
    assert output.startswith("Code execution failed:")
    assert "ZeroDivisionError" in output


async def test_run_python_embeds_chart_markdown(tmp_path, monkeypatch):
    from ai_agent_template.analysis import file_manager

    monkeypatch.setattr(file_manager, "_file_registry", file_manager.FileRegistry(tmp_path))
    service = _FakeService(
        ExecutionOutcome(output="done", files=[("chart.plotly.json", b'{"data": []}')])
    )
    run_python = code_exec_tool(service)

    output = await run_python.ainvoke({"code": "..."})
    assert "done" in output
    assert "![chart.plotly.json](/api/agent/download/" in output


async def test_run_python_handles_no_text_output():
    service = _FakeService(ExecutionOutcome(output="", error=None))
    run_python = code_exec_tool(service)

    output = await run_python.ainvoke({"code": "x = 1"})
    assert "(code ran with no text output)" in output


async def test_run_python_surfaces_unexpected_error_without_crashing():
    service = _FakeService(ExecutionOutcome(output=""), raise_error=RuntimeError("boto3 exploded"))
    run_python = code_exec_tool(service)

    output = await run_python.ainvoke({"code": "..."})
    assert "Code execution error" in output
    assert "boto3 exploded" in output

"""Native `run_python` agent tool — server-owned code execution for chat.

Distinct from Task 17's `run_orchestration` (agent/orchestration/): that tool
is a local, sub-second, in-process sandbox whose entire purpose is calling
this project's own bound tools in a loop — it has no pandas/numpy/plotly and
cannot do math. This tool is the opposite: a remote AgentCore sandbox for
data analysis/statistics/charts, with NO access to this project's own tools
at all. Use `run_orchestration` for dependent tool calls, `run_python` for
computation or visualization.

Backed by the shared `CodeExecutionService` (boto3 -> AgentCore) — the LLM
NEVER sees start/stop, the server owns the sandbox lifecycle.

Lifecycle: each call is self-contained — start a sandbox, run the code,
register any produced files for download, stop the sandbox. Leak-free by
construction. Because each call is a fresh sandbox with no memory of
previous calls, the model must put a complete analysis (data + computation +
chart) in a SINGLE call.

Registered chart/file URLs (`/api/agent/download/<id>`) survive sandbox
release (FileRegistry keeps them ~1h).

Gated solely by `CodeExecutionConfig.interpreter_id` presence — see
`code_execution_service.get_code_execution_service()`.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from ai_agent_template.analysis.chart_extraction import register_outputs
from ai_agent_template.analysis.code_execution_service import CodeExecutionService

logger = logging.getLogger(__name__)


def code_exec_tool(service: CodeExecutionService):
    """Return the `run_python` LangChain tool bound to a CodeExecutionService."""

    @tool
    async def run_python(code: str) -> str:
        """STOP — only use this tool when the user explicitly asks for a chart/plot/graph OR
        the answer requires mathematical computation (statistics, correlation, regression,
        cost formulas). If the task is finding, listing, filtering, or checking data — use
        get_* tools and answer from their JSON directly. NEVER run Python just to look things up.

        GATE — ask yourself BEFORE calling: "Does the answer require MATH or a VISUAL?"
        - If NO -> do NOT call this tool. Use get_* tools and answer directly.
        - If YES -> proceed below.

        DON'T UNDERSTAND A CONCEPT? If the user mentions a term you cannot map to a
        known tool or entity, call kb_retrieve FIRST to learn what it means (if available).
        Do NOT use run_python to "figure out" unfamiliar domain concepts.

        NEVER use this tool for:
        - "Which tasks are overdue?" -> get_tasks, read the JSON, answer.
        - "Are there projects without a milestone?" -> get_projects with fields, check, answer.
        - "List/find/filter items where ..." -> get_* tool, read result, answer.
        - "What properties does X have?" -> get_* tool, read result, answer.
        - Any retrieval, create, update, count, lookup, or existence check.
        These are DATA QUERIES — the get_* tools already return the answer as JSON.

        ONLY use this tool for:
        - Computation the get_* result alone cannot answer — statistics, aggregation,
          distributions, correlations, pivots, outliers, cost math, formulas.
        - Visualizations the user explicitly asks to "show"/"plot"/"chart"/"graph"/"compare".

        HYBRID PATTERN (for genuine computation/plots ONLY): fetch data with this project's
        get_* tools, then pass those rows INTO this code (as a Python literal) and
        compute/plot. Never try to re-fetch project data inside Python — this sandbox
        has no access to this project's own tools (use run_orchestration for that instead).

        SANDBOX: the server owns the sandbox — you only write Python and call this tool; you never
        start, stop, install, upload, or download anything. Each call runs in a FRESH sandbox with
        no memory of previous calls, so put the COMPLETE analysis — load/build data, compute, and
        save any chart — in ONE call (do not split loading and plotting across calls). pandas, numpy
        and plotly are preinstalled. Print a concise text summary; never print large raw dataframes.

        CHARTS: build charts with PLOTLY and save the figure as JSON (NOT an image):
            import plotly.express as px            # or plotly.graph_objects as go
            fig = px.bar(df, x="category", y="value")
            with open("chart.plotly.json", "w") as f:
                f.write(fig.to_json())
        Use ONLY these chart types — bar, line, scatter, pie (the client renders a BASIC Plotly build).
        NOT available (would render blank), so substitute:
          - distribution/histogram -> compute the bins/counts yourself and draw a BAR chart;
          - matrix/heatmap          -> aggregate (row/column means or totals) into a BAR chart;
          - box/violin/3D/maps/contour -> pick a bar/line/scatter that conveys the same point.
        Do NOT call fig.show() or fig.write_image(). Save every figure to its own '*.plotly.json'.
        The result returns ready-to-paste markdown (e.g. ![chart.plotly.json](/api/agent/download/...));
        ALWAYS paste that exact markdown into your final answer so the user sees the interactive chart.
        """
        try:
            outcome = await service.run(code)

            if not outcome.ok:
                logger.info("run_python: execution error")
                return f"Code execution failed:\n{outcome.error}"

            charts, downloads = register_outputs(outcome.files)

            # Hand the model ready-to-paste markdown so it embeds the chart in its
            # answer (react-markdown renders it as an <img>). This is the rendering
            # path — there is no separate chart wire event (yet).
            parts: list[str] = [outcome.output.strip() or "(code ran with no text output)"]
            for c in charts:
                parts.append(
                    f"Chart produced — include this markdown image in your answer:\n"
                    f"![{c.filename}]({c.url})"
                )
            for d in downloads:
                parts.append(f"File produced — link it in your answer: [{d.filename}]({d.url})")
            logger.info("run_python: ok (charts=%d files=%d)", len(charts), len(downloads))
            return "\n".join(parts)

        except Exception as e:  # noqa: BLE001 — never crash the agent turn
            logger.error("run_python: unexpected error: %s", e)
            from ai_agent_template.agent.resilient_persistence import _is_credential_error

            if _is_credential_error(e):
                from ai_agent_template.agent.graph_builder import _on_persistence_degraded

                _on_persistence_degraded(
                    "code_interpreter",
                    "Code execution (charts/analysis) is disabled: AWS credentials "
                    "for AgentCore Code Interpreter have expired or are missing.",
                )
                return (
                    "Code execution is temporarily unavailable: AWS credentials for "
                    "AgentCore Code Interpreter have expired. Please inform the user that "
                    "code execution/charts are disabled until credentials are refreshed. "
                    "Answer their question using available data without running code."
                )
            return f"Code execution error: {e}"

    return run_python

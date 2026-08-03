"""CloudWatch metrics integration for AgentCore observability.

Emits custom metrics from PerfCounters to CloudWatch (via ADOT) for
dashboard visualization: turn duration, LLM/tool call counts and latency,
HITL wait, checkpoint-read time.

Token/cache-read/cache-write counts are NOT emitted here — same as the
reference. They already reach structured logs and the client's own `perf`
wire event (agent/ai_gateway.py's `_emit_turn_summary`, via
PerfCounters.summary()); this module is CloudWatch dashboard duration/count
metrics specifically, gated by whether ADOT tracing is enabled at all
(see observability/tracer.py's is_enabled()).

Usage:
    from ai_agent_template.observability.metrics import emit_turn_metrics

    # After a turn completes
    emit_turn_metrics(perf, service_name=config.observability.otel_service_name,
                       session_id=session_id, user_id=user_id)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def emit_turn_metrics(
    perf_counters: Any,
    *,
    service_name: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Emit CloudWatch metrics from PerfCounters after a turn completes.

    No-op if observability isn't enabled (checked via tracer.is_enabled()).
    """
    from ai_agent_template.observability.tracer import is_enabled, get_meter

    if not is_enabled():
        return

    try:
        meter = get_meter()

        # Common attributes for all metrics
        attrs: dict[str, Any] = {"service.name": service_name}
        if session_id:
            attrs["session_id"] = session_id[:8]  # Truncate for cardinality
        if user_id:
            attrs["user_id"] = user_id

        # ── Turn-level metrics ────────────────────────────────────────────────
        turn_duration = meter.create_histogram(
            "agent.turn.duration_ms",
            description="Total turn duration in milliseconds",
            unit="ms",
        )
        total_ms = int((time.monotonic() - perf_counters.turn_start) * 1000)
        turn_duration.record(total_ms, attributes=attrs)

        # ── LLM metrics ───────────────────────────────────────────────────────
        if perf_counters.llm_calls > 0:
            llm_calls = meter.create_counter(
                "agent.llm.calls",
                description="Number of LLM API calls",
                unit="calls",
            )
            llm_calls.add(perf_counters.llm_calls, attributes=attrs)

            llm_duration = meter.create_histogram(
                "agent.llm.total_duration_ms",
                description="Total LLM call duration in milliseconds",
                unit="ms",
            )
            llm_duration.record(perf_counters.llm_total_ms, attributes=attrs)

            avg_llm_ms = perf_counters.llm_total_ms / perf_counters.llm_calls
            llm_avg = meter.create_histogram(
                "agent.llm.avg_duration_ms",
                description="Average LLM call duration in milliseconds",
                unit="ms",
            )
            llm_avg.record(int(avg_llm_ms), attributes=attrs)

        # ── Tool metrics ──────────────────────────────────────────────────────
        if perf_counters.tool_calls > 0:
            tool_calls = meter.create_counter(
                "agent.tool.calls",
                description="Number of tool calls",
                unit="calls",
            )
            tool_calls.add(perf_counters.tool_calls, attributes=attrs)

            tool_duration = meter.create_histogram(
                "agent.tool.total_duration_ms",
                description="Total tool call duration in milliseconds",
                unit="ms",
            )
            tool_duration.record(perf_counters.tool_total_ms, attributes=attrs)

            avg_tool_ms = perf_counters.tool_total_ms / perf_counters.tool_calls
            tool_avg = meter.create_histogram(
                "agent.tool.avg_duration_ms",
                description="Average tool call duration in milliseconds",
                unit="ms",
            )
            tool_avg.record(int(avg_tool_ms), attributes=attrs)

        # ── HITL metrics ──────────────────────────────────────────────────────
        if perf_counters.interrupt_wait_ms > 0:
            hitl_wait = meter.create_histogram(
                "agent.hitl.wait_duration_ms",
                description="Human-in-the-loop wait duration in milliseconds",
                unit="ms",
            )
            hitl_wait.record(perf_counters.interrupt_wait_ms, attributes=attrs)

        # ── Checkpoint metrics ────────────────────────────────────────────────
        if perf_counters.checkpoint_read_ms > 0:
            checkpoint_read = meter.create_histogram(
                "agent.checkpoint.read_duration_ms",
                description="Checkpoint read duration in milliseconds",
                unit="ms",
            )
            checkpoint_read.record(perf_counters.checkpoint_read_ms, attributes=attrs)

        logger.debug(
            "metrics.emitted: total_ms=%d llm_calls=%d tool_calls=%d",
            total_ms,
            perf_counters.llm_calls,
            perf_counters.tool_calls,
        )

    except Exception as err:
        logger.warning("metrics.emit_failed: %s", err)

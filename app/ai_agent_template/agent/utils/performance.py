"""Performance tracking utilities for agent operations."""

from __future__ import annotations

import time


class PerfCounters:
    """Lightweight per-turn timing accumulator.

    Tracks wall-clock time in each pipeline phase so the developer can see
    exactly where time is going. Emitted as a structured log + optional
    client event after each turn.
    """

    def __init__(self) -> None:
        self.turn_start: float = time.monotonic()
        # Phase durations (ms)
        self.graph_build_ms: int = 0
        self.llm_calls: int = 0
        self.llm_total_ms: int = 0
        self.tool_calls: int = 0
        self.tool_total_ms: int = 0
        self.interrupt_wait_ms: int = 0
        self.checkpoint_read_ms: int = 0
        # Token totals across ALL llm calls in the turn (select + execute + answer).
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.cache_write_tokens: int = 0
        # Internal scratch
        self._llm_start: float | None = None
        self._interrupt_start: float | None = None
        self._checkpoint_start: float | None = None

    def start_llm(self) -> None:
        self._llm_start = time.monotonic()

    def end_llm(self) -> None:
        if self._llm_start:
            self.llm_calls += 1
            self.llm_total_ms += int((time.monotonic() - self._llm_start) * 1000)
            self._llm_start = None

    def add_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        """Accumulate token usage from one llm response (called per llm call in the turn)."""
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cache_read_tokens += int(cache_read or 0)
        self.cache_write_tokens += int(cache_write or 0)

    def record_tool_call(self) -> None:
        """Count one tool call, as the model emits it.

        Count and duration are recorded separately because they occur at different
        times: emission and result arrival.
        """
        self.tool_calls += 1

    def record_tool_call_duration(self, duration_ms: int) -> None:
        """Add one tool call's elapsed time, measured emission -> result.

        The v3 tool node runs a message's calls CONCURRENTLY, so these spans overlap and
        tool_total_ms can exceed total_ms on a parallel batch. That is the concurrency
        being visible, not a bug — compare it against tool_calls, not against the clock.
        """
        self.tool_total_ms += max(0, int(duration_ms))

    def start_interrupt_wait(self) -> None:
        self._interrupt_start = time.monotonic()

    def end_interrupt_wait(self) -> None:
        if self._interrupt_start:
            self.interrupt_wait_ms += int((time.monotonic() - self._interrupt_start) * 1000)
            self._interrupt_start = None

    def start_checkpoint_read(self) -> None:
        self._checkpoint_start = time.monotonic()

    def end_checkpoint_read(self) -> None:
        if self._checkpoint_start:
            self.checkpoint_read_ms += int((time.monotonic() - self._checkpoint_start) * 1000)
            self._checkpoint_start = None

    @property
    def total_ms(self) -> int:
        return int((time.monotonic() - self.turn_start) * 1000)

    def summary(self) -> dict:
        total = self.total_ms
        accounted = (
            self.graph_build_ms
            + self.llm_total_ms
            + self.tool_total_ms
            + self.interrupt_wait_ms
            + self.checkpoint_read_ms
        )
        return {
            "total_ms": total,
            "graph_build_ms": self.graph_build_ms,
            "llm_calls": self.llm_calls,
            "llm_total_ms": self.llm_total_ms,
            "tool_calls": self.tool_calls,
            "tool_total_ms": self.tool_total_ms,
            "interrupt_wait_ms": self.interrupt_wait_ms,
            "checkpoint_read_ms": self.checkpoint_read_ms,
            "overhead_ms": max(0, total - accounted),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


class TurnContext:
    """Mutable state tracked across one turn execution."""

    def __init__(self) -> None:
        self.has_executed_tools = False
        # Per-message text buffer (key "_msg"); flushed at on_llm_end as thinking-or-answer.
        self.pending_text_buffers: dict[str, str] = {}
        # Set True when the model call streams through with no further tool
        # calls -- the in-process signal that this graph pass reached END
        # normally (vs. paused on an interrupt).
        self.finalize_completed = False

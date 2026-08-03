"""Project streamed model output into wire events.

This module consumes text and tool-call projections concurrently so events
are emitted in arrival order. It emits text deltas immediately, commits
narration at tool-call boundaries, and emits final answer commits when a
message completes.

Uses LangGraph's `astream_events(..., version="v3")` — marked experimental
upstream, but this is exactly what the reference runs in production and
what this template pins its `langgraph`/`langchain` versions to match.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Callable

from langchain_core.messages import ToolMessage

from ai_agent_template.agent.utils.performance import PerfCounters, TurnContext

logger = logging.getLogger(__name__)

_DONE = object()  # queue sentinel


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        return tc.get("name") or ""
    return getattr(tc, "name", "") or ""


def _tool_call_args(tc: Any) -> dict:
    if isinstance(tc, dict):
        return tc.get("args") or {}
    return getattr(tc, "args", None) or {}


def _tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return tc.get("id") or tc.get("tool_call_id") or tc.get("call_id") or ""
    return (
        getattr(tc, "id", "") or getattr(tc, "tool_call_id", "") or getattr(tc, "call_id", "") or ""
    )


def _normalize_tool_call_args(raw: Any) -> tuple[dict, bool]:
    """Return (args_dict, is_complete).

    In v3 streaming, some providers surface incremental tool args as numeric-key maps where each
    key is a character position. Those chunks should not be emitted until parseable JSON exists.
    """
    if raw is None:
        return {}, False

    if isinstance(raw, dict):
        if not raw:
            return {}, True
        # Incremental character chunks: {"0": "{", "1": "\"", ...}
        if all(isinstance(k, str) and k.isdigit() for k in raw.keys()):
            try:
                s = "".join(str(raw[k]) for k in sorted(raw.keys(), key=int))
                parsed = json.loads(s)
            except Exception:
                return {}, False
            return (parsed if isinstance(parsed, dict) else {}), True
        return raw, True

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}, False
        return (parsed if isinstance(parsed, dict) else {}), True

    return {}, False


class _MessagePump:
    """Projects ONE LLM call's streams onto the wire queue."""

    def __init__(
        self,
        queue: asyncio.Queue,
        ctx: TurnContext,
        perf: PerfCounters,
        log: logging.Logger,
        tool_started: dict[str, float],
    ) -> None:
        self.q = queue
        self.ctx = ctx
        self.perf = perf
        self.log = log
        # call_id -> monotonic at emission; read by pump_values to time each result.
        self.tool_started = tool_started
        self._parts: list[str] = []
        self._committed = False
        self._anon_tool_seq = 0

    def _count_tool_call(self, call_id: str) -> None:
        """Count one tool call as the model emits it.

        Counted HERE rather than where results arrive because pump_values reads full
        state snapshots, which on turn 2+ still contain every earlier turn's
        ToolMessages — counting there would tally history into this turn.
        """
        self.tool_started[call_id] = time.monotonic()
        self.perf.record_tool_call()

    async def _commit(self, kind: str) -> None:
        """Flush streamed text with its classification: "thinking" (narration) or "answer"."""
        text = "".join(self._parts)
        self._parts.clear()
        if not text.strip():
            return
        await self.q.put({"type": "text_commit", "as": kind, "text": text})

    async def run(self, handle: Any) -> None:
        async def pump_text() -> None:
            async for delta in handle.text:
                if not delta:
                    continue
                self._parts.append(delta)
                await self.q.put({"type": "text_delta", "delta": delta})

        async def pump_tool_calls() -> None:
            pending: dict[str, dict[str, Any]] = {}
            emitted: set[str] = set()

            async for tc in handle.tool_calls:
                if not self._committed:
                    # Block boundary: everything streamed so far was narration.
                    self._committed = True
                    await self._commit("thinking")
                name = _tool_call_name(tc)
                call_id = _tool_call_id(tc)
                if not call_id:
                    self._anon_tool_seq += 1
                    call_id = f"anon:{self._anon_tool_seq}:{name}"

                raw_args = _tool_call_args(tc)
                args, complete = _normalize_tool_call_args(raw_args)
                if not complete:
                    self.log.debug("tool.call.partial: %s call_id=%s", name, call_id)
                    continue

                pending[call_id] = {"toolName": name, "input": args, "callId": call_id}

                # For streamed calls, wait for parseable/non-empty args before first emission.
                # Empty-args calls are flushed after stream end if they never receive updates.
                if args and call_id not in emitted:
                    self.log.debug("tool.call: %s call_id=%s", name, call_id)
                    await self.q.put({"type": "tool_auto", **pending[call_id]})
                    emitted.add(call_id)
                    self._count_tool_call(call_id)

            # Flush calls that never got non-empty args (legitimate empty-input tools).
            for call_id, evt in pending.items():
                if call_id in emitted:
                    continue
                self.log.debug("tool.call: %s call_id=%s", evt["toolName"], call_id)
                await self.q.put({"type": "tool_auto", **evt})
                self._count_tool_call(call_id)

        self.perf.start_llm()
        results = await asyncio.gather(pump_text(), pump_tool_calls(), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                self.log.warning("wire message projection failed: %s", r)
        self.perf.end_llm()

        await self._record_usage(handle)

        if not self._committed:
            # No tool-call block => this message is terminal => the turn is done.
            self.ctx.finalize_completed = True
            await self._commit("answer")

    async def _record_usage(self, handle: Any) -> None:
        """Token accounting from the assembled message's usage_metadata."""
        try:
            msg = await handle.output
            usage = getattr(msg, "usage_metadata", None)
            if not isinstance(usage, dict):
                self.log.debug("wire usage_metadata absent on assembled message")
                return
            details = usage.get("input_token_details") or {}
            self.perf.add_tokens(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read=details.get("cache_read", 0),
                cache_write=details.get("cache_creation", 0),
            )
        except Exception as err:  # never break a turn over metrics
            self.log.debug("wire usage unavailable: %s", err)


async def stream_wire_events(
    run: Any,
    ctx: TurnContext,
    perf: PerfCounters,
    log: logging.Logger,
    is_mutation_result: Callable[[str], bool],
) -> AsyncGenerator[dict, None]:
    """Project a v3 run stream onto wire events, in arrival order.

    A queue fans the concurrent projections back into one ordered generator — an async generator
    cannot itself `yield` from inside asyncio.gather.
    """
    queue: asyncio.Queue = asyncio.Queue()
    # call_id -> monotonic at emission. Written by the message pumps, read by pump_values
    # to attribute a duration to each tool result.
    tool_started: dict[str, float] = {}

    async def pump_messages() -> None:
        async for handle in run.messages:
            # Only project the main agent model call. Handles without a node
            # (for tests/stubs) are allowed through.
            node = getattr(handle, "node", None)
            if node is not None and node != "model":
                log.debug("wire skip non-agent message handle: node=%s", node)
                continue
            await _MessagePump(queue, ctx, perf, log, tool_started).run(handle)

    async def pump_values() -> None:
        """Tool RESULTS. There is no `tools` channel in langgraph (channels: messages,
        values, subgraphs, extensions), so results are picked out of state snapshots."""
        seen: set[str] = set()
        async for snapshot in run.values:
            if not isinstance(snapshot, dict):
                continue
            for msg in snapshot.get("messages") or []:
                if not isinstance(msg, ToolMessage):
                    continue
                key = msg.tool_call_id or id(msg)
                if key in seen:
                    continue
                seen.add(key)
                # Duration for calls this turn emitted. A result with no start time is a
                # ToolMessage replayed from earlier-turn state (see _count_tool_call) —
                # it was never counted, so it contributes no time either.
                started = tool_started.pop(msg.tool_call_id, None)
                if started is not None:
                    perf.record_tool_call_duration(int((time.monotonic() - started) * 1000))
                await queue.put(_activity_event(msg, is_mutation_result, ctx, log))
                if _is_refresh(msg, is_mutation_result):
                    await queue.put({"type": "refresh"})

    async def driver() -> None:
        try:
            await asyncio.gather(pump_messages(), pump_values())
        finally:
            await queue.put(_DONE)

    task = asyncio.create_task(driver())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item
        await task
    finally:
        if not task.done():
            task.cancel()


def _content_str(msg: ToolMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content or "")


def _is_error(msg: ToolMessage) -> bool:
    # Prefer status when present; keep a content fallback for tools without status.
    if getattr(msg, "status", None) == "error":
        return True
    content = _content_str(msg)
    return content.startswith("Tool error:") or content == '"Tool call rejected by user."'


def _is_refresh(msg: ToolMessage, is_mutation_result: Callable[[str], bool]) -> bool:
    return not _is_error(msg) and is_mutation_result(_content_str(msg))


def _activity_event(
    msg: ToolMessage, is_mutation_result, ctx: TurnContext, log: logging.Logger
) -> dict:
    name = getattr(msg, "name", "unknown") or "unknown"
    content = _content_str(msg)
    ctx.has_executed_tools = True
    first_line = next((line for line in content.split("\n") if line.strip()), "")
    detail = (first_line[:80] + "...") if len(first_line) > 80 else first_line
    log.debug("tool.result: %s is_error=%s chars=%d", name, _is_error(msg), len(content))
    return {
        "type": "activity",
        "header": name.replace("_", " "),
        "detail": detail or None,
        "toolName": name,
        "isError": _is_error(msg),
    }

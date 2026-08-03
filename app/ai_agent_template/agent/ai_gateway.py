"""Gateway for one chat session.

Builds the graph lazily, streams turn events, and handles resume paths for
human-in-the-loop decisions.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from ai_agent_template.agent.context import AgentContext
from ai_agent_template.agent.graph_builder import (
    BuiltGraph,
    build_graph,
    drain_degradation_warnings,
    _on_persistence_degraded,
)
from ai_agent_template.agent.middleware.risk_classifier import is_blocked_tool
from ai_agent_template.agent.resilient_persistence import _is_credential_error
from ai_agent_template.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from ai_agent_template.agent.utils.log import agent_log, set_log_context
from ai_agent_template.agent.utils.performance import PerfCounters, TurnContext
from ai_agent_template.agent.utils.tool_helpers import create_synthetic_tools, mcp_tool_to_langchain
from ai_agent_template.agent.wire import stream_wire_events
from ai_agent_template.config import AppConfig
from ai_agent_template.knowledge_base import get_kb_client
from ai_agent_template.mcp.client import MCPClient

logger = logging.getLogger(__name__)


def _compact(value: Any, max_len: int = 300) -> str:
    """Compact value for tracing logs without flooding output."""
    try:
        s = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        s = repr(value)
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


# ── Mutation result ──────────────────────────────────────────────────────────
# demo-schema.graphql's mutations return {ok, message, <entity>} -- none of
# the reference's product-specific changeset markers (createdChangeSetUid/
# id/uid) exist in this domain.
# mcp-server-template's own tool_factory._mutation_result() maps `ok` to a
# `success` key specifically so downstream consumers like this gateway have
# one stable field to check regardless of the schema's own naming.


class MutationResult(BaseModel):
    success: bool


def _is_successful_mutation(content: str) -> bool:
    """Check if a tool result indicates a successful mutation."""
    try:
        result = MutationResult.model_validate_json(content)
        return result.success
    except (ValidationError, json.JSONDecodeError, ValueError):
        return False


class AIGateway:
    """One instance per WebSocket connection.

    MCP client lifetime:
      - Opened lazily on the first run_turn() call (alongside graph build).
      - Kept open for the full session so tool calls during execution work.
      - Closed in destroy(), called when the WS disconnects.
    """

    def __init__(self, config: AppConfig, init_config: dict[str, Any]) -> None:
        self._config = config
        self._ctx = init_config
        self._mcp: MCPClient | None = None  # opened on first turn
        self._graph_runtime: BuiltGraph | None = None
        self._thread_id = init_config.get("agent_session_id") or str(uuid.uuid4())
        self._user_id: str | None = init_config.get("ctx_user_id")

    async def destroy(self) -> None:
        """Close the MCP connection. Call when the WebSocket disconnects."""
        if self._mcp:
            try:
                await self._mcp.disconnect()
            except Exception as err:
                logger.debug("gateway.destroy: mcp disconnect error: %s", err)
            self._mcp = None
        self._graph_runtime = None

    # ── Graph config ──────────────────────────────────────────────────────────

    def _graph_config(self) -> dict:
        cfg: dict = {"thread_id": self._thread_id}
        # actor_id is required by an AgentCore-backed checkpointer to scope
        # memory per user. Falls back to "anonymous" when user_id is not
        # available (no-op for MemorySaver — it ignores unknown configurable
        # keys).
        if self._user_id:
            cfg["actor_id"] = self._user_id
        return {"configurable": cfg, "recursion_limit": 100}

    async def _build_graph_input(self, user_message: str) -> dict:
        # Clear turn-scoped tool cache — stale reads from the previous turn must not leak.
        if self._mcp:
            self._mcp.clear_turn_cache()

        return {
            "messages": [HumanMessage(content=user_message)],
            "context": AgentContext(
                user_id=self._ctx.get("ctx_user_id"),
                user_name=self._ctx.get("ctx_user_name"),
            ),
        }

    # ── Stateless HITL ────────────────────────────────────────────────────────
    # The turn ENDS at a HITL interrupt; any process holding the checkpointer resumes it later via
    # resume_turn(). No Future kept alive across the pause, no checkpoint polling — v3 event
    # streaming surfaces interrupts as API surface.

    @staticmethod
    def _hitl_request_event(value: dict) -> dict:
        """HumanInTheLoopMiddleware interrupt -> one wire event.

        Decisions are POSITIONAL (decision[i] answers actions[i]) and the middleware raises if the
        counts differ, so `index` is echoed explicitly. `allowed` tells the client which controls
        to render — approve/reject for a mutation, respond-only for ask_user.
        """
        configs = value.get("review_configs") or []
        return {
            "type": "hitl_request",
            "actions": [
                {
                    "index": i,
                    "toolName": req.get("name", ""),
                    "input": req.get("args") or {},
                    "description": req.get("description"),
                    "allowed": list(
                        (configs[i] if i < len(configs) else {}).get("allowed_decisions")
                        or ["approve", "reject"]
                    ),
                }
                for i, req in enumerate(value.get("action_requests") or [])
            ],
        }

    @staticmethod
    def _trace_wire_step(log: logging.Logger, step: int, evt: dict[str, Any]) -> None:
        """Log one high-signal execution step from the wire event stream."""
        et = evt.get("type")
        if et == "text_delta":
            # Token-level deltas are too noisy for execution tracing.
            return

        if et == "tool_auto":
            agent_log(
                log,
                logging.INFO,
                "trace.step",
                step=step,
                type=et,
                tool=evt.get("toolName"),
                call_id=evt.get("callId"),
                input=_compact(evt.get("input") or {}),
            )
            return

        if et == "activity":
            agent_log(
                log,
                logging.INFO,
                "trace.step",
                step=step,
                type=et,
                tool=evt.get("toolName"),
                is_error=evt.get("isError"),
                detail=evt.get("detail"),
            )
            return

        if et == "text_commit":
            agent_log(
                log,
                logging.INFO,
                "trace.step",
                step=step,
                type=et,
                as_type=evt.get("as"),
                chars=len(evt.get("text") or ""),
            )
            return

        agent_log(log, logging.INFO, "trace.step", step=step, type=et)

    async def _stream_stateless(
        self,
        stream_input: Any,
        graph_config: dict,
        ctx: TurnContext,
        perf: PerfCounters,
        log: logging.Logger,
    ) -> AsyncGenerator[dict, None]:
        """Stream until the turn finishes OR pauses. Never blocks on a human."""
        run = await self._graph_runtime.graph.astream_events(
            stream_input,
            graph_config,
            version="v3",
        )
        step = 0
        async for evt in stream_wire_events(run, ctx, perf, log, _is_successful_mutation):
            step += 1
            self._trace_wire_step(log, step, evt)
            yield evt

        # `interrupted` is a coroutine METHOD — `if run.interrupted:` is always True (a bound
        # method is truthy). Must be awaited.
        if await run.interrupted():
            interrupts = await run.interrupts()
            # HITL middleware fires exactly ONE interrupt carrying every pending action.
            value = (
                interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]["value"]
            )
            log.info(
                "turn.paused: awaiting %d decision(s)", len(value.get("action_requests") or [])
            )
            evt = self._hitl_request_event(value)
            step += 1
            agent_log(
                log,
                logging.INFO,
                "trace.step",
                step=step,
                type="hitl_request",
                actions=len(evt.get("actions") or []),
                tools=[(a or {}).get("toolName") for a in (evt.get("actions") or [])],
            )
            yield evt
            return

        step += 1
        done_evt = {"type": "done"}
        self._trace_wire_step(log, step, done_evt)
        yield done_evt

    async def resume_turn(
        self, decisions: list[dict], trace_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """Resume a thread paused on a HITL interrupt.

        `decisions` is positional, same length as the `actions` from the `hitl_request` event:
            {"type": "approve"}
            {"type": "edit", "edited_action": {"name": ..., "args": {...}}}
            {"type": "reject", "message": "..."}     # message optional
            {"type": "respond", "message": "..."}    # the human answers AS the tool
        """
        tid = trace_id or str(uuid.uuid4())[:8]
        log = logger.getChild(f"resume.{tid}")
        set_log_context(session_id=self._thread_id[:8], trace_id=tid, user_id=self._user_id)
        agent_log(log, logging.INFO, "turn.resume", decisions=len(decisions))

        if not self._graph_runtime:
            yield {
                "type": "error",
                "message": "No conversation to resume. Please send your message again.",
            }
            return

        perf, ctx = PerfCounters(), TurnContext()
        graph_config = self._graph_config()
        try:
            async for evt in self._stream_stateless(
                Command(resume={"decisions": decisions}),
                graph_config,
                ctx,
                perf,
                log,
            ):
                yield evt
        except Exception as err:
            log.error("resume.error: %s", err)
            yield {"type": "error", "message": str(err)}
            return

        async for evt in self._emit_turn_summary(log, perf, graph_config):
            yield evt

    # ── Main turn loop ────────────────────────────────────────────────────────

    async def run_turn(
        self, user_message: str, trace_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """Execute one turn of the agent conversation.

        Streams wire events as the graph executes. Handles a HITL interrupt by
        ending the turn — the client resumes via resume_turn().
        """
        tid = trace_id or str(uuid.uuid4())[:8]
        log = logger.getChild(f"turn.{tid}")
        set_log_context(session_id=self._thread_id[:8], trace_id=tid, user_id=self._user_id)
        agent_log(
            log, logging.INFO, "turn.start", message_len=len(user_message), query=user_message[:150]
        )

        perf = PerfCounters()

        # Input guardrail — validate the USER QUERY ONCE per turn, before any model call or
        # MCP connect. Decoupled from the model (not bound per-iteration). Fail-open.
        from ai_agent_template.agent.middleware.input_guardrail import apply_input_guardrail

        gr = await apply_input_guardrail(
            user_message,
            guardrail=self._config.guardrail,
            region=self._config.bedrock.region,
            profile=self._config.bedrock.profile,
        )
        if gr.blocked:
            agent_log(log, logging.WARNING, "turn.guardrail_blocked", policies=gr.policies)
            yield {"type": "text_commit", "as": "answer", "text": gr.message}
            yield {"type": "done"}
            return

        # Build graph on first turn
        try:
            async for evt in self._ensure_graph_built(log, perf):
                yield evt
        except Exception as err:
            yield {"type": "error", "message": str(err)}
            return

        # Emit any AgentCore degradation warnings (credentials expired mid-session)
        for _feature, _msg in drain_degradation_warnings():
            log.warning("feature.disabled: %s - %s", _feature, _msg)
            yield {"type": "warning", "feature": _feature, "message": _msg}

        agent_log(log, logging.INFO, "trace.step", step=0, type="status", message="Thinking...")
        yield {"type": "status", "message": "Thinking..."}

        graph_config = self._graph_config()
        stream_input: Any = await self._build_graph_input(user_message)
        ctx = TurnContext()

        # Stream until the turn finishes OR pauses for a HITL approval. On pause the turn ENDS
        # (see _stream_stateless) and the client resumes via resume_turn().
        try:
            async for evt in self._stream_stateless(stream_input, graph_config, ctx, perf, log):
                yield evt
        except Exception as err:
            log.error("graph.error: %s", err)
            if _is_credential_error(err):
                yield {
                    "type": "error",
                    "message": (
                        "Authentication failed - your AWS credentials have expired. "
                        "Please refresh your credentials and try again."
                    ),
                }
                yield {
                    "type": "warning",
                    "feature": "bedrock_model",
                    "message": (
                        "The Bedrock model (LLM) cannot be reached due to expired credentials. "
                        "All features are disabled until credentials are refreshed."
                    ),
                }
            else:
                yield {"type": "error", "message": str(err)}
            return

        # Emit warnings that arose during the turn (e.g. AgentCore credential failure mid-turn)
        for _feature, _msg in drain_degradation_warnings():
            log.warning("feature.disabled: %s - %s", _feature, _msg)
            yield {"type": "warning", "feature": _feature, "message": _msg}

        async for evt in self._emit_turn_summary(log, perf, graph_config):
            yield evt

    async def _ensure_graph_built(
        self, log: logging.Logger, perf: PerfCounters
    ) -> AsyncGenerator[dict, None]:
        """Connect to MCP, source tools, and build the graph on first turn."""
        if self._graph_runtime:
            return

        yield {"type": "status", "message": "Loading tools..."}
        t0 = time.monotonic()
        try:
            mcp_config = replace(self._config.mcp, agent_session_id=self._thread_id)
            self._mcp = MCPClient(mcp_config)
            await self._mcp.connect()

            tools_response = await self._mcp.list_tools()
            blocked = [t.name for t in tools_response.tools if is_blocked_tool(t.name)]
            if blocked:
                log.info("graph.build: excluded blocked tool(s): %s", ", ".join(blocked))
            mcp_tools = [
                mcp_tool_to_langchain(t, self._mcp)
                for t in tools_response.tools
                if not is_blocked_tool(t.name)
            ]
            kb = get_kb_client(
                self._config.knowledge_base,
                bedrock_region=self._config.bedrock.region,
                bedrock_profile=self._config.bedrock.profile,
            )
            kb_tools = [kb.kb_retrieve_tool(on_degraded=_on_persistence_degraded)] if kb else []
            all_tools = [*mcp_tools, *create_synthetic_tools(), *kb_tools]

            self._graph_runtime = build_graph(
                config=self._config, tools=all_tools, system_prompt=DEFAULT_SYSTEM_PROMPT
            )
            perf.graph_build_ms = int((time.monotonic() - t0) * 1000)
            agent_log(
                log, logging.DEBUG, "graph.built", duration_ms=perf.graph_build_ms, tools=len(all_tools)
            )
        except Exception as err:
            log.error("graph.build_failed: %s", err)
            if self._mcp:
                try:
                    await self._mcp.disconnect()
                except Exception:
                    pass
                self._mcp = None
            raise

    async def _emit_turn_summary(
        self, log: logging.Logger, perf: PerfCounters, graph_config: dict
    ) -> AsyncGenerator[dict, None]:
        """Emit performance metrics and state snapshot after turn completes."""
        _perf_summary = perf.summary()
        agent_log(log, logging.INFO, "turn.perf", **_perf_summary)
        yield {"type": "perf", **_perf_summary}

        # Best-effort — emit_turn_metrics() itself no-ops when ADOT isn't
        # enabled (tracer.is_enabled()), this try/except is only for the
        # rare case metrics emission itself misbehaves; never break a turn
        # over it.
        try:
            from ai_agent_template.observability.metrics import emit_turn_metrics

            emit_turn_metrics(
                perf,
                service_name=self._config.observability.otel_service_name,
                session_id=self._thread_id,
                user_id=self._user_id,
            )
        except Exception as _metrics_err:
            log.debug("turn.metrics_emission_failed: %s", _metrics_err)

        try:
            _state = await self._graph_runtime.graph.aget_state(graph_config)
            _sv = _state.values if _state else {}
            agent_log(
                log,
                logging.INFO,
                "turn.complete",
                messages=len(_sv.get("messages") or []),
            )
        except Exception as _state_err:
            log.debug("turn.complete: state read failed: %s", _state_err)

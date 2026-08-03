"""Structured logging helper for the agent pipeline.

Context fields (session_id, trace_id, user_id) are stored in a ContextVar so
they propagate automatically to every agent_log() call on the same asyncio
task without being passed explicitly.

Usage:
    from ai_agent_template.agent.utils.log import agent_log, set_log_context

    # Set once at turn start:
    set_log_context(session_id="abc12345", trace_id="f3e8a1b2", user_id="u_42")

    # Then anywhere in the call tree:
    agent_log(logger, logging.INFO, "agent.invoke", stage="select", selected=0)
    # Emits: "agent.invoke  session_id='abc12345'  trace_id='f3e8a1b2'  user_id='u_42'  stage='select'  selected=0"

Explicit kwargs always override the context values for that one call.
All None values (from context or explicit kwargs) are silently omitted.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

# Per-asyncio-task log context — set once per turn, inherited by all sub-tasks
_log_context: ContextVar[dict[str, Any]] = ContextVar("_log_context", default={})


def set_log_context(**fields: Any) -> None:
    """Bind context fields to the current asyncio task.

    Call once at the start of each turn. Fields are automatically appended
    to every subsequent agent_log() call on this task, including calls from
    the graph nodes and tool executor.

    Common fields: session_id, trace_id, user_id
    """
    _log_context.set({k: v for k, v in fields.items() if v is not None})


def agent_log(log: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log line: 'event  key=value  key=value ...'

    Context fields set via set_log_context() are prepended automatically.
    Explicit kwargs override context values for the same key in this call only.

    Args:
        log:    Logger instance (or child logger).
        level:  logging.INFO / logging.DEBUG / logging.WARNING etc.
        event:  Dotted event name, e.g. "turn.start", "agent.invoke".
        **fields: Per-call key/value pairs. None values are silently dropped.
    """
    ctx = _log_context.get()
    # Merge: context first, then explicit fields (explicit wins on collision)
    merged = {**ctx, **{k: v for k, v in fields.items() if v is not None}}
    parts = "  ".join(f"{k}={v!r}" for k, v in merged.items())
    log.log(level, "%s  %s" % (event, parts) if parts else event)

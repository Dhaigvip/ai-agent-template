"""Render ephemeral per-turn context blocks.

Memory and document context are assembled into one transient block that is
appended during model invocation. Never written into `messages` — see
agent/state.py's docstring on why.
"""

from __future__ import annotations

from typing import Any

from ai_agent_template.agent.context import AgentContext

TURN_CONTEXT_OPEN = "<turn_context>"
TURN_CONTEXT_CLOSE = "</turn_context>"


def render_turn_context(state: dict[str, Any]) -> str:
    """Render memory + documents into one block. Empty string when nothing applies."""
    parts: list[str] = []

    memory = state.get("memory_context")
    if memory:
        parts.append(str(memory))

    ctx: AgentContext | None = state.get("context")
    if ctx is not None and ctx.document_context:
        parts.append(str(ctx.document_context))

    return "\n\n".join(parts).strip()


def wrap_turn_context(rendered: str) -> str:
    """Wrap rendered context in its delimiters — the exact string sent to the model."""
    return f"{TURN_CONTEXT_OPEN}\n{rendered}\n{TURN_CONTEXT_CLOSE}"

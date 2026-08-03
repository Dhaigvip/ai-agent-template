"""Inject per-turn context into model calls.

This middleware appends one ephemeral turn-context block containing memory
and documents. Ephemeral means exactly that: it's appended to the messages
passed to THIS model call only, via request.override(), and never written
back into checkpointed state. If it ever lands in `messages` it compounds
every turn and destroys the prompt-cache prefix.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from ai_agent_template.agent.state import AgentState
from ai_agent_template.agent.utils.turn_context import render_turn_context, wrap_turn_context

logger = logging.getLogger(__name__)


class DynamicToolsMiddleware(AgentMiddleware[AgentState]):
    """Ephemeral <turn_context> injection at model-call time."""

    state_schema = AgentState

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """Append a trailing turn-context message before invoking the model."""
        messages = list(request.messages)
        turn_ctx = render_turn_context(request.state)
        if turn_ctx:
            messages.append(HumanMessage(content=wrap_turn_context(turn_ctx)))

        return await handler(request.override(messages=messages))

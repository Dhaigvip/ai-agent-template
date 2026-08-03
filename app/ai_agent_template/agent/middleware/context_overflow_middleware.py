"""Reactive safety net for context-window overflow.

If a model call fails because the request is too large, this middleware
applies an emergency trim and retries once.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages

from ai_agent_template.agent.state import AgentState

logger = logging.getLogger(__name__)

# Conservative budget for the emergency pass — well under any current model's window, since this
# only runs when the normal budget already overflowed.
_EMERGENCY_TOKEN_BUDGET = 40_000

_OVERFLOW_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "reduce the length",
    "request entity too large",
    "payload too large",
    "request too large",
    "too many tokens",
)


def is_context_length_error(err: Exception) -> bool:
    """True if the exception indicates the context window / request size was exceeded."""
    msg = str(getattr(err, "message", "") or err)
    status = getattr(err, "status_code", 0) or getattr(err, "status", 0)
    if any(kw in msg.lower() for kw in _OVERFLOW_KEYWORDS):
        return True
    return status == 413 or "413" in msg


def _emergency_trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the most recent messages under a tight budget; first kept message is a human turn."""
    return trim_messages(
        messages,
        max_tokens=_EMERGENCY_TOKEN_BUDGET,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        include_system=False,  # create_agent manages the system message separately
        allow_partial=False,
    )


class ContextOverflowRetryMiddleware(AgentMiddleware[AgentState]):
    """On a context-window overflow, emergency-trim the messages and retry the model call ONCE."""

    state_schema = AgentState

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        try:
            return await handler(request)
        except Exception as err:
            if not is_context_length_error(err):
                raise
            trimmed = _emergency_trim(list(request.messages))
            logger.warning(
                "ContextOverflowRetry: context window exceeded - emergency trim %d -> %d messages, "
                "retrying once",
                len(request.messages),
                len(trimmed),
            )
            return await handler(request.override(messages=trimmed))

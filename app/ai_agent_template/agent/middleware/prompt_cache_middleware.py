"""Inject Bedrock prompt-cache settings into each model call.

This middleware writes `cache_control` into `ModelRequest.model_settings`,
which the agent factory forwards to `bind_tools(..., **model_settings)`.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from ai_agent_template.agent.state import AgentState

logger = logging.getLogger(__name__)


class PromptCacheMiddleware(AgentMiddleware[AgentState]):
    """Inject `cache_control` into model_settings so Bedrock emits cachePoints."""

    state_schema = AgentState

    def __init__(self, cache_control: dict) -> None:
        super().__init__()
        self.cache_control = cache_control

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        settings = {**request.model_settings, "cache_control": self.cache_control}
        return await handler(request.override(model_settings=settings))

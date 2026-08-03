"""AgentState — the create_agent state schema for the middleware engine.

Extends LangChain's built-in AgentState (which owns `messages` + framework
channels) with this template's side-state channels. Side state is rendered
ephemerally at model-call time by the turn-context middleware and never
written into `messages`: if it lands in history it compounds every turn and
destroys the prompt-cache prefix.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain.agents.middleware import AgentState as BaseAgentState
from typing_extensions import NotRequired

from ai_agent_template.agent.context import AgentContext


def _keep_last(a: Any, b: Any) -> Any:
    """Reducer that keeps the last writer — used for read-only/side-state channels."""
    return b


class AgentState(BaseAgentState):
    """BaseAgentState + this template's side-state channels (rendered
    ephemerally, never persisted into messages)."""

    context: NotRequired[Annotated[AgentContext, _keep_last]]
    memory_context: NotRequired[Annotated[Optional[str], _keep_last]]
    attached_documents: NotRequired[Annotated[dict[str, str], _keep_last]]

"""AgentContext — per-turn session context for the single-agent graph."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AgentContext(BaseModel):
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    document_context: Optional[str] = None
    """Formatted block of user-attached documents, injected into the turn context
    at the start of each agent invocation."""

"""Shared agent types.

Expands into the full wire-protocol event union when streaming lands; for
now this is just what the HITL policy needs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class RiskInfo(BaseModel):
    level: Literal["high", "medium", "low", "readonly"]
    label: str
    color: Literal["red", "yellow", "green", "blue"]
    emoji: str
    requires_approval: bool
    default_approve: bool
    warning: Optional[str] = None

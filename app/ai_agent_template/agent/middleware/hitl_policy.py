"""HITL approval policy — which tools require a human, and what the human may decide.

Current UX rule (re-derived for the demo domain — see risk_classifier.py's
module docstring for why the reference's workspace-precondition special
case doesn't carry over):
- Mutations (create_*/update_*/...): require explicit `approve` or `reject`
- `ask_user`: `respond` only (the human reply IS the tool result)
- Read-only tools (get_*): auto-approved

Decision vocabulary (LangChain HumanInTheLoopMiddleware):
    approve  — run the tool as the model proposed it
    reject   — skip it; the model is told and continues
    respond  — the human's text becomes the ToolMessage itself (no tool runs)
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware

from ai_agent_template.agent.middleware.risk_classifier import is_blocked_tool

logger = logging.getLogger(__name__)

ASK_USER_TOOL = "ask_user"


def build_hitl_middleware(
    tool_names: list[str],
    *,
    description_prefix: str = "Approval needed",
) -> HumanInTheLoopMiddleware:
    """Build the approval policy for the given toolset.

    A tool absent from `interrupt_on` is auto-approved, so read-only tools
    stay frictionless.

    Args:
        tool_names: every tool bound to the agent.
        description_prefix: shown to the user above the pending action.
    """
    interrupt_on: dict[str, dict] = {
        # The human IS the tool: their reply comes back as the tool result.
        ASK_USER_TOOL: {"allowed_decisions": ["respond"]},
    }

    gated: set[str] = set()
    for name in tool_names:
        if name == ASK_USER_TOOL or is_blocked_tool(name):
            # Blocked tools are never bound; never offer approval for them.
            continue
        if not name.startswith("get_"):
            interrupt_on[name] = {"allowed_decisions": ["approve", "reject"]}
            gated.add(name)
        # get_* tools: absent => auto-approved

    logger.info(
        "build_hitl_middleware: gating %d of %d tools",
        len(gated),
        len(tool_names),
    )
    return HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        description_prefix=description_prefix,
    )

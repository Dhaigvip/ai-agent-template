"""Risk classification for tool calls.

Re-derived for the demo project/task domain — the reference's policy hinges
entirely on a product-specific workspace/changeset workflow (create_workspace
as a precondition, commit_workspace as an irreversible publish action),
neither of which exists here. This template's tool names come from
mcp-server-template's generator (get_*/create_*/update_*), so risk is
classified by that naming convention instead of hardcoded tool names.
"""

from __future__ import annotations

from ai_agent_template.agent.utils.types import RiskInfo

# Tools the agent must NEVER call. Empty by default — this demo domain has
# no irreversible, high-blast-radius action to guard (the reference's
# commit_workspace published ALL pending changes to a live branch; nothing
# in the demo schema is analogous). If your own domain has an action like
# that, add its tool name here — it gets excluded from the toolset at
# graph-build time AND hard-denied at execution, not just gated behind
# approval, because offering approval would imply the agent could initiate
# it at all.
BLOCKED_TOOLS: frozenset[str] = frozenset()


def is_blocked_tool(tool_name: str) -> bool:
    """True if the agent is forbidden from calling this tool."""
    return tool_name in BLOCKED_TOOLS


def classify_risk(tool_name: str) -> RiskInfo:
    """Classify by naming convention: get_* is read-only, everything else
    (create_*/update_*/...) is a mutation and requires approval."""
    if tool_name.startswith("get_"):
        return RiskInfo(
            level="readonly",
            label="Auto-approved",
            color="blue",
            emoji="🔍",
            requires_approval=False,
            default_approve=True,
        )

    return RiskInfo(
        level="medium",
        label="Mutation",
        color="yellow",
        emoji="✏️",
        requires_approval=True,
        default_approve=False,
        warning="This will change data. Review before approving.",
    )

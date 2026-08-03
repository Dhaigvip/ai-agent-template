"""Helpers for converting MCP tools into LangChain tools.

Schema conversion uses the adapter-provided JSON Schema. Tool execution stays
on `MCPClient.call_tool()` so runtime caching and retry behavior remains
centralized.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import Tool as MCPTool

from ai_agent_template.agent.utils.content import to_llm_tool_content
from ai_agent_template.mcp.client import MCPClient

logger = logging.getLogger(__name__)


def _coerce_json_container(v: Any) -> Any:
    """If the model stringified a list/dict param (e.g. modules='[{"name":"vip1"}]'),
    parse it back. Models occasionally JSON-encode array/object args even when the
    schema says array — this coerces it so the MCP call validates instead of
    erroring + retrying.
    """
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("[", "{"):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _schema_container_fields(input_schema: dict[str, Any]) -> set[str]:
    """Field names whose JSON-schema type is array/object (incl. the anyOf-wrapped-optional
    shape, e.g. {"anyOf": [{"type":"array"}, {"type":"null"}]}).

    Used to scope `_coerce_json_container` to the fields that can actually be stringified
    containers — with args_schema now the raw MCP JSON Schema, there's no per-field
    pydantic BeforeValidator hook to attach the coercion to, so it runs generically in
    the call coroutine below, gated by this to avoid touching plain string fields.
    """
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    fields: set[str] = set()
    for field_name, field_schema in properties.items():
        fs = field_schema if isinstance(field_schema, dict) else {}
        t = fs.get("type")
        if t is None:
            for sub in fs.get("anyOf") or fs.get("oneOf") or []:
                if isinstance(sub, dict) and sub.get("type") not in (None, "null"):
                    t = sub.get("type")
                    break
        if t in ("array", "object"):
            fields.add(field_name)
    return fields


def create_synthetic_tools() -> list[StructuredTool]:
    """Synthetic tools — handled by HITL middleware, never sent to MCP.

    Must be included in the bound model's tool list so the LLM knows they exist.
    """
    from pydantic import BaseModel, Field

    class AskUserInput(BaseModel):
        question: str = Field(description="The exact question to ask the user.")
        reason: str | None = Field(default=None, description="Why this information is needed.")

    async def _ask_user_stub(**_kwargs: Any) -> str:
        # Never executed — intercepted by the HITL "respond" decision
        # (see agent/middleware/hitl_policy.py's ASK_USER_TOOL).
        return "This tool is handled client-side."

    return [
        StructuredTool.from_function(
            coroutine=_ask_user_stub,
            name="ask_user",
            description=(
                "Ask the user for a value that is a mandatory parameter of the next tool call "
                "and cannot be defaulted or inferred from context. "
                "If a tool has optional parameters, skip them and call the tool without them. "
                "If you can make a reasonable attempt with the information available, do so — "
                "only use this tool when a required parameter has no value and the tool call would fail without it."
            ),
            args_schema=AskUserInput,
        )
    ]


def _is_mcp_error_result(result: Any) -> bool:
    """True if the raw MCP result indicates a tool-level failure.

    The MCP protocol's own `isError` field is the correct, generic signal
    here — confirmed against a real failing call: FastMCP's error-handling
    middleware sets it correctly on the CallToolResult even when the
    failure happens deep in the backend (e.g. the demo GraphQL backend
    being unreachable). No text-prefix sniffing needed — that was a
    convention specific to the reference's own backend error messages,
    which this template doesn't have."""
    if result is None:
        return True
    if isinstance(result, dict):
        return bool(result.get("isError"))
    return bool(getattr(result, "isError", False))


def mcp_tool_to_langchain(tool: MCPTool, mcp: MCPClient) -> StructuredTool:
    """Convert an MCP tool to a LangChain StructuredTool.

    Schema comes from langchain_mcp_adapters (raw JSON-Schema passthrough); the
    adapter-built tool's own coroutine is discarded and replaced with one that
    calls MCPClient.call_tool(), preserving the read-cache/retry behavior that
    lives there.

    handle_tool_error/handle_validation_error=True so a genuine failure (MCP
    isError=True, or bad args from the model) comes back as a ToolMessage
    with status="error" — the signal wire.py's _is_error() actually checks —
    instead of LangChain's default of re-raising, which surfaces as an
    unhelpfully generic "Internal error" with no error status set at all.
    """
    name = tool.name
    input_schema: dict[str, Any] = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    container_fields = _schema_container_fields(input_schema)

    adapted = convert_mcp_tool_to_langchain_tool(mcp.session, tool, server_name="mcp")

    async def _call(**kwargs: Any) -> str:
        args = {
            k: (_coerce_json_container(v) if k in container_fields else v)
            for k, v in kwargs.items()
        }
        try:
            result = await mcp.call_tool(name, args)
        except Exception as err:
            raise ToolException(str(err)) from err

        text = to_llm_tool_content(result)
        if _is_mcp_error_result(result):
            raise ToolException(text)
        return text

    return StructuredTool(
        name=name,
        description=adapted.description,
        args_schema=adapted.args_schema,
        coroutine=_call,
        handle_tool_error=True,
        handle_validation_error=True,
    )

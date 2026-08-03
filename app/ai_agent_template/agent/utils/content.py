"""Content utilities for message and tool-result normalization."""

from __future__ import annotations

import json
from typing import Any


def stringify_safe(value: Any) -> str:
    try:
        return json.dumps(value)
    except Exception:
        if hasattr(value, "model_dump"):
            try:
                return json.dumps(value.model_dump(), default=str)
            except Exception:
                pass
        return str(value)


def to_llm_tool_content(result: Any) -> str:
    """Convert a raw MCP tool result to a string for the LLM.

    Handles both shapes returned across MCP SDK versions:
      - a CallToolResult pydantic object (result.content is a list of content blocks)
      - a plain dict with a "content" list

    Full content passes through. Downstream context middleware is responsible
    for managing transcript size.
    """
    content = (
        result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    )

    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"])
            else:
                # pydantic content block (e.g. TextContent) — has .type and .text
                if getattr(c, "type", None) == "text":
                    text = getattr(c, "text", "")
                    if text:
                        parts.append(text)
        if parts:
            return "\n".join(parts)

    return stringify_safe(result)


def extract_text(content: Any) -> str:
    """Extract plain text from LangChain message content (str or list of blocks)."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            c if isinstance(c, str) else c.get("text", "")
            for c in content
            if not (isinstance(c, dict) and c.get("type", "").lower() == "reasoning_content")
        )
    else:
        return ""

    # Strip wrapping double-quotes left by JSON stringification of simple strings
    # e.g. '""' -> '', '"Hello"' -> 'Hello' (but leave interior quotes untouched)
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        inner = text[1:-1]
        if '"' not in inner:
            return inner

    return text

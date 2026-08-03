"""Bedrock Knowledge Base retrieval — read-only, bound as a tool.

Read-only by construction: only ever calls bedrock-agent-runtime's Retrieve
API, never StartIngestionJob or any write operation. On-demand only — the
LLM calls kb_retrieve() itself when it needs guidance beyond what's already
in its system prompt or MCP-sourced tools; nothing here injects KB content
into a turn automatically.

The reference's actual KB content (product-specific methodology decks)
does not port — this is the retrieval mechanism only. Point AGENT_KB_ID at
whatever knowledge base your own deployment populates and syncs; how that
KB gets populated is out of scope for this template.

Fallback behavior:
  AGENT_KB_ID not set
    -> KB entirely disabled. get_kb_client() returns None, no tool
       registered. The agent runs normally on its existing tools/prompt.

  kb_retrieve fails mid-task (transient error, service blip)
    -> One automatic retry. If both attempts fail, the tool returns a
       clear unavailability message so the LLM continues with what it
       already knows rather than stalling the turn.

  Credential/auth failure
    -> Not retried (retrying an expired credential just repeats the same
       failure). Reported once through the same degradation-warning
       channel as AgentCore persistence (agent/graph_builder.py's
       _on_persistence_degraded), so the client can show one notice
       instead of a warning per call.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import boto3
from langchain_core.tools import tool

from ai_agent_template.agent.resilient_persistence import _is_credential_error
from ai_agent_template.config import KnowledgeBaseConfig

logger = logging.getLogger(__name__)

# Singleton instance — one per process, matching the reference's own pattern
# (and this template's degradation-warning bookkeeping in graph_builder.py).
_KB_CLIENT: Optional["KBClient"] = None

DegradedCallback = Callable[[str, str], None]


def _format_retrieve_results(results: list[dict], max_chars: int) -> str:
    """Format Bedrock Retrieve results into a prompt-ready string.

    Prefixes each chunk with `[source: <uri>]` for traceability, dedupes by
    source URI (one chunk per unique document), and truncates the total to
    max_chars with a `[... truncated]` suffix.
    """
    seen_uris: set[str] = set()
    chunks: list[str] = []
    total = 0

    for result in results:
        text = (result.get("content") or {}).get("text", "").strip()
        if not text:
            continue

        location = result.get("location") or {}
        if location.get("type") == "S3":
            uri = (location.get("s3Location") or {}).get("uri", "")
        else:
            # Other vector stores fall back to KB-provided metadata.
            uri = (result.get("metadata") or {}).get("x-amz-bedrock-kb-source-uri", "")

        if uri:
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

        label = f"[source: {uri}]" if uri else "[source: unknown]"
        chunk = f"{label}\n{text}"

        if total + len(chunk) > max_chars:
            remaining = max_chars - total - len("\n[... truncated]")
            if remaining > 0:
                chunks.append(chunk[:remaining] + "\n[... truncated]")
            break

        chunks.append(chunk)
        total += len(chunk) + 1

    return "\n\n".join(chunks)


class KBClient:
    """Read-only Bedrock KB client. Process singleton via get_kb_client()."""

    def __init__(
        self,
        kb_id: str,
        region: str,
        chunks: int,
        max_chars: int,
        profile: str | None = None,
    ) -> None:
        self._kb_id = kb_id
        self._region = region
        self._chunks = chunks
        self._max_chars = max_chars
        self._profile = profile

    def kb_retrieve_tool(self, on_degraded: DegradedCallback | None = None):
        """Return a LangChain @tool for on-demand knowledge-base retrieval.

        A plain sync @tool — langchain's tool-execution node runs sync
        tools in a thread pool automatically, so this doesn't need its own
        asyncio wrapping (boto3 has no async client anyway).
        """
        kb_id, region, profile = self._kb_id, self._region, self._profile
        n, max_chars = self._chunks, self._max_chars

        _UNAVAILABLE_MSG = (
            "The knowledge base is temporarily unavailable. Continue using "
            "your existing knowledge and available tools to complete the task."
        )

        @tool
        def kb_retrieve(query: str) -> str:
            """Retrieve supporting knowledge-base content for a specific query.

            Use this when you need guidance beyond what's already in your
            system prompt or available tools — e.g. a concept you're unsure
            about, a methodology step, or a best practice. Do not use it for
            tool discovery (tools are already loaded) or for questions you
            can answer from what you already know. Keep queries specific;
            call sparingly.
            """
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            client = session.client("bedrock-agent-runtime", region_name=region)

            for attempt in range(2):  # one retry on transient failure
                try:
                    response = client.retrieve(
                        knowledgeBaseId=kb_id,
                        retrievalQuery={"text": query},
                        retrievalConfiguration={
                            "vectorSearchConfiguration": {"numberOfResults": n}
                        },
                    )
                    results = response.get("retrievalResults", [])
                    formatted = _format_retrieve_results(results, max_chars=max_chars)
                    return formatted or "No relevant content found for this query."

                except Exception as exc:
                    if _is_credential_error(exc):
                        logger.warning("kb_retrieve: credential/auth failure - %s", exc)
                        if on_degraded:
                            on_degraded(
                                "knowledge_base",
                                "Knowledge base is disabled: AWS credentials have "
                                "expired or lack permission for Bedrock KB retrieval.",
                            )
                        return _UNAVAILABLE_MSG
                    if attempt == 0:
                        logger.warning("kb_retrieve: transient error, retrying - %s", exc)
                        time.sleep(1)
                        continue
                    logger.warning("kb_retrieve: failed after retry - %s", exc)
                    return _UNAVAILABLE_MSG

            return _UNAVAILABLE_MSG  # unreachable, satisfies type checker

        return kb_retrieve


def get_kb_client(
    config: KnowledgeBaseConfig, *, bedrock_region: str, bedrock_profile: str | None = None
) -> KBClient | None:
    """Return the process-level KBClient singleton, or None if AGENT_KB_ID unset.

    bedrock_profile threads through AWS_PROFILE the same way agent/model.py
    and middleware/input_guardrail.py already do — boto3 will pick up
    AWS_PROFILE implicitly if it happens to be set in the process
    environment, but this template reads env vars explicitly through
    config.py rather than relying on that, same as everywhere else.

    Callers must guard:
        kb = get_kb_client(config.knowledge_base, bedrock_region=config.bedrock.region,
                            bedrock_profile=config.bedrock.profile)
        if kb:
            tools.append(kb.kb_retrieve_tool(on_degraded=...))
    """
    global _KB_CLIENT

    if not config.kb_id:
        return None

    if _KB_CLIENT is None:
        region = config.region or bedrock_region
        _KB_CLIENT = KBClient(
            kb_id=config.kb_id,
            region=region,
            chunks=config.chunks,
            max_chars=config.max_chars,
            profile=bedrock_profile,
        )
        logger.info("KB client initialized: kb_id=%s region=%s", config.kb_id, region)

    return _KB_CLIENT

"""AgentCore Memory helpers with correct namespace handling.

No-op when MemoryConfig.memory_id is unset — get_memory_store()
(graph_builder.py) hands these functions a plain InMemoryStore in that
case, which doesn't support the AgentCore-specific namespace/SearchOp
scheme used here anyway. The real backend replaces nothing here except the
store instance and config values; these functions' shape doesn't change
between the no-op and real paths.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.store.base import BaseStore, SearchOp

from ai_agent_template.config import MemoryConfig

logger = logging.getLogger(__name__)


def save_turn(
    *, store: BaseStore, actor_id: str, session_id: str, message: BaseMessage, memory: MemoryConfig
) -> None:
    """Save conversation turn to short-term memory. Triggers automatic
    extraction on the AgentCore side (~2 min). No-op if memory_id is unset.

    Args:
        store: LangGraph BaseStore (AgentCoreMemoryStore, or its resilient wrapper)
        actor_id: User identifier
        session_id: Conversation thread ID
        message: LangChain BaseMessage (HumanMessage, AIMessage, etc.)
        memory: MemoryConfig — no-op unless memory_id is set
    """
    if not memory.memory_id:
        return
    try:
        store.put(
            namespace=(actor_id, session_id),
            key=f"turn_{message.id or 'unknown'}",
            value={"message": message},
        )
        logger.info(
            "memory.save_turn: saved to (%s..., %s...) - extraction happens automatically",
            actor_id[:8],
            session_id[:8],
        )
    except Exception as err:
        logger.error("memory.save_turn: failed: %s", err, exc_info=True)


def search_semantic_memories(
    *,
    store: BaseStore,
    actor_id: str,
    query: str,
    top_k: int,
    min_score: float,
    memory: MemoryConfig,
) -> list[dict[str, Any]] | None:
    """Search extracted semantic memories (facts, preferences). Uses the
    strategy-specific namespace. Returns None if memory_id/semantic_strategy_id
    is unset.

    Args:
        store: LangGraph BaseStore (AgentCoreMemoryStore, or its resilient wrapper)
        actor_id: User identifier
        query: Search query string
        top_k: Maximum results to return
        min_score: Minimum relevance score (0.0-1.0)
        memory: MemoryConfig — no-op unless memory_id/semantic_strategy_id are set

    Returns:
        List of memory dicts with 'content', 'score', 'strategy' keys, or None.
    """
    if not memory.memory_id or not memory.semantic_strategy_id:
        return None
    try:
        # Namespace format: strategies/{strategy_id}/actors/{actor_id}/
        namespace_prefix = (f"strategies/{memory.semantic_strategy_id}/actors/{actor_id}",)

        logger.info(
            "memory.search_semantic_memories: namespace_prefix=%s query=%.50s",
            namespace_prefix,
            query,
        )

        search_op = SearchOp(namespace_prefix=namespace_prefix, query=query, limit=top_k)
        results = store.batch([search_op])[0]
        logger.info("memory.search_semantic_memories: %d raw result(s)", len(results))

        # Results are SearchItem objects with .score, .value, .namespace attributes
        memories: list[dict[str, Any]] = []
        for r in results:
            score = r.score if r.score is not None else 0.0
            if score < min_score:
                continue
            value = r.value or {}
            content = value.get("content") or value.get("text") or str(value)
            memories.append(
                {
                    "content": content,
                    "score": score,
                    "strategy": memory.semantic_strategy_id,
                    "namespace": "/".join(r.namespace) if r.namespace else "",
                }
            )

        logger.info(
            "memory.search_semantic_memories: %d/%d above threshold %.2f",
            len(memories),
            len(results),
            min_score,
        )
        return memories
    except Exception as err:
        logger.error("memory.search_semantic_memories: failed: %s", err, exc_info=True)
        return None

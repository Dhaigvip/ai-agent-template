"""Lifecycle hooks that run once per turn.

EntryMiddleware reads memory before the agent call.
MemoryWriteMiddleware stores a turn summary after the agent call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config

from ai_agent_template.agent.context import AgentContext
from ai_agent_template.agent.memory import save_turn, search_semantic_memories
from ai_agent_template.agent.state import AgentState
from ai_agent_template.agent.utils.content import extract_text
from ai_agent_template.agent.utils.log import agent_log
from ai_agent_template.config import MemoryConfig

logger = logging.getLogger(__name__)


def _actor_id(context: AgentContext) -> str:
    """Resolve actor id with fallback order: config, context, anonymous."""
    try:
        actor = (get_config() or {}).get("configurable", {}).get("actor_id")
    except Exception:
        actor = None
    return actor or context.user_id or "anonymous"


def _thread_id() -> str:
    try:
        return (get_config() or {}).get("configurable", {}).get("thread_id", "default_session")
    except Exception:
        return "default_session"


def _last_human_query(messages: list) -> str:
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return ""
    return last_human.content if isinstance(last_human.content, str) else str(last_human.content)


class EntryMiddleware(AgentMiddleware[AgentState]):
    """Load semantic memory and write it into side-state for this turn."""

    state_schema = AgentState

    def __init__(self, memory: MemoryConfig) -> None:
        super().__init__()
        self._memory = memory

    async def abefore_agent(self, state: AgentState, runtime: Any) -> dict | None:
        context: AgentContext = state.get("context") or AgentContext()
        query = _last_human_query(state.get("messages", []))
        store = getattr(runtime, "store", None)
        actor_id = _actor_id(context)

        async def _retrieve_memory() -> str | None:
            if store is None:
                return None
            try:
                memories = await asyncio.to_thread(
                    search_semantic_memories,
                    store=store,
                    actor_id=actor_id,
                    query=query,
                    top_k=self._memory.top_k,
                    min_score=0.3,
                    memory=self._memory,
                )
                if memories:
                    lines = [f"- {m['content']} (score: {m['score']:.2f})" for m in memories]
                    agent_log(
                        logger,
                        logging.INFO,
                        "agent.memory_retrieved",
                        user_id=actor_id,
                        count=len(lines),
                    )
                    return "<user_memory>\n" + "\n".join(lines) + "\n</user_memory>"
                return None
            except Exception as err:  # non-fatal — memory must never break a turn
                logger.warning("EntryMiddleware: memory retrieval failed: %s", err)
                return None

        memory_context = await _retrieve_memory()

        agent_log(
            logger, logging.INFO, "agent.turn_start", message_count=len(state.get("messages", []))
        )

        return {"memory_context": memory_context}


class MemoryWriteMiddleware(AgentMiddleware[AgentState]):
    """Persist a compact turn summary to long-term memory after agent execution."""

    state_schema = AgentState

    def __init__(self, memory: MemoryConfig) -> None:
        super().__init__()
        self._memory = memory

    async def aafter_agent(self, state: AgentState, runtime: Any) -> dict | None:
        context: AgentContext = state.get("context") or AgentContext()
        messages = state.get("messages", [])
        store = getattr(runtime, "store", None)
        if store is None:
            return None

        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        query = extract_text(last_human.content) if last_human else ""
        answer = extract_text(last_ai.content) if last_ai else ""

        try:
            actor_id = _actor_id(context)
            session_id = _thread_id()
            summary = f"User asked: {query[:200]}. Agent answered: {answer[:500]}."
            await asyncio.to_thread(
                save_turn,
                store=store,
                actor_id=actor_id,
                session_id=session_id,
                message=HumanMessage(content=summary),
                memory=self._memory,
            )
            agent_log(logger, logging.INFO, "agent.memory_written", user_id=actor_id)
        except Exception as err:  # non-fatal — a write failure must never break a turn
            logger.warning("MemoryWriteMiddleware: memory write failed: %s", err)

        return None

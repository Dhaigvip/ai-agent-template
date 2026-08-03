"""Graph builder for the single-agent runtime.

Assembles create_agent() on this template's AgentState with the middleware
chain. This module gives you the ReAct loop: model call -> tool calls? -> run
tools -> back to model -> ... -> no tool calls -> END, wrapped by lifecycle
(memory read/write), turn-context injection, prompt caching, and
context-overflow retry.

Middleware order is significant because before/after/wrap hooks compose:
  - PromptCacheMiddleware is OUTERMOST (prepended) so model_settings carries
    cache_control before any inner middleware sets other fields on it.
  - EntryMiddleware / MemoryWriteMiddleware bookend the turn (before_agent /
    after_agent hooks — list order for these, unlike wrap_model_call hooks,
    doesn't nest the same way, but keeping them adjacent to their natural
    position in the turn lifecycle keeps the chain readable).
  - HITL sits between EntryMiddleware and DynamicToolsMiddleware, same
    relative position as the reference — approval happens after memory is
    loaded but before ephemeral turn-context is injected into the call that
    follows resume.
  - ContextOverflowRetryMiddleware is placed to catch an overflow from the
    actual model call, not from another middleware's retry wrapping it.

Tools are bound on EVERY model call (bind-all, never a dynamic per-turn
subset). This isn't a simplification — the reference measured it: Bedrock's
cache prefix orders tools before system, so a stable full bind reads back at
a ~90% cache discount while a dynamic subset re-writes the tool-schema
tokens (a cache WRITE, not a read) on every set change. A per-query dynamic
subset would need a very high tool-set repeat rate to break even.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ai_agent_template.agent.middleware.context_overflow_middleware import (
    ContextOverflowRetryMiddleware,
)
from ai_agent_template.agent.middleware.dynamic_tools_middleware import DynamicToolsMiddleware
from ai_agent_template.agent.middleware.hitl_policy import build_hitl_middleware
from ai_agent_template.agent.middleware.lifecycle_middleware import (
    EntryMiddleware,
    MemoryWriteMiddleware,
)
from ai_agent_template.agent.middleware.prompt_cache_middleware import PromptCacheMiddleware
from ai_agent_template.agent.model import build_cache_control, create_model
from ai_agent_template.agent.resilient_persistence import ResilientCheckpointer, ResilientStore
from ai_agent_template.agent.state import AgentState
from ai_agent_template.config import AppConfig, BedrockConfig, MemoryConfig

logger = logging.getLogger(__name__)

# Module-level queue for degradation warnings (feature, message).
# AIGateway drains this after graph build / at turn start to emit warning events.
_degradation_warnings: list[tuple[str, str]] = []
_degradation_warned_features: set[str] = set()


def drain_degradation_warnings() -> list[tuple[str, str]]:
    """Return and clear any pending degradation warnings."""
    warnings = list(_degradation_warnings)
    _degradation_warnings.clear()
    return warnings


def _on_persistence_degraded(feature: str, message: str) -> None:
    """Callback invoked on credential failure. Fires only once per feature."""
    if feature in _degradation_warned_features:
        return
    _degradation_warned_features.add(feature)
    _degradation_warnings.append((feature, message))


def get_checkpointer(config: AppConfig) -> BaseCheckpointSaver:
    """Return the checkpointer for conversation state.

    1. AgentCoreMemorySaver — when AGENTCORE_MEMORY_ID is set AND
       AGENT_AGENTCORE_CHECKPOINTER_ENABLED=true (opt-in, off by default)
    2. MemorySaver — in-process fallback (dev/local; lost on restart)

    AgentCore checkpointing is opt-in and OFF by default even when
    AGENTCORE_MEMORY_ID is set: langgraph-checkpoint-aws 1.0.7 doesn't
    reliably round-trip LangGraph interrupt pending-writes with the
    langgraph version this template pins, which can break HITL resume (the
    approved tool never executes — the paused state isn't restored on
    read-back). MemorySaver handles graph state + interrupts reliably
    in-process. This does NOT affect long-term memory: cross-session
    facts/summaries live in the STORE (get_memory_store), independent of
    the checkpointer. Trade-off: no cross-restart resume of an in-progress
    conversation transcript unless explicitly opted in.
    """
    memory = config.memory
    use_agentcore = bool(memory.memory_id) and memory.checkpointer_enabled
    if use_agentcore:
        try:
            from langgraph_checkpoint_aws import AgentCoreMemorySaver  # type: ignore[import]

            saver = AgentCoreMemorySaver(memory.memory_id, region_name=config.bedrock.region)
            logger.info(
                "get_checkpointer: AgentCoreMemorySaver memory_id=%s region=%s (opt-in)",
                memory.memory_id,
                config.bedrock.region,
            )
            return ResilientCheckpointer(saver, on_degraded=_on_persistence_degraded)
        except ImportError:
            logger.warning(
                "get_checkpointer: langgraph-checkpoint-aws not installed - "
                "falling back to MemorySaver. Run: pip install langgraph-checkpoint-aws"
            )
        except Exception as err:
            logger.warning(
                "get_checkpointer: AgentCoreMemorySaver init failed (%s) - "
                "falling back to MemorySaver",
                err,
            )

    logger.info(
        "get_checkpointer: using in-process MemorySaver "
        "(AgentCore checkpointer disabled - set AGENT_AGENTCORE_CHECKPOINTER_ENABLED=true to "
        "opt in; long-term memory store is unaffected)"
    )
    return MemorySaver()


def get_memory_store(config: AppConfig) -> BaseStore:
    """Return a store for long-term cross-session memory.

    1. AgentCoreMemoryStore — when AGENTCORE_MEMORY_ID is set (AWS-managed, with semantic search)
    2. InMemoryStore — always-present in-process fallback (lost on restart, dev/local only)

    The store is always non-None so nodes can call store.asearch/aput unconditionally.
    """
    memory: MemoryConfig = config.memory
    if memory.memory_id:
        try:
            from langgraph_checkpoint_aws import AgentCoreMemoryStore  # type: ignore[import]

            store = AgentCoreMemoryStore(memory_id=memory.memory_id, region_name=config.bedrock.region)
            logger.info(
                "get_memory_store: AgentCoreMemoryStore enabled memory_id=%s region=%s",
                memory.memory_id,
                config.bedrock.region,
            )
            return ResilientStore(store, on_degraded=_on_persistence_degraded)
        except ImportError:
            logger.warning(
                "get_memory_store: langgraph-checkpoint-aws not installed - "
                "falling back to InMemoryStore. Run: pip install langgraph-checkpoint-aws"
            )
        except Exception as err:
            logger.warning(
                "get_memory_store: AgentCoreMemoryStore init failed (%s) - "
                "falling back to InMemoryStore",
                err,
            )

    logger.debug("get_memory_store: using InMemoryStore (no persistent store configured)")
    return InMemoryStore()


class BuiltGraph:
    """Bundles the compiled graph with the checkpointer/store that built it
    — callers that need stateless HITL resume or memory access don't have to
    dig either back out of the compiled graph object."""

    def __init__(
        self, graph: Any, checkpointer: BaseCheckpointSaver, memory_store: BaseStore
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.memory_store = memory_store


def _unwrap_model_cache_binding(model: BaseChatModel) -> BaseChatModel:
    """create_model() may hand us a model with cache_control already bound
    via .bind(). PromptCacheMiddleware re-applies cache_control through
    model_settings instead, so unwrap it here — keeping any OTHER bound
    kwargs intact — and avoid two competing cache_control sources."""
    bound_kwargs = dict(getattr(model, "kwargs", None) or {})
    if "cache_control" in bound_kwargs and getattr(model, "bound", None) is not None:
        underlying = model.bound
        bound_kwargs.pop("cache_control")
        return underlying.bind(**bound_kwargs) if bound_kwargs else underlying
    return model


def assemble_agent_graph(
    *,
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    bedrock: BedrockConfig,
    memory: MemoryConfig,
) -> Any:
    """Assemble create_agent on AgentState with the middleware chain."""
    from langchain.agents import create_agent

    logger.info("assemble_agent_graph: binding %d tool(s)", len(tools))

    model = _unwrap_model_cache_binding(model)

    middleware: list = [
        EntryMiddleware(memory),
        build_hitl_middleware(sorted(t.name for t in tools)),
        DynamicToolsMiddleware(),
        ContextOverflowRetryMiddleware(),
        MemoryWriteMiddleware(memory),
    ]
    cache_control = build_cache_control(bedrock)
    if cache_control is not None:
        # Outermost wrap_model_call so model_settings are set before inner
        # middleware overrides other fields; each middleware overrides a
        # disjoint set, so nesting composes cleanly.
        middleware.insert(0, PromptCacheMiddleware(cache_control))
        logger.info(
            "assemble_agent_graph: prompt caching enabled (scope=%s, ttl=%s)",
            cache_control["scope"],
            cache_control.get("ttl", "5m"),
        )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        state_schema=AgentState,
        middleware=middleware,
        checkpointer=checkpointer,
        store=store,
    )


def build_graph(
    *,
    config: AppConfig,
    tools: list[BaseTool],
    system_prompt: str,
) -> BuiltGraph:
    """Build and compile the single-agent graph.

    tools/system_prompt are taken as plain arguments here rather than
    resolved from a live source — sourcing them (from an MCP server) is a
    separate concern layered in later without this function's shape
    changing, just what its caller passes in.
    """
    model = create_model(config.bedrock)
    checkpointer = get_checkpointer(config)
    store = get_memory_store(config)
    graph = assemble_agent_graph(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        bedrock=config.bedrock,
        memory=config.memory,
    )
    logger.info("build_graph: graph compiled (%d tool(s))", len(tools))
    return BuiltGraph(graph=graph, checkpointer=checkpointer, memory_store=store)

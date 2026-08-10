"""Tests that get_checkpointer()/get_memory_store() never touch AgentCore
when provider != "bedrock", even if AGENTCORE_MEMORY_ID is set — AgentCore
fundamentally requires AWS/Bedrock regardless of that env var. No AWS calls;
langgraph_checkpoint_aws import is spied on to confirm it's never reached,
matching the project's existing "assert zero calls, not just no crash"
verification style (see TASKS.md Task 6)."""

from __future__ import annotations

import sys

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ai_agent_template.agent.graph_builder import get_checkpointer, get_memory_store
from ai_agent_template.config import AppConfig, BedrockConfig, MemoryConfig


def _fully_configured_memory_config() -> MemoryConfig:
    """memory_id AND checkpointer_enabled both set — the strongest possible
    signal that AgentCore SHOULD be used, so the provider gate is the only
    thing standing in the way when provider != bedrock."""
    return MemoryConfig(
        memory_id="fake-memory-id",
        semantic_strategy_id="fake-strategy-id",
        checkpointer_enabled=True,
    )


def test_checkpointer_uses_memorysaver_for_bedrock_without_opt_in():
    cfg = AppConfig(bedrock=BedrockConfig(region="us-east-1"), memory=MemoryConfig(memory_id="x"))
    assert isinstance(get_checkpointer(cfg), MemorySaver)


def test_checkpointer_ignores_agentcore_when_provider_is_openai():
    cfg = AppConfig(
        bedrock=BedrockConfig(region=""),
        provider="openai",
        memory=_fully_configured_memory_config(),
    )
    assert isinstance(get_checkpointer(cfg), MemorySaver)


def test_memory_store_ignores_agentcore_when_provider_is_openai():
    cfg = AppConfig(
        bedrock=BedrockConfig(region=""),
        provider="openai",
        memory=_fully_configured_memory_config(),
    )
    assert isinstance(get_memory_store(cfg), InMemoryStore)


def test_agentcore_module_never_imported_when_provider_is_openai(monkeypatch):
    """Spy on the import itself — proves the provider gate short-circuits
    before any AgentCore construction is even attempted, not just that the
    end result happens to be MemorySaver/InMemoryStore."""
    sys.modules.pop("langgraph_checkpoint_aws", None)
    called = {"import": False}

    real_import = __import__

    def spy_import(name, *args, **kwargs):
        if name == "langgraph_checkpoint_aws":
            called["import"] = True
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", spy_import)

    cfg = AppConfig(
        bedrock=BedrockConfig(region=""),
        provider="openai",
        memory=_fully_configured_memory_config(),
    )
    get_checkpointer(cfg)
    get_memory_store(cfg)

    assert called["import"] is False, "langgraph_checkpoint_aws was imported despite provider='openai'"

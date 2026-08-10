"""Tests for agent/model.py's create_model() provider dispatch.

Construction only — no live AWS or OpenAI calls anywhere in this file, per
the project's standing "no live provider calls during the build" rule,
extended to OpenAI. Both provider SDKs are core dependencies (pyproject.toml)
— no importorskip/ImportError-path testing needed, a plain `pip install -e .`
guarantees both are present.
"""

from __future__ import annotations

from ai_agent_template.agent.model import (
    _create_bedrock_model,
    _create_openai_model,
    create_model,
)
from ai_agent_template.config import AppConfig, BedrockConfig, OpenAIConfig


def test_create_model_dispatches_to_openai_by_default():
    """AppConfig() with no explicit provider is OpenAI — the zero-config default."""
    cfg = AppConfig()
    assert cfg.provider == "openai"
    model = create_model(cfg)
    assert type(model).__name__ == "ChatOpenAI"


def test_create_model_dispatches_to_bedrock_when_selected():
    cfg = AppConfig(provider="bedrock", bedrock=BedrockConfig(region="us-east-1"))
    model = create_model(cfg)
    assert type(model).__name__ == "ScopedCacheChatBedrockConverse"


def test_bedrock_model_kwargs():
    bedrock = BedrockConfig(
        region="us-east-1",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        connect_timeout=5,
        read_timeout=120,
        max_retries=3,
    )
    model = _create_bedrock_model(bedrock)
    assert model.region_name == "us-east-1"


def test_bedrock_model_id_override():
    bedrock = BedrockConfig(region="us-east-1", model_id="default-model")
    model = _create_bedrock_model(bedrock, model_id="override-model")
    assert model.model_id == "override-model"


def test_bedrock_cache_control_binding_when_enabled():
    bedrock = BedrockConfig(region="us-east-1", prompt_cache_enabled=True, prompt_cache_scope="full")
    model = _create_bedrock_model(bedrock)
    # bound model: cache_control present in the bound kwargs
    assert getattr(model, "kwargs", {}).get("cache_control") is not None


def test_bedrock_cache_control_absent_when_disabled():
    bedrock = BedrockConfig(region="us-east-1", prompt_cache_enabled=False)
    model = _create_bedrock_model(bedrock)
    assert not getattr(model, "kwargs", {})


def test_openai_model_kwargs():
    openai_cfg = OpenAIConfig(api_key="sk-test", model_id="gpt-4o", timeout=45, max_retries=5)
    model = _create_openai_model(openai_cfg)
    assert model.model_name == "gpt-4o"
    assert model.max_retries == 5


def test_openai_model_id_override():
    openai_cfg = OpenAIConfig(api_key="sk-test", model_id="default-model")
    model = _create_openai_model(openai_cfg, model_id="gpt-4o-override")
    assert model.model_name == "gpt-4o-override"

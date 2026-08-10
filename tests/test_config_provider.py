"""Tests for provider selection and the conditional AWS_REGION requirement
in config.py. Env-var monkeypatching only — no network calls."""

from __future__ import annotations

import importlib

import pytest


def _reload_config(monkeypatch, env: dict[str, str | None]):
    """Reload ai_agent_template.config with a controlled environment so
    the real .env file on disk (which may set AWS_REGION) never leaks in."""
    for key in (
        "AWS_REGION",
        "AGENT_MODEL_PROVIDER",
        "OPENAI_API_KEY",
        "AGENT_KB_ID",
        "AGENTCORE_CODE_INTERPRETER_ID",
        "AGENT_OPENAI_MODEL_ID",
        "AGENT_OPENAI_BASE_URL",
        "AGENT_OPENAI_TIMEOUT",
        "AGENT_OPENAI_MAX_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is not None:
            monkeypatch.setenv(key, value)

    # Patch dotenv's own load_dotenv, not config's already-bound name — reload()
    # re-executes "from dotenv import load_dotenv", which would otherwise
    # rebind past a patch applied to the already-imported config module and
    # re-populate AWS_REGION etc. from the real .env file on disk.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)

    import ai_agent_template.config as config_module

    importlib.reload(config_module)
    return config_module


def test_provider_defaults_to_openai(monkeypatch):
    """OpenAI is the zero-config default — Bedrock is opt-in for production."""
    cfg_mod = _reload_config(monkeypatch, {})
    cfg = cfg_mod.load_config()
    assert cfg.provider == "openai"


def test_invalid_provider_warns_and_defaults(monkeypatch, caplog):
    cfg_mod = _reload_config(monkeypatch, {"AGENT_MODEL_PROVIDER": "not-a-real-provider"})
    with caplog.at_level("WARNING"):
        cfg = cfg_mod.load_config()
    assert cfg.provider == "openai"
    assert any("Unrecognized AGENT_MODEL_PROVIDER" in r.message for r in caplog.records)


def test_region_required_for_bedrock(monkeypatch):
    cfg_mod = _reload_config(monkeypatch, {"AGENT_MODEL_PROVIDER": "bedrock"})
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        cfg_mod.load_config()


def test_region_not_required_for_openai_alone(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch, {"AGENT_MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    )
    cfg = cfg_mod.load_config()
    assert cfg.provider == "openai"
    assert cfg.bedrock.region == ""


def test_region_required_for_openai_plus_kb(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch,
        {
            "AGENT_MODEL_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "AGENT_KB_ID": "kb-123",
        },
    )
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        cfg_mod.load_config()


def test_region_required_for_openai_plus_code_execution(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch,
        {
            "AGENT_MODEL_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "AGENTCORE_CODE_INTERPRETER_ID": "interp-123",
        },
    )
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        cfg_mod.load_config()


def test_openai_config_populates_from_env(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch,
        {
            "AGENT_MODEL_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key",
            "AGENT_OPENAI_MODEL_ID": "gpt-4o",
            "AGENT_OPENAI_BASE_URL": "https://example.com/v1",
            "AGENT_OPENAI_TIMEOUT": "45",
            "AGENT_OPENAI_MAX_RETRIES": "5",
        },
    )
    cfg = cfg_mod.load_config()
    assert cfg.openai.api_key == "sk-test-key"
    assert cfg.openai.model_id == "gpt-4o"
    assert cfg.openai.base_url == "https://example.com/v1"
    assert cfg.openai.timeout == 45
    assert cfg.openai.max_retries == 5


def test_openai_config_defaults_when_unset(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch, {"AGENT_MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    )
    cfg = cfg_mod.load_config()
    assert cfg.openai.model_id == "gpt-4o-mini"
    assert cfg.openai.base_url is None
    assert cfg.openai.timeout == 60
    assert cfg.openai.max_retries == 2


def test_guardrail_blocklist_terms_from_env(monkeypatch):
    cfg_mod = _reload_config(
        monkeypatch,
        {"AWS_REGION": "us-east-1", "AGENT_GUARDRAIL_BLOCKLIST": "foo,bar"},
    )
    cfg = cfg_mod.load_config()
    assert cfg.guardrail.blocklist_terms == "foo,bar"

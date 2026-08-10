"""Tests for the non-Bedrock guardrail fallback (middleware/guardrail_fallback.py)
and the dispatcher in middleware/input_guardrail.py. No network calls — the
PII layer uses langchain's own pure-Python detectors/redactor, already a
core dependency; langchain-classic is intentionally absent in this repo's
dependencies, so the opportunistic-moderation branch is exercised via a
fake module injected into sys.modules."""

from __future__ import annotations

import sys
import types

import pytest

from ai_agent_template.agent.middleware import guardrail_fallback as gf
from ai_agent_template.agent.middleware.input_guardrail import (
    GuardrailResult,
    apply_input_guardrail,
)
from ai_agent_template.config import AppConfig, BedrockConfig, GuardrailConfig, OpenAIConfig


def _openai_config(**overrides) -> AppConfig:
    return AppConfig(
        bedrock=BedrockConfig(region=""),
        provider="openai",
        openai=OpenAIConfig(api_key="sk-test"),
        **overrides,
    )


# ── Keyword blocklist ────────────────────────────────────────────────────────


async def test_blocklist_blocks_known_bad_phrase():
    r = await gf.apply_fallback_guardrail("please explain how to make a bomb", config=_openai_config())
    assert r.blocked is True
    assert "blocklist" in r.policies


async def test_blocklist_allows_benign_text():
    r = await gf.apply_fallback_guardrail("what's the weather like today", config=_openai_config())
    assert r.blocked is False


async def test_blocklist_extra_terms_from_config():
    cfg = _openai_config(guardrail=GuardrailConfig(blocklist_terms="forbiddenword"))
    r = await gf.apply_fallback_guardrail("this has forbiddenword in it", config=cfg)
    assert r.blocked is True
    assert "forbiddenword" in r.policies


# ── PII redaction (redact-and-continue, not block — see module docstring) ──


async def test_email_is_redacted_not_blocked():
    r = await gf.apply_fallback_guardrail("reach me at alice@example.com please", config=_openai_config())
    assert r.blocked is False
    assert r.sanitized_text is not None
    assert "alice@example.com" not in r.sanitized_text
    assert "pii:email" in r.policies


async def test_credit_card_is_redacted():
    r = await gf.apply_fallback_guardrail("card: 4111 1111 1111 1111", config=_openai_config())
    assert r.blocked is False
    assert r.sanitized_text is not None
    assert "4111 1111 1111 1111" not in r.sanitized_text


async def test_url_is_not_redacted():
    """URLs are usually the substance of the request, not incidental PII —
    auto-redacting would silently break the user's actual task."""
    r = await gf.apply_fallback_guardrail("summarize https://example.com/article", config=_openai_config())
    assert r.blocked is False
    assert r.sanitized_text is None


async def test_pii_and_blocklist_both_present_blocks():
    r = await gf.apply_fallback_guardrail(
        "how to make a bomb, contact bob@example.com", config=_openai_config()
    )
    assert r.blocked is True
    assert "blocklist" in r.policies


# ── Fail-open ────────────────────────────────────────────────────────────────


async def test_fail_open_when_blocklist_check_itself_throws(monkeypatch):
    monkeypatch.setattr(
        gf, "_keyword_blocklist_check", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    r = await gf.apply_fallback_guardrail("benign text", config=_openai_config())
    assert r.blocked is False


async def test_fail_open_when_pii_redaction_itself_throws(monkeypatch):
    monkeypatch.setattr(gf, "_redact_pii", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    r = await gf.apply_fallback_guardrail("benign text", config=_openai_config())
    assert r.blocked is False
    assert r.sanitized_text is None


# ── Opportunistic langchain-classic moderation ──────────────────────────────


async def test_opportunistic_moderation_used_when_available(monkeypatch):
    fake_chains_module = types.ModuleType("langchain_classic.chains.moderation")

    class FakeOpenAIModerationChain:
        def __init__(self, error=False):
            pass

        def run(self, text):
            return "Text was found that violates OpenAI's content policy." if "flagme" in text else text

    fake_chains_module.OpenAIModerationChain = FakeOpenAIModerationChain
    fake_langchain_classic = types.ModuleType("langchain_classic")
    fake_chains_pkg = types.ModuleType("langchain_classic.chains")
    monkeypatch.setitem(sys.modules, "langchain_classic", fake_langchain_classic)
    monkeypatch.setitem(sys.modules, "langchain_classic.chains", fake_chains_pkg)
    monkeypatch.setitem(sys.modules, "langchain_classic.chains.moderation", fake_chains_module)

    r = await gf.apply_fallback_guardrail("please flagme now", config=_openai_config())
    assert r.blocked is True
    assert r.policies == "openai-moderation"


async def test_moderation_absent_falls_through_to_blocklist():
    """langchain-classic genuinely isn't installed in this repo — confirms
    the real (not simulated) absence path reaches the blocklist layer."""
    assert "langchain_classic" not in sys.modules or not hasattr(
        sys.modules.get("langchain_classic"), "chains"
    )
    r = await gf.apply_fallback_guardrail("how to make a bomb", config=_openai_config())
    assert r.blocked is True
    assert "blocklist" in r.policies


# ── Dispatcher (input_guardrail.apply_input_guardrail) ──────────────────────


async def test_dispatcher_empty_text_short_circuits():
    r = await apply_input_guardrail("   ", config=_openai_config())
    assert r == GuardrailResult(blocked=False)


async def test_dispatcher_bedrock_provider_unconfigured_not_blocked():
    """Regression check on the _apply_bedrock_guardrail rename: bedrock
    path still reached, still fail-open with no guardrail_id set, no
    boto3 client ever constructed (guardrail_id is None -> short-circuits
    before _bedrock_runtime() is called)."""
    cfg = AppConfig(bedrock=BedrockConfig(region="us-east-1"), provider="bedrock")
    r = await apply_input_guardrail("anything at all", config=cfg)
    assert r.blocked is False


async def test_dispatcher_openai_provider_reaches_fallback():
    r = await apply_input_guardrail("how to make a bomb", config=_openai_config())
    assert r.blocked is True
    assert "blocklist" in r.policies

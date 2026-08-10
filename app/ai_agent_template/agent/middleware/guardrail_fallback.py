"""Fallback input guardrail for non-Bedrock providers.

There's no LangChain-native equivalent to Bedrock's full ApplyGuardrail
(denied topics, prompt-attack detection, general content moderation) to
port here — checked both the reference project (its guardrail code is
100% Bedrock/boto3, nothing provider-agnostic exists, confirmed again
against a fresh read) and langchain.chains.moderation.OpenAIModerationChain
(the obvious candidate for "inbuilt LangChain guardrails"), which was
moved out of core `langchain` into a separate `langchain-classic` package
this project doesn't depend on. That part is genuinely new design work,
same conclusion CLAUDE.md predicted.

One piece IS a real, first-class LangChain built-in though:
langchain.agents.middleware.pii's PII detectors (detect_email,
detect_credit_card, detect_mac_address) and its apply_strategy() redactor —
the same functions PIIMiddleware itself is built from. They're plain
`str -> list[PIIMatch]` / `(str, matches, strategy) -> str` functions, no
AgentMiddleware/graph machinery required, so they run at this template's
existing pre-graph gateway check point exactly like the Bedrock path does
— the middleware CLASS runs `before_model` inside the graph (after MCP
tools are already sourced), which would contradict this template's
deliberate "cheapest possible short-circuit, before the graph is even
built" guardrail placement (see docs GUARDRAILS.md / input_guardrail.py);
the underlying functions have no such constraint. Already part of the
`langchain` core dependency this project has — zero new package.

Only email/credit_card/mac_address are checked, not the other two
PIIMiddleware ships (url, ip): those are usually the actual substance of a
chat request (a user pasting a URL almost always wants the agent to act on
it), so auto-redacting them would silently break the request rather than
protect anything. email/credit_card/mac_address are much more reliably
*incidental* sensitive data.

Three layers, in order:
1. Opportunistic langchain-classic moderation (provider=="openai" only,
   only if installed) — general content moderation when available. Blocks.
2. LangChain's built-in PII detection + redaction — real, provider-
   agnostic, always available. REDACTS and continues rather than blocking:
   this template's own reference docs (GUARDRAILS.md) explicitly recommend
   "Anonymize rather than Block for PII so the conversation can continue
   with masked values" — matched here via GuardrailResult.sanitized_text,
   which the gateway substitutes for the raw user message before it enters
   the graph.
3. A small built-in keyword blocklist — the last-resort net for content
   categories the PII layer doesn't cover (violence/self-harm phrasing),
   always available, zero dependencies. Runs against the PII-redacted text
   so a blocked phrase is still caught even if it happened to sit next to
   something that got redacted.

Same fail-open contract as the Bedrock path (see input_guardrail.py /
Known Pitfall #1): any error in any of the three layers returns
blocked=False (with no redaction applied) rather than raising. Moderation
must never take chat down.
"""

from __future__ import annotations

import asyncio
import logging
import re

from langchain.agents.middleware._redaction import apply_strategy
from langchain.agents.middleware.pii import detect_credit_card, detect_email, detect_mac_address

from ai_agent_template.agent.middleware.input_guardrail import GuardrailResult
from ai_agent_template.config import AppConfig

logger = logging.getLogger(__name__)

# Deliberately NOT url/ip — see module docstring. These three are reliably
# incidental sensitive data rather than the substance of a request.
_PII_DETECTORS: tuple[tuple[str, object], ...] = (
    ("email", detect_email),
    ("credit_card", detect_credit_card),
    ("mac_address", detect_mac_address),
)

# Deliberately small — a last-resort net, not a moderation product. Extend
# per-deployment via AGENT_GUARDRAIL_BLOCKLIST rather than growing this list.
_BUILTIN_BLOCKLIST: tuple[str, ...] = (
    "kill yourself",
    "make a bomb",
    "how to make a bomb",
    "build a weapon",
)

_BLOCKED_MESSAGE = "Your request was blocked by the content policy. Please rephrase and try again."


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """Detect + redact email/credit_card/mac_address. Returns (possibly
    redacted) text and the list of PII types found, empty if none."""
    working = text
    found: list[str] = []
    for pii_type, detector in _PII_DETECTORS:
        matches = detector(working)  # type: ignore[operator]
        if not matches:
            continue
        found.append(pii_type)
        working = apply_strategy(working, matches, "redact")
    return working, found


def _compile_terms(extra_terms: str | None) -> tuple[str, ...]:
    extra = tuple(
        t.strip().lower() for t in (extra_terms or "").split(",") if t.strip()
    )
    return _BUILTIN_BLOCKLIST + extra


def _keyword_blocklist_check(text: str, extra_terms: str | None) -> GuardrailResult | None:
    """Case-insensitive substring match against the built-in list plus any
    operator-supplied extra terms. Returns None (no hit) or a blocked
    GuardrailResult. Pure Python — cannot raise on a missing credential,
    but the caller still wraps this in try/except (Known Pitfall #1
    discipline applies even to a path that "shouldn't" fail)."""
    lowered = text.lower()
    hits = [term for term in _compile_terms(extra_terms) if term in lowered]
    if not hits:
        return None

    policies = ", ".join(f"blocklist:{re.sub(r'[^a-z0-9]+', '-', h)}" for h in hits)
    logger.warning("guardrail_fallback.blocked: policies=%s", policies)
    return GuardrailResult(blocked=True, message=_BLOCKED_MESSAGE, policies=policies)


async def _try_opportunistic_moderation(text: str) -> GuardrailResult | None:
    """Return a result if langchain-classic's OpenAIModerationChain is
    installed and usable; None if it's absent, errors, or doesn't flag the
    text, so the caller falls through to the PII/blocklist layers. Absence
    is normal — logged at debug, not warning."""
    try:
        from langchain_classic.chains.moderation import OpenAIModerationChain
    except ImportError:
        logger.debug("guardrail_fallback: langchain-classic not installed - using PII/blocklist only")
        return None

    try:
        chain = OpenAIModerationChain(error=False)
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: chain.run(text)
        )
    except Exception as err:  # fail-open — an unusable moderation chain falls through, not raises
        logger.debug("guardrail_fallback: OpenAIModerationChain call failed (%s) - falling through", err)
        return None

    if result != text:
        # OpenAIModerationChain (error=False) returns a fixed rejection
        # string when flagged, the original text unchanged otherwise.
        logger.warning("guardrail_fallback.blocked: policies=openai-moderation")
        return GuardrailResult(blocked=True, message=_BLOCKED_MESSAGE, policies="openai-moderation")
    return None


async def apply_fallback_guardrail(text: str, *, config: AppConfig) -> GuardrailResult:
    """Non-Bedrock guardrail dispatch. Caller (input_guardrail.py) already
    handles the empty/whitespace-only short-circuit."""
    if config.provider == "openai":
        try:
            result = await _try_opportunistic_moderation(text)
            if result is not None:
                return result
        except Exception as err:  # belt-and-suspenders — fail-open even on an unexpected bug here
            logger.warning("guardrail_fallback: opportunistic moderation errored (%s) - using PII/blocklist", err)

    working_text = text
    pii_types: list[str] = []
    try:
        working_text, pii_types = _redact_pii(text)
        if pii_types:
            logger.info("guardrail_fallback.redacted: types=%s", ",".join(pii_types))
    except Exception as err:  # fail-open — redaction failure must not block or corrupt the turn
        logger.warning("guardrail_fallback: PII redaction failed (%s) - passing text through unredacted", err)
        working_text = text

    try:
        blocklist_hit = _keyword_blocklist_check(working_text, config.guardrail.blocklist_terms)
        if blocklist_hit is not None:
            return blocklist_hit
    except Exception as err:  # fail-open — moderation must never take chat down
        logger.warning("guardrail_fallback: blocklist check failed (%s) - allowing request", err)

    if pii_types:
        return GuardrailResult(
            blocked=False,
            sanitized_text=working_text,
            policies=", ".join(f"pii:{t}" for t in pii_types),
        )
    return GuardrailResult(blocked=False)

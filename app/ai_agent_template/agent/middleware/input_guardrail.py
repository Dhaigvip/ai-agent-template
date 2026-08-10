"""Input-only guardrail — validates the user's query once per turn.

Runs separately from model inference: called on the user's text before
graph execution, not as part of the create_agent middleware chain (there's
no per-model-call hook for it — one guardrail check per turn, not one per
ReAct iteration, is the point). The gateway is the actual call site.

apply_input_guardrail() dispatches on config.provider: "bedrock" reaches
_apply_bedrock_guardrail() below (the original Bedrock ApplyGuardrail path,
unchanged); anything else reaches middleware/guardrail_fallback.py — a
keyword blocklist plus an opportunistic LangChain-ecosystem moderation
chain when one happens to be installed. Both paths are fail-open: if the
guardrail is unconfigured or errors (e.g. credentials), the turn proceeds.
Moderation must never take chat down.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ai_agent_template.config import AppConfig, GuardrailConfig

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    blocked: bool
    message: str = ""  # guardrail's configured block message, shown to the user
    policies: str = ""  # which policies tripped (for logging/diagnostics)
    # Set only by the non-Bedrock PII fallback layer (guardrail_fallback.py):
    # the turn is NOT blocked, but the caller must substitute this for the
    # raw user message before it enters the graph — matches the reference's
    # own documented guidance (GUARDRAILS.md) to anonymize PII rather than
    # hard-block, so the conversation continues with masked values. Always
    # None on the Bedrock path (ApplyGuardrail's PII policy is Anonymize/
    # Block configured in AWS, not surfaced back through this dataclass).
    sanitized_text: str | None = None


_client = None


def _bedrock_runtime(region: str, profile: str | None):
    """Lazily build a bedrock-runtime client (respects an explicit AWS profile/region)."""
    global _client
    if _client is None:
        import boto3

        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        _client = session.client("bedrock-runtime", region_name=region)
    return _client


def _summarize_assessments(assessments: list) -> str:
    """Compact 'which policy tripped' summary for logs."""
    hits: list[str] = []
    for a in assessments or []:
        for t in (a.get("topicPolicy") or {}).get("topics", []) or []:
            hits.append(f"topic:{t.get('name')}")
        for f in (a.get("contentPolicy") or {}).get("filters", []) or []:
            hits.append(f"content:{f.get('type')}")
        for w in (a.get("wordPolicy") or {}).get("customWords", []) or []:
            hits.append(f"word:{w.get('match')}")
        for w in (a.get("wordPolicy") or {}).get("managedWordLists", []) or []:
            hits.append(f"wordlist:{w.get('type')}")
        for p in (a.get("sensitiveInformationPolicy") or {}).get("piiEntities", []) or []:
            hits.append(f"pii:{p.get('type')}")
    return ", ".join(dict.fromkeys(hits)) or "intervened"


async def _apply_bedrock_guardrail(
    text: str, *, guardrail: GuardrailConfig, region: str, profile: str | None = None
) -> GuardrailResult:
    """Validate the user's query against the configured Bedrock guardrail (INPUT source only).

    Returns GuardrailResult(blocked=False) when no guardrail is configured, the text is
    empty, or the API errors (fail-open).
    """
    if not guardrail.guardrail_id or not text or not text.strip():
        return GuardrailResult(blocked=False)

    try:
        client = _bedrock_runtime(region, profile)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.apply_guardrail(
                guardrailIdentifier=guardrail.guardrail_id,
                guardrailVersion=guardrail.guardrail_version,
                source="INPUT",
                content=[{"text": {"text": text}}],
            ),
        )
    except Exception as err:  # fail-open — never break a turn on moderation failure
        logger.warning("input_guardrail: apply_guardrail failed (%s) - allowing request", err)
        return GuardrailResult(blocked=False)

    if resp.get("action") != "GUARDRAIL_INTERVENED":
        return GuardrailResult(blocked=False)

    outputs = resp.get("outputs") or []
    message = (outputs[0].get("text") if outputs else "") or (
        "Your request was blocked by the content policy. Please rephrase and try again."
    )
    policies = _summarize_assessments(resp.get("assessments") or [])
    logger.warning("input_guardrail.blocked: policies=%s", policies)
    return GuardrailResult(blocked=True, message=message, policies=policies)


async def apply_input_guardrail(text: str, *, config: AppConfig) -> GuardrailResult:
    """Validate the user's query once per turn — the single call site every
    caller uses regardless of provider. Dispatches to the Bedrock guardrail
    above when config.provider == "bedrock"; otherwise to the fallback
    guardrail (middleware/guardrail_fallback.py). Empty/whitespace-only
    text short-circuits before either path — matches the "no meaningful
    query" behavior the Bedrock path already had.
    """
    if not text or not text.strip():
        return GuardrailResult(blocked=False)

    if config.provider == "bedrock":
        return await _apply_bedrock_guardrail(
            text,
            guardrail=config.guardrail,
            region=config.bedrock.region,
            profile=config.bedrock.profile,
        )

    from ai_agent_template.agent.middleware.guardrail_fallback import apply_fallback_guardrail

    return await apply_fallback_guardrail(text, config=config)

"""
Environment-variable-driven configuration for ai-agent-template.

Mirrors mcp-server-template/config.py's shape on purpose: plain dataclasses,
one load_config() entry point, .env loaded once via python-dotenv. The two
templates should feel like the same project.

Naming: unprefixed vars are server-level (HOST, PORT, LOG_LEVEL) or standard
AWS SDK names (AWS_REGION, AWS_PROFILE); AGENT_* is this project's own
prefix, used throughout instead of any product-specific prefix.

Feature flags default OFF. Every flagged module (document ingestion, code
execution, orchestration, knowledge base, AgentCore memory, observability)
must leave the agent working end-to-end when its flag is unset — flipping a
flag on is what should be allowed to fail loudly if its module isn't wired
up yet, not the other way around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ai_agent_template.mcp.client import MCPClientConfig

# app/ai_agent_template/config.py -> .venv-sibling repo root is 3 parents up.
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    """Tolerant bool parsing: env files pick up inline comments and stray
    whitespace in practice. An unrecognized value warns and falls back to
    the default rather than crashing the server over a typo in a flag
    nobody's actively using."""
    if raw is None:
        return default
    value = raw.split("#", 1)[0].strip().lower()
    if not value:
        return default
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    import logging

    logging.getLogger(__name__).warning(
        "Unrecognized bool value %r - using default %r", raw, default
    )
    return default


_VALID_CACHE_SCOPES = {"full", "prefix_only"}


def _parse_cache_scope(raw: str | None, *, default: str = "full") -> str:
    """Tolerant cache-scope parsing — same inline-comment/whitespace tolerance
    as _parse_bool. An unrecognized scope warns and falls back rather than
    letting a typo silently disable cache-point placement at Bedrock-call time."""
    if raw is None:
        return default
    value = raw.split("#", 1)[0].strip().lower()
    if not value:
        return default
    if value in _VALID_CACHE_SCOPES:
        return value
    import logging

    logging.getLogger(__name__).warning(
        "Unrecognized AGENT_PROMPT_CACHE_SCOPE=%r (valid: %s) - using %r",
        raw,
        ", ".join(sorted(_VALID_CACHE_SCOPES)),
        default,
    )
    return default


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 3002
    log_level: str = "INFO"
    session_cap: int = 5  # max sessions per user; LRU-evicted beyond this


@dataclass
class BedrockConfig:
    """Bedrock is the model provider, full stop — no abstraction over it.
    `region` has no default on purpose: hardcoding a real deployment's
    region as a fallback is a cloud-settings leak, so operators must set
    AWS_REGION explicitly instead.
    """

    region: str
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # Cross-region inference-profile ARNs need an explicit vendor hint — this
    # is Bedrock's own "provider" kwarg (anthropic/amazon/meta/...), unrelated
    # to the multi-vendor abstraction this template deliberately doesn't have.
    model_vendor_override: str | None = None
    api_key: str | None = None  # bearer token -> AWS_BEARER_TOKEN_BEDROCK
    profile: str | None = None  # AWS_PROFILE; ignored when api_key is set
    connect_timeout: int = 10
    read_timeout: int = 300
    max_retries: int = 2
    prompt_cache_enabled: bool = False
    prompt_cache_scope: str = "full"  # "full" | "prefix_only"
    prompt_cache_ttl: str | None = None


@dataclass
class GuardrailConfig:
    """Bedrock input guardrail — fail-open by design (see
    middleware/input_guardrail.py): unconfigured or erroring, the turn
    proceeds. `guardrail_id` unset (the default) means no guardrail at all,
    not an error."""

    guardrail_id: str | None = None
    guardrail_version: str = "DRAFT"


@dataclass
class MemoryConfig:
    """AgentCore-backed persistence. `memory_id` unset (default) means
    AgentCore is not configured — get_checkpointer()/get_memory_store() use
    plain in-memory persistence instead, and the agent runs a complete turn
    either way. No real AWS resource IDs default here: the reference
    hardcodes a memory ID with a person's name baked in and a real
    strategy ID as a module constant — exactly the kind of cloud-settings
    leak this template avoids.

    `checkpointer_enabled` is a SEPARATE opt-in, off by default even when
    `memory_id` is set: `langgraph-checkpoint-aws` 1.0.7 doesn't reliably
    round-trip LangGraph interrupt pending-writes with the langgraph
    version this template pins, which can break HITL resume (the approved
    tool never executes — the paused state isn't restored on read-back).
    The long-term memory STORE is unaffected by this and always uses
    AgentCore once `memory_id` is set; only checkpointing (conversation
    turn/interrupt state) has this extra gate.
    """

    memory_id: str | None = None
    semantic_strategy_id: str | None = None
    top_k: int = 5
    checkpointer_enabled: bool = False


@dataclass
class KnowledgeBaseConfig:
    """Bedrock Knowledge Base — read-only retrieval bound as a tool.
    `kb_id` unset (default) means KB is entirely disabled: get_kb_client()
    returns None and no kb_retrieve tool is registered. Same pattern as
    MemoryConfig.memory_id / GuardrailConfig.guardrail_id — no separate
    enabled flag, to avoid the drift risk called out on FeatureFlags below.
    """

    kb_id: str | None = None
    region: str | None = None  # falls back to BedrockConfig.region when unset
    chunks: int = 3  # numberOfResults per retrieve call
    max_chars: int = 2000  # char cap on the tool's formatted response


@dataclass
class ObservabilityConfig:
    """ADOT (OpenTelemetry) tracing/metrics + CloudWatch Logs shipping.

    ADOT tracing/metrics is gated by FeatureFlags.observability_enabled (see
    below) — this dataclass only holds ITS settings. CloudWatch Logs is
    gated independently by `cloudwatch_log_group` presence, same pattern as
    MemoryConfig.memory_id: unset means disabled, not an error, and it can
    be on even when ADOT tracing is off (or vice versa) — they're separate
    AWS integrations in the reference too.
    """

    otel_service_name: str = "ai-agent-template"
    # Raw "key1=val1,key2=val2" string, same tolerant parsing as the
    # reference — parsed downstream in observability/tracer.py.
    otel_resource_attributes: str | None = None
    cloudwatch_log_group: str | None = None
    cloudwatch_log_stream: str | None = None  # default: <hostname>/<date> when unset


@dataclass
class FeatureFlags:
    """One flag per optional module. All default False — see module
    docstring. Each flag's real config (model IDs, endpoints, limits) lands
    with its own task; this is just the on/off switch.

    AgentCore memory and the knowledge base have no flag here —
    MemoryConfig.memory_id / KnowledgeBaseConfig.kb_id unset IS "disabled",
    same pattern as GuardrailConfig.guardrail_id. A separate enabled flag
    next to either would just invite drift (id set but the flag off,
    silently unused)."""

    observability_enabled: bool = False
    document_ingestion_enabled: bool = False
    code_execution_enabled: bool = False
    orchestration_enabled: bool = False


@dataclass
class AppConfig:
    # Required, no default — must come before the defaulted fields below
    # (dataclass field-ordering rule), same shape as mcp-server-template's
    # AppConfig.graphql (also required-first).
    bedrock: BedrockConfig
    server: ServerConfig = field(default_factory=ServerConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    # Base config for the app-wide MCP client — agent_session_id is left
    # unset here; it's a per-session value set when a session's own client
    # is built, not a process-wide setting.
    mcp: MCPClientConfig = field(default_factory=MCPClientConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)


def load_config() -> AppConfig:
    region = os.environ.get("AWS_REGION")
    if not region:
        raise RuntimeError(
            "AWS_REGION environment variable is required - no default, so a region "
            "never silently leaks from this template into your deployment"
        )

    return AppConfig(
        bedrock=BedrockConfig(
            region=region,
            model_id=os.environ.get(
                "AGENT_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            model_vendor_override=os.environ.get("AGENT_MODEL_VENDOR_OVERRIDE"),
            api_key=os.environ.get("BEDROCK_API_KEY"),
            profile=os.environ.get("AWS_PROFILE"),
            connect_timeout=int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10")),
            read_timeout=int(os.environ.get("BEDROCK_READ_TIMEOUT", "300")),
            max_retries=int(os.environ.get("BEDROCK_MAX_RETRIES", "2")),
            prompt_cache_enabled=_parse_bool(
                os.environ.get("AGENT_PROMPT_CACHE_ENABLED"), default=False
            ),
            prompt_cache_scope=_parse_cache_scope(os.environ.get("AGENT_PROMPT_CACHE_SCOPE")),
            prompt_cache_ttl=os.environ.get("AGENT_PROMPT_CACHE_TTL") or None,
        ),
        server=ServerConfig(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "3002")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            session_cap=int(os.environ.get("AGENT_SESSION_CAP", "5")),
        ),
        guardrail=GuardrailConfig(
            guardrail_id=os.environ.get("AGENT_GUARDRAIL_ID"),
            guardrail_version=os.environ.get("AGENT_GUARDRAIL_VERSION", "DRAFT"),
        ),
        mcp=MCPClientConfig(
            server_url=os.environ.get("MCP_SERVER_URL", "http://localhost:3001/mcp"),
            auth_token=os.environ.get("MCP_AUTH_TOKEN", ""),
            verify_ssl=_parse_bool(os.environ.get("MCP_VERIFY_SSL"), default=True),
        ),
        memory=MemoryConfig(
            memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
            semantic_strategy_id=os.environ.get("AGENTCORE_SEMANTIC_STRATEGY_ID"),
            top_k=int(os.environ.get("AGENT_MEMORY_TOP_K", "5")),
            checkpointer_enabled=_parse_bool(
                os.environ.get("AGENT_AGENTCORE_CHECKPOINTER_ENABLED"), default=False
            ),
        ),
        knowledge_base=KnowledgeBaseConfig(
            kb_id=os.environ.get("AGENT_KB_ID"),
            region=os.environ.get("AGENT_KB_REGION"),
            chunks=int(os.environ.get("AGENT_KB_CHUNKS", "3")),
            max_chars=int(os.environ.get("AGENT_KB_MAX_CHARS", "2000")),
        ),
        observability=ObservabilityConfig(
            otel_service_name=os.environ.get("OTEL_SERVICE_NAME", "ai-agent-template"),
            otel_resource_attributes=os.environ.get("OTEL_RESOURCE_ATTRIBUTES"),
            cloudwatch_log_group=os.environ.get("CLOUDWATCH_LOG_GROUP"),
            cloudwatch_log_stream=os.environ.get("CLOUDWATCH_LOG_STREAM"),
        ),
        features=FeatureFlags(
            observability_enabled=_parse_bool(
                os.environ.get("AGENT_OBSERVABILITY_ENABLED"), default=False
            ),
            document_ingestion_enabled=_parse_bool(
                os.environ.get("AGENT_DOCUMENT_INGESTION_ENABLED"), default=False
            ),
            code_execution_enabled=_parse_bool(
                os.environ.get("AGENT_CODE_EXECUTION_ENABLED"), default=False
            ),
            orchestration_enabled=_parse_bool(
                os.environ.get("AGENT_ORCHESTRATION_ENABLED"), default=False
            ),
        ),
    )

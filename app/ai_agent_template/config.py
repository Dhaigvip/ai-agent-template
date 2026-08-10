"""
Environment-variable-driven configuration for ai-agent-template.

Mirrors mcp-server-template/config.py's shape on purpose: plain dataclasses,
one load_config() entry point, .env loaded once via python-dotenv. The two
templates should feel like the same project.

Naming: unprefixed vars are server-level (HOST, PORT, LOG_LEVEL) or standard
SDK names (AWS_REGION, AWS_PROFILE, OPENAI_API_KEY — the latter follows the
same "use the SDK's own standard var name" convention as the AWS ones, not
an inconsistency); AGENT_* is this project's own prefix, used throughout
instead of any product-specific prefix.

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


_VALID_PROVIDERS = {"bedrock", "openai"}


def _parse_provider(raw: str | None, *, default: str = "openai") -> str:
    """Tolerant provider parsing — same inline-comment/whitespace tolerance
    as _parse_bool/_parse_cache_scope. Defaults to OpenAI: zero-config,
    just OPENAI_API_KEY. Bedrock is still the recommended provider for
    production (AgentCore memory, Bedrock Guardrails, Bedrock Knowledge
    Base, CloudWatch/ADOT all require it) but needs real AWS setup, so it's
    opt-in via AGENT_MODEL_PROVIDER=bedrock rather than the default. An
    unrecognized provider warns and falls back to this default rather than
    crashing the server over a typo."""
    if raw is None:
        return default
    value = raw.split("#", 1)[0].strip().lower()
    if not value:
        return default
    if value in _VALID_PROVIDERS:
        return value
    import logging

    logging.getLogger(__name__).warning(
        "Unrecognized AGENT_MODEL_PROVIDER=%r (valid: %s) - using %r",
        raw,
        ", ".join(sorted(_VALID_PROVIDERS)),
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
    """Bedrock is the recommended provider for production use — AgentCore
    memory, Bedrock Guardrails, Bedrock Knowledge Base, and CloudWatch/ADOT
    observability all require it and stay Bedrock-only regardless of which
    provider AppConfig.provider selects — but it's opt-in
    (AGENT_MODEL_PROVIDER=bedrock), not the default; OpenAI is (see
    OpenAIConfig below), since it needs nothing but an API key to run.

    `region` defaults to empty string, not a real region: hardcoding one as
    a fallback is a cloud-settings leak, so operators must set AWS_REGION
    explicitly to use anything Bedrock-shaped. load_config() enforces this
    is actually set (raises) exactly when it's required — provider is
    "bedrock", or knowledge_base.kb_id / code_execution.interpreter_id is
    set, since those two stay Bedrock-only unconditionally. Defaulted here
    (rather than required-no-default) so AppConfig itself needs zero
    arguments to construct — matches OpenAI being the zero-config default.
    """

    region: str = ""
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # Cross-region inference-profile ARNs need an explicit vendor hint — this
    # is Bedrock's own "provider" kwarg (anthropic/amazon/meta/...), unrelated
    # to AppConfig.provider (which model provider is active at all).
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
class OpenAIConfig:
    """OpenAI provider config — only consulted when AppConfig.provider ==
    "openai". `api_key` left None by default: langchain-openai's ChatOpenAI
    reads the standard OPENAI_API_KEY env var itself if unset here, the same
    permissive pattern as BedrockConfig.profile falling back to ambient AWS
    credential resolution. `base_url` supports OpenAI-compatible endpoints
    (Azure OpenAI, local proxies, etc.) without a separate config shape.

    Selecting this provider does not disable AgentCore memory, Bedrock
    Guardrails, or Bedrock Knowledge Base configuration — it just means
    those features silently stay off (memory/guardrails degrade to
    in-memory / a keyword-blocklist fallback; KB requires Bedrock regardless
    and would need AWS_REGION set separately if used alongside OpenAI).
    """

    api_key: str | None = None
    model_id: str = "gpt-4o-mini"
    base_url: str | None = None
    timeout: int = 60
    max_retries: int = 2


@dataclass
class GuardrailConfig:
    """Input guardrail — fail-open by design (see
    middleware/input_guardrail.py): unconfigured or erroring, the turn
    proceeds. `guardrail_id` unset (the default) means no Bedrock guardrail
    configured. When AppConfig.provider != "bedrock", the Bedrock guardrail
    is never consulted regardless of `guardrail_id` — see
    middleware/guardrail_fallback.py for the non-Bedrock path, which uses a
    small built-in keyword blocklist plus `blocklist_terms` below."""

    guardrail_id: str | None = None
    guardrail_version: str = "DRAFT"
    blocklist_terms: str | None = None  # comma-separated extra terms for the fallback guardrail


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
class OrchestrationConfig:
    """`run_orchestration` — a local, in-process RestrictedPython sandbox for
    calling already-bound tools in a dependent chain/branch/reshape/loop.
    Gated solely by `FeatureFlags.orchestration_enabled`: unlike code
    execution (Task 16), this tool has no external AWS resource to gate on,
    so there's nothing here that needs its own id-presence check.

    Defaults match `agent/orchestration/sandbox.py`'s own module constants —
    kept here instead of read from `os.environ` inline so every env read in
    this project stays centralized in `load_config()` (see Task 13's
    observability config for the same reasoning).
    """

    timeout_seconds: float = 15.0
    max_result_chars: int = 4000
    max_calls: int = 256
    call_timeout_seconds: float = 15.0


@dataclass
class CodeExecutionConfig:
    """`run_python` — a remote AgentCore Code Interpreter sandbox for data
    analysis/charts (statistics, aggregation, Plotly). Distinct from
    Task 17's `run_orchestration`: this tool has NO access to this project's
    own tools at all, it only runs Python with pandas/numpy/plotly.

    `interpreter_id` unset (default) means code execution is entirely
    disabled: `code_exec_tool()` is never added to the toolset. Same
    id-presence gating as `KnowledgeBaseConfig.kb_id` / `MemoryConfig.memory_id`
    — no separate enabled flag. (This template's Task 2 originally scaffolded
    `FeatureFlags.code_execution_enabled` before this config existed; retired
    here for the same drift reason `AGENT_KNOWLEDGE_BASE_ENABLED` was retired
    in Task 18 — an id set with the flag off would silently do nothing.)
    """

    interpreter_id: str | None = None
    region: str | None = None  # falls back to BedrockConfig.region when unset
    exec_timeout_seconds: int = 90  # wall-clock ceiling per run_python call
    sandbox_idle_timeout_seconds: int = 300  # AgentCore session idle timeout


@dataclass
class FeatureFlags:
    """One flag per optional module. All default False — see module
    docstring. Each flag's real config (model IDs, endpoints, limits) lands
    with its own task; this is just the on/off switch.

    AgentCore memory, the knowledge base, and code execution have no flag
    here — MemoryConfig.memory_id / KnowledgeBaseConfig.kb_id /
    CodeExecutionConfig.interpreter_id unset IS "disabled", same pattern as
    GuardrailConfig.guardrail_id. A separate enabled flag next to any of
    these would just invite drift (id set but the flag off, silently
    unused)."""

    observability_enabled: bool = False
    document_ingestion_enabled: bool = False
    orchestration_enabled: bool = False


@dataclass
class AppConfig:
    # "openai" (default, zero-config) or "bedrock" (recommended for
    # production — see BedrockConfig docstring for why). Neither this nor
    # `bedrock` below is required-no-default anymore: AppConfig() alone is
    # a valid, fully zero-config construction now that OpenAI is the
    # default provider.
    provider: str = "openai"
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    # Base config for the app-wide MCP client — agent_session_id is left
    # unset here; it's a per-session value set when a session's own client
    # is built, not a process-wide setting.
    mcp: MCPClientConfig = field(default_factory=MCPClientConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    code_execution: CodeExecutionConfig = field(default_factory=CodeExecutionConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)


def load_config() -> AppConfig:
    provider = _parse_provider(os.environ.get("AGENT_MODEL_PROVIDER"))

    # Read ahead of the region check: KB and code execution stay Bedrock-only
    # regardless of provider, so either one being configured still requires
    # AWS_REGION even when provider=="openai". Reused below when building
    # KnowledgeBaseConfig/CodeExecutionConfig rather than re-reading.
    kb_id = os.environ.get("AGENT_KB_ID")
    interpreter_id = os.environ.get("AGENTCORE_CODE_INTERPRETER_ID")

    region_env = os.environ.get("AWS_REGION")
    region_required = provider == "bedrock" or bool(kb_id) or bool(interpreter_id)
    if region_required and not region_env:
        raise RuntimeError(
            "AWS_REGION environment variable is required when AGENT_MODEL_PROVIDER=bedrock "
            "(the default), or when AGENT_KB_ID / AGENTCORE_CODE_INTERPRETER_ID is set - "
            "those two stay Bedrock-only regardless of provider. No default, so a region "
            "never silently leaks from this template into your deployment."
        )
    region = region_env or ""

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
        provider=provider,
        openai=OpenAIConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_id=os.environ.get("AGENT_OPENAI_MODEL_ID", "gpt-4o-mini"),
            base_url=os.environ.get("AGENT_OPENAI_BASE_URL") or None,
            timeout=int(os.environ.get("AGENT_OPENAI_TIMEOUT", "60")),
            max_retries=int(os.environ.get("AGENT_OPENAI_MAX_RETRIES", "2")),
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
            blocklist_terms=os.environ.get("AGENT_GUARDRAIL_BLOCKLIST"),
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
            kb_id=kb_id,
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
        orchestration=OrchestrationConfig(
            timeout_seconds=float(os.environ.get("AGENT_ORCHESTRATION_TIMEOUT_SECONDS", "15.0")),
            max_result_chars=int(os.environ.get("AGENT_ORCHESTRATION_MAX_RESULT_CHARS", "4000")),
            max_calls=int(os.environ.get("AGENT_ORCHESTRATION_MAX_CALLS", "256")),
            call_timeout_seconds=float(
                os.environ.get("AGENT_ORCHESTRATION_CALL_TIMEOUT_SECONDS", "15.0")
            ),
        ),
        code_execution=CodeExecutionConfig(
            interpreter_id=interpreter_id,
            region=os.environ.get("AGENT_CODE_EXECUTION_REGION"),
            exec_timeout_seconds=int(os.environ.get("AGENT_CODE_EXECUTION_TIMEOUT_SECONDS", "90")),
            sandbox_idle_timeout_seconds=int(
                os.environ.get("AGENT_CODE_EXECUTION_SANDBOX_IDLE_TIMEOUT_SECONDS", "300")
            ),
        ),
        features=FeatureFlags(
            observability_enabled=_parse_bool(
                os.environ.get("AGENT_OBSERVABILITY_ENABLED"), default=False
            ),
            document_ingestion_enabled=_parse_bool(
                os.environ.get("AGENT_DOCUMENT_INGESTION_ENABLED"), default=False
            ),
            orchestration_enabled=_parse_bool(
                os.environ.get("AGENT_ORCHESTRATION_ENABLED"), default=False
            ),
        ),
    )

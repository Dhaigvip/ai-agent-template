"""Bedrock model factory.

No provider abstraction — "Cloud coupling: Bedrock only, on
purpose". create_model() is still the one entry point every call site uses
(the main agent model, a cheaper summarization model, ...): even with a
single provider, one seam here means every caller stays stable if the
construction details change later, matching how the reference itself always
calls create_model() rather than the Bedrock-specific builder directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse

from ai_agent_template.config import BedrockConfig

logger = logging.getLogger(__name__)

_VALID_CACHE_SCOPES = {"full", "prefix_only"}


class ScopedCacheChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with application-level cache-scope control.

    LangChain's default Converse cache logic adds cache points to system,
    tools, and the last message. For high-churn turns that last-message
    cache point can cause repeated cache writes on volatile context. This
    override keeps full behavior in "full" mode and adds "prefix_only"
    (system+tools only).
    """

    def _apply_cache_points(self, cache_control, system, bedrock_messages, params=None) -> None:
        if not cache_control:
            return

        is_nova = "amazon.nova" in self._get_base_model().lower()

        cache_point: dict[str, Any] = {"type": "default"}
        ttl = cache_control.get("ttl")
        if ttl and ttl != "5m" and not is_nova:
            cache_point["ttl"] = ttl
        cache_block = {"cachePoint": cache_point}

        def _is_cache_point_block(block: Any) -> bool:
            return isinstance(block, dict) and "cachePoint" in block

        scope = str(cache_control.get("scope", "full")).strip().lower()
        if scope not in _VALID_CACHE_SCOPES:
            scope = "full"

        if system and not any(_is_cache_point_block(b) for b in system):
            system.append(cache_block)

        if scope == "full" and bedrock_messages:
            last_content = bedrock_messages[-1].get("content")
            if isinstance(last_content, list):
                has_tool_block = is_nova and any(
                    isinstance(b, dict) and ("toolResult" in b or "toolUse" in b)
                    for b in last_content
                )
                if not has_tool_block and not any(_is_cache_point_block(b) for b in last_content):
                    last_content.append(cache_block)

        if params and not is_nova:
            tools = params.get("toolConfig", {}).get("tools")
            if tools and not any(_is_cache_point_block(t) for t in tools):
                tools.append(cache_block)


def build_cache_control(bedrock: BedrockConfig) -> dict | None:
    """Return Bedrock cache-control kwargs, or None when caching is disabled."""
    if not bedrock.prompt_cache_enabled:
        return None
    cache_control: dict[str, Any] = {"type": "default", "scope": bedrock.prompt_cache_scope}
    if bedrock.prompt_cache_ttl and bedrock.prompt_cache_ttl != "5m":
        cache_control["ttl"] = bedrock.prompt_cache_ttl
    return cache_control


def create_model(bedrock: BedrockConfig, model_id: str | None = None) -> ChatBedrockConverse:
    """Create a Bedrock chat model.

    model_id overrides bedrock.model_id when supplied — used for per-purpose
    model overrides (e.g. a cheaper model for summarization). When prompt
    caching is enabled, the returned model has cache_control pre-bound; the
    middleware chain may unwind and re-apply this through model_settings
    instead so it composes with other middleware.
    """
    name = model_id or bedrock.model_id

    if bedrock.api_key:
        # AWS SDK natively recognises AWS_BEARER_TOKEN_BEDROCK and sends it as
        # "Authorization: Bearer <token>".
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bedrock.api_key

    kwargs: dict[str, Any] = {
        "model": name,
        "region_name": bedrock.region,
        "config": BotoConfig(
            read_timeout=bedrock.read_timeout,
            connect_timeout=bedrock.connect_timeout,
            retries={"max_attempts": bedrock.max_retries},
        ),
    }
    if bedrock.model_vendor_override:
        kwargs["provider"] = bedrock.model_vendor_override
    if bedrock.profile and not bedrock.api_key:
        kwargs["credentials_profile_name"] = bedrock.profile

    model = ScopedCacheChatBedrockConverse(**kwargs)

    cache_control = build_cache_control(bedrock)
    if cache_control is not None:
        logger.info(
            "create_model: prompt caching enabled for %s (scope=%s, ttl=%s)",
            name,
            cache_control["scope"],
            cache_control.get("ttl", "5m"),
        )
        return model.bind(cache_control=cache_control)
    return model

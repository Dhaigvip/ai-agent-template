"""Resilient wrappers for AgentCore checkpointer and store.

When AWS session credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
AWS_SESSION_TOKEN) expire, AgentCore calls fail. Rather than crashing the
entire chat turn, these wrappers catch credential/auth errors and fall back
to in-memory persistence — the chat continues to work (just without
cross-session persistence until credentials are refreshed).

A warning callback is invoked once (not per-call) so the client can display
a degradation notice.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore, GetOp, ListNamespacesOp, PutOp, SearchOp
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

# Botocore/boto3 error codes that indicate expired or invalid credentials.
_CREDENTIAL_ERROR_CODES = frozenset(
    {
        "ExpiredTokenException",
        "ExpiredToken",
        "InvalidIdentityToken",
        "UnrecognizedClientException",
        "InvalidClientTokenId",
        "AccessDeniedException",
        "AuthFailure",
        "InvalidSignatureException",
        "IncompleteSignature",
    }
)


def _is_credential_error(exc: Exception) -> bool:
    """Return True if the exception is an AWS credential/auth failure."""
    # botocore.exceptions.ClientError
    if hasattr(exc, "response"):
        code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
        if code in _CREDENTIAL_ERROR_CODES:
            return True
    # botocore.exceptions.NoCredentialsError / PartialCredentialsError
    exc_type = type(exc).__name__
    if exc_type in ("NoCredentialsError", "PartialCredentialsError", "TokenRetrievalError"):
        return True
    # Check message as last resort
    msg = str(exc).lower()
    if any(term in msg for term in ("expired", "security token", "credentials", "not authorized")):
        return True
    return False


WarningCallback = Callable[[str, str], None]


class ResilientCheckpointer(BaseCheckpointSaver):
    """Wraps an AgentCore checkpointer; falls back to MemorySaver on credential errors."""

    def __init__(
        self,
        primary: BaseCheckpointSaver,
        on_degraded: Optional[WarningCallback] = None,
    ) -> None:
        super().__init__()
        self._primary = primary
        self._fallback = MemorySaver()
        self._degraded = False
        self._on_degraded = on_degraded

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _degrade(self, err: Exception) -> None:
        if not self._degraded:
            self._degraded = True
            reason = str(err)
            logger.warning(
                "ResilientCheckpointer: AgentCore credentials failed - "
                "falling back to in-memory. Reason: %s",
                reason,
            )
            if self._on_degraded:
                self._on_degraded(
                    "memory",
                    "Conversation memory (AgentCore) is unavailable due to expired AWS "
                    "credentials. Chat will continue but conversation state won't persist "
                    "across restarts.",
                )

    @property
    def _active(self) -> BaseCheckpointSaver:
        return self._fallback if self._degraded else self._primary

    # ── BaseCheckpointSaver interface ─────────────────────────────────────────

    async def aget_tuple(self, config, **kwargs):
        try:
            return await self._active.aget_tuple(config, **kwargs)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return await self._fallback.aget_tuple(config, **kwargs)
            raise

    async def alist(self, config, **kwargs):
        try:
            async for item in self._active.alist(config, **kwargs):
                yield item
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                async for item in self._fallback.alist(config, **kwargs):
                    yield item
            else:
                raise

    async def aput(self, config, checkpoint, metadata, new_versions):
        try:
            return await self._active.aput(config, checkpoint, metadata, new_versions)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return await self._fallback.aput(config, checkpoint, metadata, new_versions)
            raise

    async def aput_writes(self, config, writes, task_id, **kwargs):
        try:
            return await self._active.aput_writes(config, writes, task_id, **kwargs)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return await self._fallback.aput_writes(config, writes, task_id, **kwargs)
            raise

    # Sync variants (LangGraph may call these in some paths)
    def get_tuple(self, config, **kwargs):
        try:
            return self._active.get_tuple(config, **kwargs)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return self._fallback.get_tuple(config, **kwargs)
            raise

    def list(self, config, **kwargs):
        try:
            yield from self._active.list(config, **kwargs)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                yield from self._fallback.list(config, **kwargs)
            else:
                raise

    def put(self, config, checkpoint, metadata, new_versions):
        try:
            return self._active.put(config, checkpoint, metadata, new_versions)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return self._fallback.put(config, checkpoint, metadata, new_versions)
            raise

    def put_writes(self, config, writes, task_id, **kwargs):
        try:
            return self._active.put_writes(config, writes, task_id, **kwargs)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return self._fallback.put_writes(config, writes, task_id, **kwargs)
            raise


class ResilientStore(BaseStore):
    """Wraps an AgentCore memory store; falls back to InMemoryStore on credential errors."""

    def __init__(
        self,
        primary: BaseStore,
        on_degraded: Optional[WarningCallback] = None,
    ) -> None:
        self._primary = primary
        self._fallback = InMemoryStore()
        self._degraded = False
        self._on_degraded = on_degraded

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _degrade(self, err: Exception) -> None:
        if not self._degraded:
            self._degraded = True
            reason = str(err)
            logger.warning(
                "ResilientStore: AgentCore credentials failed - "
                "falling back to in-memory. Reason: %s",
                reason,
            )
            if self._on_degraded:
                self._on_degraded(
                    "long_term_memory",
                    "Long-term memory (AgentCore) is unavailable due to expired AWS "
                    "credentials. Chat will continue but cross-session context won't be "
                    "available.",
                )

    @property
    def _active(self) -> BaseStore:
        return self._fallback if self._degraded else self._primary

    # ── BaseStore interface ───────────────────────────────────────────────────

    async def abatch(self, ops: Sequence[GetOp | PutOp | SearchOp | ListNamespacesOp]) -> list:
        try:
            return await self._active.abatch(ops)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return await self._fallback.abatch(ops)
            raise

    def batch(self, ops: Sequence[GetOp | PutOp | SearchOp | ListNamespacesOp]) -> list:
        try:
            return self._active.batch(ops)
        except Exception as exc:
            if not self._degraded and _is_credential_error(exc):
                self._degrade(exc)
                return self._fallback.batch(ops)
            raise

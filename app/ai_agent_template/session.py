"""Session management for ai-agent-template.

Session model:
    Client owns session identity — it stores the sessionId from init_ok in localStorage
    and sends it back on every reconnect.

    On every init the server resolves the session:
      1. Client provides sessionId -> look up _session_cache -> resume (resumed=true)
      2. No sessionId (or not found in cache) -> create a new session (resumed=false)

    On disconnect: gateway is NOT destroyed — session stays alive in _session_cache
    so a fast reconnect with the same sessionId resumes instantly.

    On clear: server creates a new session and sends init_ok with the new sessionId.
    The client must replace the stored sessionId.

    Cap = AGENT_SESSION_CAP (default 5, env) sessions per user. LRU eviction on cache only.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_agent_template.config import AppConfig

logger = logging.getLogger(__name__)


# ── Session model ────────────────────────────────────────────────────────────


@dataclass
class SessionRuntime:
    """One live session — owns the gateway (compiled graph + MCP) and metadata."""

    session_id: str
    gateway: Any  # AIGateway — imported lazily to avoid circular deps
    user_id: str | None
    title: str  # first 80 chars of first user message; "" until first turn
    created_at: str  # ISO 8601 UTC
    last_active: str  # ISO 8601 UTC, updated after every successful turn
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Process-level session cache. Key = session_id = LangGraph thread_id.
_session_cache: dict[str, SessionRuntime] = {}


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def _touch(runtime: SessionRuntime, title: str = "") -> None:
    """Update last_active and title (once) in the in-process cache."""
    runtime.last_active = _now_iso()
    if not runtime.title and title:
        runtime.title = title


async def _evict_lru_for_user(user_id: str | None, session_cap: int) -> None:
    """Evict the oldest session for a user when the per-user cap is exceeded.

    Uses the in-process cache as the source of truth. The evicted session is
    removed from _session_cache and its gateway is destroyed.
    """
    user_sessions = [r for r in _session_cache.values() if r.user_id == user_id]
    if len(user_sessions) < session_cap:
        return
    oldest = sorted(user_sessions, key=lambda r: r.last_active)[0]
    logger.info(
        "session.evict: session_id=%s user_id=%s last_active=%s",
        oldest.session_id,
        user_id,
        oldest.last_active,
    )
    _session_cache.pop(oldest.session_id, None)
    try:
        await oldest.gateway.destroy()
    except Exception as err:
        logger.debug("session.evict.destroy_error: %s", err)


async def _resolve_session(
    session_id: str | None,
    user_id: str | None,
    init_config: dict,
    config: "AppConfig",
) -> tuple[SessionRuntime, bool]:
    """Return (SessionRuntime, resumed).

    Resolution:
      1. Client provides sessionId AND it is in _session_cache -> resume (resumed=True)
      2. Otherwise -> create a new session (resumed=False)

    The client is responsible for persisting the sessionId (e.g. localStorage) and
    sending it back on every reconnect. No server-side lookup table is needed.
    """
    from ai_agent_template.agent.ai_gateway import AIGateway

    # 1. Resume if client provided a known session_id
    if session_id and session_id in _session_cache:
        runtime = _session_cache[session_id]
        logger.debug("session.resume.cache: session_id=%s user_id=%s", session_id, user_id)
        return runtime, True

    # 2. Create a new session
    await _evict_lru_for_user(user_id, config.server.session_cap)

    new_id = str(uuid.uuid4())
    now = _now_iso()
    new_config = {**init_config, "agent_session_id": new_id}
    runtime = SessionRuntime(
        session_id=new_id,
        gateway=AIGateway(config, new_config),
        user_id=user_id,
        title="",
        created_at=now,
        last_active=now,
    )
    _session_cache[new_id] = runtime
    logger.debug("session.created: session_id=%s user_id=%s", new_id, user_id)
    return runtime, False


async def destroy_all_sessions() -> int:
    """Destroy all active sessions. Called during server shutdown.

    Returns:
        Number of sessions destroyed.
    """
    count = 0
    for runtime in list(_session_cache.values()):
        try:
            await runtime.gateway.destroy()
            count += 1
        except Exception:
            pass
    _session_cache.clear()
    return count

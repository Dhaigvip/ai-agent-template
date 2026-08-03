"""WebSocket handler for ai-agent-template.

WebSocket message protocol (Client -> Server):
    All fields are optional — absent fields are treated as null/undefined server-side.

    { type: "init";    userId?, userName?, sessionId? }
      sessionId — client-stored UUID from a previous init_ok; omit on first connect.
                  When present the server resumes that session (history preserved).
                  When absent the server creates a new session.

    { type: "turn";    text: string }
    { type: "hitl_decisions"; decisions: Decision[] }

    Decision (positional — decisions[i] answers actions[i] of the hitl_request event):
      { type: "approve" }
      { type: "edit";    edited_action: { name: string; args: object } }
      { type: "reject";  message?: string }
      { type: "respond"; message: string }   // human answers AS the tool (ask_user)

    HITL is stateless: the turn ENDS at the interrupt and hitl_decisions resumes it from the
    checkpoint, so decisions are positional and survive a server restart.
    { type: "clear" }

WebSocket protocol (Server -> Client):
    All fields beyond "type" are optional — the UI must treat absent fields as safe defaults.

    { type: "init_ok"; sessionId?: string; resumed?: boolean }
      sessionId — the session UUID the client must store (e.g. localStorage) and send
                  back in every subsequent init message for this conversation.
      resumed   — true if the provided sessionId was found and history was restored,
                  false if a new session was created.

    Wire event JSON objects (see agent/wire.py) — text_delta, text_commit, tool_auto,
      activity, refresh, hitl_request, done, error. All fields beyond "type" are optional;
      UI must handle missing fields gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ai_agent_template.config import AppConfig
from ai_agent_template.session import (
    SessionRuntime,
    _evict_lru_for_user,
    _now_iso,
    _resolve_session,
    _session_cache,
    _touch,
)

logger = logging.getLogger(__name__)


async def handle_websocket(websocket: WebSocket, *, config: AppConfig) -> None:
    """One WebSocket connection = one agent session.

    Session identity is server-generated and stable across reconnects.
    session_id == LangGraph thread_id == checkpoint key.

    Cold-start:  client sends { type: "init" } without sessionId.
                 Server generates session_id, creates SessionRuntime, replies init_ok.
    Warm path:   client sends { type: "init", sessionId: "..." }.
                 Server resumes existing SessionRuntime (history preserved).
    Reconnect:   client stores sessionId from init_ok in localStorage, sends it
                 in every subsequent init. Gateway stays alive in _session_cache.
    """
    await websocket.accept()

    connection_id = str(uuid.uuid4())[:8]  # for log correlation only
    ip = websocket.client.host if websocket.client else "unknown"

    logger.info("ws.connected: connection_id=%s ip=%s", connection_id, ip)

    # runtime is set on "init"; all other handlers use it
    runtime: SessionRuntime | None = None

    # Background task running the current turn
    _turn_task: asyncio.Task | None = None

    async def send(event: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(event))
        except Exception:
            pass  # client already disconnected

    async def _run_turn(text: str, trace_id: str) -> None:
        """Runs the agent turn as a background asyncio.Task.

        Critical for HITL: the outer receive loop must stay live to process
        approve/clarify messages while the turn is suspended.
        asyncio.create_task() ensures iter_text() keeps running concurrently.
        """
        nonlocal runtime
        assert runtime is not None
        _turn_start = time.monotonic()
        async with runtime.lock:
            try:
                async for event in runtime.gateway.run_turn(text, trace_id):
                    logger.debug(
                        "ws.event.send: type=%s session_id=%s",
                        event.get("type", "?"),
                        runtime.session_id[:8],
                    )
                    await send(event)
                # Pass first turn title to _touch; stored once
                _first_turn_title = text[:80] if not runtime.title else ""
                await _touch(runtime, title=_first_turn_title)
                _dur = int((time.monotonic() - _turn_start) * 1000)
                logger.info(
                    "ws.turn.done: session_id=%s user_id=%s trace_id=%s duration_ms=%d",
                    runtime.session_id,
                    runtime.user_id,
                    trace_id,
                    _dur,
                )
            except asyncio.CancelledError:
                logger.info(
                    "ws.turn.cancelled: session_id=%s user_id=%s trace_id=%s",
                    runtime.session_id,
                    runtime.user_id,
                    trace_id,
                )
                raise
            except Exception as err:
                logger.error(
                    "ws.turn.error: session_id=%s user_id=%s trace_id=%s error=%s",
                    runtime.session_id,
                    runtime.user_id,
                    trace_id,
                    err,
                )
                await send({"type": "error", "message": str(err)})

    async def _resume_turn(decisions: list, trace_id: str) -> None:
        """Resume a thread paused on a HITL approval.

        Same shape as _run_turn — a background task so iter_text() keeps receiving — but it
        streams the CONTINUATION from the checkpoint rather than starting a new turn.
        """
        nonlocal runtime
        assert runtime is not None
        async with runtime.lock:
            try:
                async for event in runtime.gateway.resume_turn(decisions, trace_id):
                    await send(event)
                await _touch(runtime)
                logger.info(
                    "ws.resume.done: session_id=%s trace_id=%s", runtime.session_id, trace_id
                )
            except asyncio.CancelledError:
                logger.info(
                    "ws.resume.cancelled: session_id=%s trace_id=%s", runtime.session_id, trace_id
                )
                raise
            except Exception as err:
                logger.error(
                    "ws.resume.error: session_id=%s trace_id=%s error=%s",
                    runtime.session_id,
                    trace_id,
                    err,
                )
                await send({"type": "error", "message": str(err)})

    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("ws.parse_error: connection_id=%s raw=%.100s", connection_id, raw)
                continue

            msg_type = msg.get("type")

            # ── init ─────────────────────────────────────────────────────────
            if msg_type == "init":
                user_id = msg.get("userId")
                # Client sends back the sessionId it stored from a previous init_ok.
                # Absent on first connect — server will create a new session.
                client_session_id = msg.get("sessionId") or None

                init_config = {
                    "agent_session_id": None,  # filled by _resolve_session
                    "ctx_user_id": user_id,
                    "ctx_user_name": msg.get("userName"),
                }

                runtime, resumed = await _resolve_session(
                    client_session_id, user_id, init_config, config
                )

                logger.info(
                    "ws.init: session_id=%s user_id=%s resumed=%s",
                    runtime.session_id,
                    user_id,
                    resumed,
                )
                await send(
                    {"type": "init_ok", "sessionId": runtime.session_id, "resumed": resumed}
                )

            # ── turn ─────────────────────────────────────────────────────────
            elif msg_type == "turn":
                if runtime is None:
                    await send({"type": "error", "message": "Send {type:'init'} first."})
                    continue
                if runtime.lock.locked():
                    await send({"type": "error", "message": "A turn is already in progress."})
                    continue

                text = msg.get("text", "")
                trace_id = str(uuid.uuid4())[:8]
                _turn_task = asyncio.create_task(_run_turn(text, trace_id))

            # ── hitl_decisions (stateless HITL resume) ────────────────────────
            # Positional: decisions[i] answers actions[i] from the `hitl_request` event. The turn
            # already ended at the interrupt, so this starts a NEW stream from the checkpoint.
            elif msg_type == "hitl_decisions":
                decisions = msg.get("decisions") or []
                trace_id = str(uuid.uuid4())[:8]
                logger.info("ws.hitl.decisions: count=%d trace_id=%s", len(decisions), trace_id)
                if runtime:
                    _turn_task = asyncio.create_task(_resume_turn(decisions, trace_id))

            # ── clear — destroy current session, create a fresh one ──────────
            elif msg_type == "clear":
                from ai_agent_template.agent.ai_gateway import AIGateway

                old_session_id = runtime.session_id if runtime else None
                old_user_id = runtime.user_id if runtime else None

                if runtime is not None:
                    _session_cache.pop(runtime.session_id, None)
                    try:
                        await runtime.gateway.destroy()
                    except Exception:
                        pass

                await _evict_lru_for_user(old_user_id, config.server.session_cap)

                new_id = str(uuid.uuid4())
                now = _now_iso()
                new_cfg = {
                    "agent_session_id": new_id,
                    "ctx_user_id": old_user_id,
                    "ctx_user_name": None,
                }
                runtime = SessionRuntime(
                    session_id=new_id,
                    gateway=AIGateway(config, new_cfg),
                    user_id=old_user_id,
                    title="",
                    created_at=now,
                    last_active=now,
                )
                _session_cache[new_id] = runtime
                logger.info("ws.clear: old_session_id=%s new_session_id=%s", old_session_id, new_id)
                # Client must persist the new sessionId.
                await send({"type": "init_ok", "sessionId": new_id, "resumed": False})

            else:
                logger.warning("ws.unknown_type: connection_id=%s type=%s", connection_id, msg_type)

    except WebSocketDisconnect:
        logger.info(
            "ws.disconnected: connection_id=%s session_id=%s",
            connection_id,
            runtime.session_id if runtime else "none",
        )
    except Exception as err:
        logger.error("ws.unhandled: connection_id=%s error=%s", connection_id, err)
    finally:
        # Cancel any in-flight turn task
        if _turn_task and not _turn_task.done():
            _turn_task.cancel()
            try:
                await _turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # DO NOT destroy gateway on disconnect — session stays in cache for reconnect.
        # Gateway is destroyed only on: clear, or cache eviction.
        logger.info(
            "ws.closed: connection_id=%s session_id=%s",
            connection_id,
            runtime.session_id if runtime else "none",
        )

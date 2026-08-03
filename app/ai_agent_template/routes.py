"""Routes for ai-agent-template.

HTTP Routes:
    GET  /health     — liveness probe

WebSocket Routes:
    WS   /api/agent  — agent WebSocket session
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from ai_agent_template.config import AppConfig
from ai_agent_template.websocket import handle_websocket

logger = logging.getLogger(__name__)


def build_router(config: AppConfig) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"ok": True}

    @router.websocket("/api/agent")
    async def ws_endpoint(websocket: WebSocket):
        await handle_websocket(websocket, config=config)

    return router

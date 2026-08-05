"""Routes for ai-agent-template.

HTTP Routes:
    GET  /health                     — liveness probe
    GET  /api/agent/download/{id}    — download a run_python chart/data file

WebSocket Routes:
    WS   /api/agent  — agent WebSocket session
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import Response

from ai_agent_template.analysis.file_manager import get_file_registry
from ai_agent_template.config import AppConfig
from ai_agent_template.websocket import handle_websocket

logger = logging.getLogger(__name__)


def build_router(config: AppConfig) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"ok": True}

    @router.get("/api/agent/download/{file_id}")
    async def download_file(file_id: str):
        """Download a file produced by the run_python tool (chart or data export).

        Files are auto-registered on disk when produced; the registry itself
        doesn't auto-expire them (see FileRegistry.cleanup_old_files for a
        manual sweep hook) — a 404 here means the id was never registered or
        the process restarted and the on-disk file was removed externally.
        """
        file_registry = get_file_registry()
        file = file_registry.get_file(file_id)

        if not file:
            raise HTTPException(status_code=404, detail="File not found or expired")

        # Images render inline so "Open in new tab" shows the chart in the
        # browser instead of forcing a download; data exports (CSV/Excel/...)
        # stay attachments so the download buttons still save a file.
        disposition = "inline" if file.mime_type.startswith("image/") else "attachment"

        return Response(
            content=file.content,
            media_type=file.mime_type,
            headers={"Content-Disposition": f'{disposition}; filename="{file.filename}"'},
        )

    @router.websocket("/api/agent")
    async def ws_endpoint(websocket: WebSocket):
        await handle_websocket(websocket, config=config)

    return router

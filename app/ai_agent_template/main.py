"""ai-agent-template — FastAPI server.

Server wiring ONLY. Routes/session/gateway logic live in their own modules
(routes.py, session.py, ...) — this file stays app setup, lifespan hooks,
and route registration.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_agent_template.config import load_config
from ai_agent_template.observability import setup_cloudwatch_logging, setup_observability
from ai_agent_template.routes import build_router
from ai_agent_template.session import destroy_all_sessions

logger = logging.getLogger(__name__)

config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # LangGraph warns on every custom pydantic type (AgentContext) it
    # deserializes from a checkpoint — we own the type and it deserializes
    # correctly (confirmed live in the state-schema checkpointer round-trip
    # check). Suppress to ERROR.
    logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)

    # CloudWatch Logs — ships all log records to CW when cloudwatch_log_group is
    # set. No-op if unset or watchtower isn't installed.
    setup_cloudwatch_logging(
        config.observability.cloudwatch_log_group,
        config.observability.cloudwatch_log_stream,
        region=config.bedrock.region,
    )

    # ADOT tracing and metrics for CloudWatch/AgentCore Observability dashboards.
    # No-op if AGENT_OBSERVABILITY_ENABLED isn't true or ADOT packages aren't installed.
    setup_observability(config.features.observability_enabled, config.observability)

    logger.info("ai-agent-template starting")
    yield

    session_count = await destroy_all_sessions()
    logger.info("ai-agent-template shutdown: agent_sessions_closed=%d", session_count)


app = FastAPI(title="ai-agent-template", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(build_router(config))

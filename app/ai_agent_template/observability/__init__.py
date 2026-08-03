# Observability — ADOT tracing/metrics + CloudWatch Logs, both off by default.
from ai_agent_template.observability.cloudwatch_logging import setup_cloudwatch_logging
from ai_agent_template.observability.metrics import emit_turn_metrics
from ai_agent_template.observability.tracer import get_meter, get_tracer, is_enabled, setup_observability

__all__ = [
    "setup_cloudwatch_logging",
    "setup_observability",
    "get_tracer",
    "get_meter",
    "is_enabled",
    "emit_turn_metrics",
]

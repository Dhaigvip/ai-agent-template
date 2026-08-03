"""ADOT (AWS Distro for OpenTelemetry) tracing and metrics.

Instruments the agent with distributed traces and custom metrics that
surface in CloudWatch / AgentCore Observability dashboards. Gated by
FeatureFlags.observability_enabled — off by default, and the agent runs a
complete turn either way; this module is never on the critical path.

No-op if:
    - observability_enabled is False
    - the ADOT packages aren't installed
    - setup fails for any other reason (never blocks server startup)

Usage:
    # Once at startup (main.py's lifespan), after logging.basicConfig():
    setup_observability(config.features.observability_enabled, config.observability)

    # Anywhere else:
    from ai_agent_template.observability.tracer import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("custom_operation") as span:
        span.set_attribute("key", "value")

References:
    - AWS AgentCore Observability: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
    - ADOT Python: https://aws-otel.github.io/docs/getting-started/python-sdk
    - OpenTelemetry Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import logging
from typing import Any

from ai_agent_template.config import ObservabilityConfig

logger = logging.getLogger(__name__)

# Globals — initialized by setup_observability(), one process-wide tracer/meter.
_tracer: Any = None
_meter: Any = None
_observability_enabled: bool = False


def setup_observability(enabled: bool, config: ObservabilityConfig) -> bool:
    """Initialize ADOT tracing and metrics.

    Call once at application startup (in lifespan), after
    logging.basicConfig(). Returns True if observability was enabled,
    False if disabled or unavailable.
    """
    global _tracer, _meter, _observability_enabled

    if not enabled:
        logger.debug(
            "observability: AGENT_OBSERVABILITY_ENABLED not true - skipping ADOT setup"
        )
        return False

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        # Service name and resource attributes
        resource_attrs: dict[str, str] = {SERVICE_NAME: config.otel_service_name}

        # Parse OTEL_RESOURCE_ATTRIBUTES if set (e.g., "key1=val1,key2=val2")
        if config.otel_resource_attributes:
            for pair in config.otel_resource_attributes.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    resource_attrs[k.strip()] = v.strip()

        resource = Resource.create(resource_attrs)

        # ── Tracing ───────────────────────────────────────────────────────────
        # ADOT automatically configures the OTLP exporter endpoint to send to
        # CloudWatch via X-Ray when running in AWS environment. For local dev,
        # it sends to localhost:4318 (OTEL Collector) if available.
        trace_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(trace_exporter)

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)

        _tracer = trace.get_tracer(__name__)

        # ── Metrics ───────────────────────────────────────────────────────────
        metric_exporter = OTLPMetricExporter()
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=60_000,  # Export every 60s
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)

        _meter = metrics.get_meter(__name__)

        _observability_enabled = True

        logger.info(
            "observability: ADOT enabled  service=%r resource_attrs=%s",
            config.otel_service_name,
            resource_attrs,
        )
        return True

    except ImportError as err:
        logger.warning(
            "observability: ADOT packages not installed - %s. "
            "Install with: pip install aws-opentelemetry-distro opentelemetry-api opentelemetry-sdk",
            err,
        )
        return False
    except Exception as err:
        logger.warning("observability: ADOT setup failed - %s", err)
        return False


def get_tracer() -> Any:
    """Return the global tracer instance.

    Returns a no-op tracer if observability is disabled.
    """
    if _tracer is None:
        try:
            from opentelemetry import trace

            return trace.get_tracer(__name__)
        except ImportError:
            return _NoOpTracer()
    return _tracer


def get_meter() -> Any:
    """Return the global meter instance.

    Returns a no-op meter if observability is disabled.
    """
    if _meter is None:
        try:
            from opentelemetry import metrics

            return metrics.get_meter(__name__)
        except ImportError:
            return _NoOpMeter()
    return _meter


def is_enabled() -> bool:
    """Return True if observability is enabled."""
    return _observability_enabled


# ── No-op implementations for when observability is disabled ──────────────────


class _NoOpSpan:
    """No-op span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer that returns no-op spans."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpMeter:
    """No-op meter that does nothing."""

    def create_counter(self, name: str, **kwargs: Any) -> Any:
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> Any:
        return _NoOpHistogram()


class _NoOpCounter:
    """No-op counter that does nothing."""

    def add(self, amount: int | float, attributes: dict | None = None) -> None:
        pass


class _NoOpHistogram:
    """No-op histogram that does nothing."""

    def record(self, amount: int | float, attributes: dict | None = None) -> None:
        pass

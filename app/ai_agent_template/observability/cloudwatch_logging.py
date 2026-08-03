"""CloudWatch Logs integration — attaches a watchtower handler to the root logger.

Every agent_log() / logger.xxx() call in the process is automatically
shipped to CloudWatch Logs once this is set up — no changes needed in
individual modules.

Install:
    pip install '.[cloudwatch]'    # adds watchtower>=3.0

No-op if:
    - cloudwatch_log_group is unset (disabled by default)
    - watchtower is not installed (ImportError is caught and logged)
    - boto3 cannot create a session (missing credentials etc.)
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def setup_cloudwatch_logging(
    log_group: str | None, log_stream: str | None, *, region: str
) -> bool:
    """Attach a CloudWatch Logs handler to the root logger.

    Returns True if the handler was attached, False if disabled / unavailable.
    Call once at startup (in lifespan), after logging.basicConfig().

    Gated independently of ADOT tracing (observability/tracer.py) — a
    deployment may want log shipping without traces/metrics, or vice versa.
    """
    if not log_group:
        logger.debug("cloudwatch_logging: CLOUDWATCH_LOG_GROUP not set - skipping")
        return False

    try:
        import boto3
        import watchtower

        # Default stream name: <hostname>/<UTC-date> — groups logs by host per day,
        # easy to filter in CW Logs Insights. Override with CLOUDWATCH_LOG_STREAM.
        default_stream = (
            f"{socket.gethostname()}/" f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )
        stream = log_stream or default_stream

        # Note: watchtower 3.x uses boto3_client, 4.x uses boto3_session — pinned
        # to the reference's tested 3.x line (pyproject.toml).
        boto_session = boto3.session.Session(region_name=region)
        logs_client = boto_session.client("logs")

        handler = watchtower.CloudWatchLogHandler(
            boto3_client=logs_client,
            log_group_name=log_group,
            log_stream_name=stream,
            # send_interval: flush every 5s (watchtower default 60s — too slow for debugging).
            send_interval=5,
            # create_log_group: auto-create the group if it doesn't exist.
            create_log_group=True,
        )

        # Same format as main.py's console handler, for consistency.
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        # Attach to root logger so every module's logger is covered automatically.
        logging.getLogger().addHandler(handler)

        logger.info(
            "cloudwatch_logging: enabled  group=%r stream=%r region=%s",
            log_group,
            stream,
            region,
        )
        return True

    except ImportError:
        logger.warning(
            "cloudwatch_logging: watchtower not installed - CloudWatch disabled. "
            "Install with: pip install '.[cloudwatch]'"
        )
        return False
    except Exception as err:
        logger.warning("cloudwatch_logging: setup failed - %s", err)
        return False

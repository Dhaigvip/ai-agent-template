"""Output serving — register code-interpreter output files for HTTP download.

Decoupled from execution: `CodeExecutionService.run()` runs code and returns
the raw produced files as `(filename, bytes)`. This module is the separate
*serving* concern — it registers those bytes in the FileRegistry and turns
them into downloadable charts/downloads with `/api/agent/download/<id>` URLs.

Keeping this out of the execution engine means running code never depends on
a file-serving subsystem, and the download-URL convention lives in one place.
"""

from __future__ import annotations

import logging

from ai_agent_template.analysis.file_manager import FileRegistry, get_file_registry
from ai_agent_template.analysis.types import ChartMetadata, DownloadMetadata

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg")


def register_outputs(
    files: list[tuple[str, bytes]],
    file_registry: FileRegistry | None = None,
) -> tuple[list[ChartMetadata], list[DownloadMetadata]]:
    """Register produced files for download and split them into charts vs. data.

    Args:
        files: `(filename, bytes)` pairs from `ExecutionOutcome.files`.
        file_registry: Override registry (defaults to the global one).

    Returns:
        (charts, downloads) — image files vs. everything else, each carrying a
        `/api/agent/download/<file_id>` URL.
    """
    registry = file_registry or get_file_registry()
    charts: list[ChartMetadata] = []
    downloads: list[DownloadMetadata] = []

    for filename, content in files:
        file_id = registry.register_file(filename, content)
        url = f"/api/agent/download/{file_id}"

        lower = filename.lower()
        if lower.endswith(".plotly.json"):
            # Interactive Plotly figure (fig.to_json()). Tagged "plotly" so
            # the client can render it with react-plotly.js instead of <img>.
            charts.append(
                ChartMetadata(filename=filename, url=url, format="plotly", size_bytes=len(content))
            )
        elif lower.endswith(_IMAGE_EXTS):
            charts.append(
                ChartMetadata(
                    filename=filename,
                    url=url,
                    format=filename.split(".")[-1].lower(),
                    size_bytes=len(content),
                )
            )
        else:
            downloads.append(
                DownloadMetadata(
                    filename=filename,
                    url=url,
                    size_bytes=len(content),
                    mime_type=registry._detect_mime_type(filename),
                )
            )
        logger.info("Registered output: %s -> %s", filename, file_id)

    return charts, downloads

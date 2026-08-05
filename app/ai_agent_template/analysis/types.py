"""Data models for the code-execution (`run_python`) tool.

Trimmed from the reference's `analysis/types.py`: this template has no file-
upload / Data Analysis REST tab (`analysis_handler.py`'s own request/response
shapes — `AnalysisRequest`, `AnalysisResult`, `FileInspection`, `AnalysisError`
— are that tab's, not the chat `run_python` tool's, and don't port). Only the
types the chat path actually produces are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChartMetadata:
    """Metadata for a generated chart, ready to embed as markdown."""

    filename: str
    url: str
    format: str  # "plotly" | "png" | "jpg" | "svg" | ...
    size_bytes: int


@dataclass
class DownloadMetadata:
    """Metadata for a downloadable (non-chart) file the code produced."""

    filename: str
    url: str
    size_bytes: int
    mime_type: str


@dataclass
class CodeExecutionResult:
    """Result from executing Python code in the sandbox."""

    output: str  # captured stdout
    files_created: list[str] = field(default_factory=list)  # sandbox-relative names
    error: str | None = None


@dataclass
class ExecutionOutcome:
    """Pure result of running code through CodeExecutionService.

    Holds stdout/error + the raw bytes of any files the code produced, and
    NOTHING about serving them. Turning files into downloadable URLs is a
    separate concern — callers pass `outcome.files` to
    `chart_extraction.register_outputs()`.
    """

    output: str
    error: str | None = None
    files: list[tuple[str, bytes]] = field(default_factory=list)  # (filename, bytes) produced
    files_created: list[str] = field(default_factory=list)  # sandbox names, for reference

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class DownloadFile:
    """A file available for download via the /api/agent/download/<id> route."""

    file_id: str
    filename: str
    content: bytes
    mime_type: str
    created_at: float  # unix timestamp

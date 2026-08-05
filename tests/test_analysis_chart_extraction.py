"""Tests for register_outputs — splits produced files into charts vs downloads."""

from __future__ import annotations

from ai_agent_template.analysis.chart_extraction import register_outputs
from ai_agent_template.analysis.file_manager import FileRegistry


def test_plotly_json_classified_as_chart(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    charts, downloads = register_outputs([("chart.plotly.json", b'{"data": []}')], registry)
    assert len(charts) == 1
    assert charts[0].format == "plotly"
    assert charts[0].url.startswith("/api/agent/download/")
    assert downloads == []


def test_image_file_classified_as_chart(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    charts, downloads = register_outputs([("plot.png", b"\x89PNG...")], registry)
    assert len(charts) == 1
    assert charts[0].format == "png"
    assert downloads == []


def test_other_file_classified_as_download(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    charts, downloads = register_outputs([("export.csv", b"a,b\n1,2\n")], registry)
    assert charts == []
    assert len(downloads) == 1
    # mimetypes' exact CSV mapping varies by OS registry (e.g. Windows maps
    # .csv to application/vnd.ms-excel) — just confirm it's not the
    # unrecognized-type fallback.
    assert downloads[0].mime_type != "application/octet-stream"


def test_mixed_files_split_correctly(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    charts, downloads = register_outputs(
        [("chart.plotly.json", b"{}"), ("export.csv", b"a,b\n")], registry
    )
    assert len(charts) == 1
    assert len(downloads) == 1


def test_no_files_returns_empty_lists(tmp_path):
    registry = FileRegistry(storage_dir=tmp_path)
    charts, downloads = register_outputs([], registry)
    assert charts == []
    assert downloads == []

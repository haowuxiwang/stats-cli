"""Tests for stats_engine/report.py and discover.py."""

import numpy as np

from stats_engine.discover import COMMANDS
from stats_engine.report import report


def test_report_two_sided():
    np.random.seed(42)
    values = np.random.normal(10, 0.5, 100).tolist()
    result = report(values=values, usl=12, lsl=8)
    assert "analyses" in result
    assert "descriptive" in result["analyses"]
    assert "normality" in result["analyses"]
    assert "outlier" in result["analyses"]
    assert "capability" in result["analyses"]
    assert "control_chart" in result["analyses"]
    assert "summary" in result


def test_report_one_sided():
    np.random.seed(42)
    values = np.random.normal(10, 0.5, 100).tolist()
    result = report(values=values, usl=12)
    assert "analyses" in result


def test_discover_commands():
    assert len(COMMANDS) > 20
    assert "descriptive" in COMMANDS
    assert "normality" in COMMANDS
    assert "capability" in COMMANDS
    assert "control_chart" in COMMANDS
    assert "discover" in COMMANDS


def test_discover_command_fields():
    for name, cmd in COMMANDS.items():
        assert "description" in cmd, f"{name} missing description"
        assert "category" in cmd, f"{name} missing category"
        assert "input" in cmd, f"{name} missing input"
        assert "output_fields" in cmd, f"{name} missing output_fields"


def test_export_excel():
    """Export report to Excel."""
    import os
    import tempfile

    from stats_engine.report import export_excel

    np.random.seed(42)
    values = np.random.normal(10, 0.5, 50).tolist()
    report_data = report(values=values, usl=12, lsl=8)

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = export_excel(report_data, output_path=tmp_path)
        assert result["format"] == "excel"
        assert result["output_path"] == tmp_path
        assert result["file_size"] > 0
        assert "sheets" in result
        assert os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_pdf():
    """Export report to PDF."""
    import os
    import tempfile

    from stats_engine.report import export_pdf

    np.random.seed(42)
    values = np.random.normal(10, 0.5, 50).tolist()
    report_data = report(values=values, usl=12, lsl=8)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = export_pdf(report_data, output_path=tmp_path)
        assert result["format"] == "pdf"
        assert result["output_path"] == tmp_path
        assert result["file_size"] > 0
        assert os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_excel_via_handler():
    """Export Excel through handler."""
    import os
    import tempfile

    from main import handler

    np.random.seed(42)
    values = np.random.normal(10, 0.5, 50).tolist()
    report_data = report(values=values, usl=12, lsl=8)

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = handler({"command": "export_excel", "params": {"report_data": report_data, "output_path": tmp_path}})
        assert result["status"] == "success"
        assert result["data"]["format"] == "excel"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_pdf_via_handler():
    """Export PDF through handler."""
    import os
    import tempfile

    from main import handler

    np.random.seed(42)
    values = np.random.normal(10, 0.5, 50).tolist()
    report_data = report(values=values, usl=12, lsl=8)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = handler({"command": "export_pdf", "params": {"report_data": report_data, "output_path": tmp_path}})
        assert result["status"] == "success"
        assert result["data"]["format"] == "pdf"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

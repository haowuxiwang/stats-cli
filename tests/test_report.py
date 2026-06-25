"""Tests for stats_engine/report.py and discover.py."""

import builtins
import importlib
import sys

import numpy as np
import pytest

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


class TestReportImportErrors:
    """Test ImportError guards in report export functions."""

    def test_export_excel_pandas_import_error(self):
        """export_excel raises ImportError when pandas is missing (lines 78-79)."""

        saved = {}
        keys_to_remove = [k for k in sys.modules if k in ("pandas", "openpyxl")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name in ("pandas", "openpyxl"):
                raise ImportError(f"mocked: no module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        importlib.reload(sys.modules["stats_engine.report"])
        from stats_engine.report import export_excel as export_excel_fresh

        try:
            with pytest.raises(ImportError, match="pandas and openpyxl required"):
                export_excel_fresh({"summary": {}}, output_path="/tmp/test.xlsx")
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(sys.modules["stats_engine.report"])

    def test_export_pdf_fpdf2_import_error(self):
        """export_pdf raises ImportError when fpdf2 is missing (lines 176-177)."""

        saved = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("fpdf")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "fpdf" or name.startswith("fpdf."):
                raise ImportError(f"mocked: no module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        importlib.reload(sys.modules["stats_engine.report"])
        from stats_engine.report import export_pdf as export_pdf_fresh

        try:
            with pytest.raises(ImportError, match="fpdf2 required"):
                export_pdf_fresh({"summary": {}}, output_path="/tmp/test.pdf")
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(sys.modules["stats_engine.report"])

    def test_register_cjk_font_linux_paths(self):
        """_register_cjk_font tries Linux font paths on Linux (lines 141-147)."""
        from unittest.mock import MagicMock, patch

        from stats_engine.report import _register_cjk_font

        pdf = MagicMock()
        pdf.add_font = MagicMock()

        # platform and Path are imported locally, so patch at the stdlib level
        with patch("platform.system", return_value="Linux"), patch("pathlib.Path.exists", return_value=True):
            font = _register_cjk_font(pdf)

        # Should have tried to add a font and returned the font name
        assert pdf.add_font.called
        assert font != "Helvetica"

    def test_register_cjk_font_add_font_error(self):
        """_register_cjk_font falls back to Helvetica when add_font raises (lines 158-160)."""
        from unittest.mock import MagicMock, patch

        from stats_engine.report import _register_cjk_font

        pdf = MagicMock()
        pdf.add_font.side_effect = Exception("font load failed")

        with patch("platform.system", return_value="Linux"), patch("pathlib.Path.exists", return_value=True):
            font = _register_cjk_font(pdf)

        # Should fall back to Helvetica after all fonts fail
        assert font == "Helvetica"

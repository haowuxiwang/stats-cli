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

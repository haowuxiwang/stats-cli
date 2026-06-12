"""Tests for main.py handler and routing."""

import json

import pytest

from main import handler


def test_handler_string_input():
    result = handler('{"command": "descriptive", "params": {"values": [1, 2, 3]}}')
    assert result["status"] == "success"
    assert "data" in result


def test_handler_dict_input():
    result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "success"


def test_handler_invalid_json():
    result = handler("not json")
    assert result["status"] == "error"
    assert result["error_type"] == "INVALID_INPUT"


def test_handler_missing_command():
    result = handler({"params": {}})
    assert result["status"] == "error"
    assert result["error_type"] == "MISSING_COMMAND"


def test_handler_unknown_command():
    result = handler({"command": "nonexistent"})
    assert result["status"] == "error"
    assert result["error_type"] == "VALIDATION_ERROR"


def test_handler_validation_error():
    result = handler({"command": "capability", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "error"
    assert result["error_type"] == "VALIDATION_ERROR"


def test_handler_discover():
    result = handler({"command": "discover", "params": {}})
    assert result["status"] == "success"
    assert "total_commands" in result["data"]


def test_handler_discover_specific():
    result = handler({"command": "discover", "params": {"command_name": "descriptive"}})
    assert result["status"] == "success"
    assert "description" in result["data"]


def test_handler_discover_category():
    result = handler({"command": "discover", "params": {"category": "basic"}})
    assert result["status"] == "success"
    assert result["data"]["total_commands"] > 0


def test_handler_params_none():
    """params=null must not crash."""
    result = handler({"command": "descriptive", "params": None})
    # Should return an error, not crash with TypeError
    assert result["status"] == "error"


JSON_SERIALIZABLE_CASES = [
    {"command": "descriptive", "params": {"values": [1, 2, 3, 4, 5]}},
    {"command": "normality", "params": {"values": [1, 2, 3, 4, 5]}},
    {"command": "capability", "params": {"values": [9.5, 10.2, 10.5, 10.1, 9.8, 10.3, 10.0, 9.9, 10.4, 10.2] * 3, "usl": 12, "lsl": 8}},
    {"command": "control_chart", "params": {"chart_type": "imr", "values": [10] * 25}},
    {"command": "ttest", "params": {"test_type": "one_sample", "values": [10, 12, 11, 13, 14], "mu": 12}},
    {"command": "anova", "params": {"anova_type": "one_way", "groups": [[1, 2, 3], [4, 5, 6]]}},
    {"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}},
    {"command": "correlation", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}},
    {"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": [1, 2, 3], "y": [4, 5, 6]}},
    {"command": "homogeneity", "params": {"test_type": "levene", "groups": [[1, 2, 3], [4, 5, 6]]}},
    {"command": "multiple_comparison", "params": {"test_type": "tukey", "groups": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}},
    {"command": "equivalence", "params": {"test_type": "tost", "values": [10, 11, 12], "values2": [10.5, 11.5, 12.5], "delta": 1}},
    {"command": "outlier", "params": {"values": [1, 2, 3, 4, 100], "method": "grubbs"}},
    {"command": "trend", "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "test_type": "runs"}},
    {"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}]}},
    {"command": "reliability", "params": {"analysis_type": "weibull", "times": [100, 200, 300], "status": [1, 1, 1]}},
    {"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}},
    {"command": "clean", "params": {"values": [1, 2, None, 4, 5], "method": "drop"}},
    {"command": "transform", "params": {"values": [1, 2, 3, 4, 5], "method": "log"}},
    {"command": "discover", "params": {}},
    {"command": "advanced", "params": {"analysis_type": "exact_test", "observed": [[10, 5], [3, 12]]}},
    {"command": "check_assumptions", "params": {"values": [10, 11, 12], "test_type": "ttest"}},
    {"command": "recommend", "params": {"data_description": {"goal": "compare means", "n_groups": 2}}},
    {"command": "workflow", "params": {"steps": [{"command": "descriptive"}], "values": [1, 2, 3]}},
    {"command": "workflow_template", "params": {"template_name": "manufacturing", "values": [10, 11, 12], "usl": 12, "lsl": 8}},
]


@pytest.mark.parametrize("case", JSON_SERIALIZABLE_CASES, ids=lambda c: c["command"])
def test_json_serializable(case):
    """Every command's output must be JSON-serializable (no numpy.bool_ leak)."""
    result = handler(case)
    assert result["status"] == "success", f"{case['command']}: {result.get('message', '')}"
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_json_serializable_gage_rr():
    """Gage R&R needs special setup — test separately."""
    import numpy as np
    np.random.seed(42)
    parts = list(range(1, 7)) * 6  # 6 parts, 3 operators, 2 replicates
    operators = (["O1"] * 6 + ["O2"] * 6 + ["O3"] * 6) * 2
    measurements = np.random.normal(100, 2, 36).tolist()
    result = handler({
        "command": "gage_rr",
        "params": {
            "analysis_type": "crossed",
            "measurements": measurements,
            "parts": parts,
            "operators": operators,
        },
    })
    assert result["status"] == "success"
    json.dumps(result)

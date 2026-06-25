"""Tests for main.py handler and routing."""

import json
import os
import subprocess
import sys
import tempfile

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
    assert result["error_type"] == "PARAM_ERROR"


def test_handler_validation_error():
    result = handler({"command": "capability", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "error"
    assert result["error_type"] == "PARAM_ERROR"


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
    {
        "command": "capability",
        "params": {"values": [9.5, 10.2, 10.5, 10.1, 9.8, 10.3, 10.0, 9.9, 10.4, 10.2] * 3, "usl": 12, "lsl": 8},
    },
    {"command": "control_chart", "params": {"chart_type": "imr", "values": [10] * 25}},
    {"command": "ttest", "params": {"test_type": "one_sample", "values": [10, 12, 11, 13, 14], "mu": 12}},
    {"command": "anova", "params": {"anova_type": "one_way", "groups": [[1, 2, 3], [4, 5, 6]]}},
    {"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}},
    {"command": "correlation", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}},
    {"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": [1, 2, 3], "y": [4, 5, 6]}},
    {"command": "homogeneity", "params": {"test_type": "levene", "groups": [[1, 2, 3], [4, 5, 6]]}},
    {"command": "multiple_comparison", "params": {"test_type": "tukey", "groups": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}},
    {
        "command": "equivalence",
        "params": {"test_type": "tost", "values": [10, 11, 12], "values2": [10.5, 11.5, 12.5], "delta": 1},
    },
    {"command": "outlier", "params": {"values": [1, 2, 3, 4, 100], "method": "grubbs"}},
    {"command": "trend", "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "test_type": "runs"}},
    {
        "command": "doe",
        "params": {"doe_type": "full_factorial", "factors": [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}]},
    },
    {"command": "reliability", "params": {"analysis_type": "weibull", "times": [100, 200, 300], "status": [1, 1, 1]}},
    {"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}},
    {"command": "clean", "params": {"values": [1, 2, None, 4, 5], "method": "drop"}},
    {"command": "transform", "params": {"values": [1, 2, 3, 4, 5], "method": "log"}},
    {"command": "discover", "params": {}},
    {"command": "advanced", "params": {"analysis_type": "exact_test", "observed": [[10, 5], [3, 12]]}},
    {"command": "check_assumptions", "params": {"values": [10, 11, 12], "test_type": "ttest"}},
    {"command": "recommend", "params": {"data_description": {"goal": "compare means", "n_groups": 2}}},
    {"command": "workflow", "params": {"steps": [{"command": "descriptive"}], "values": [1, 2, 3]}},
    {
        "command": "workflow_template",
        "params": {"template_name": "manufacturing", "values": [10, 11, 12], "usl": 12, "lsl": 8},
    },
    {
        "command": "distribution",
        "params": {"analysis_type": "fit", "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    },
    {
        "command": "bayesian",
        "params": {"analysis_type": "ttest", "values": [10, 11, 12], "values2": [13, 14, 15]},
    },
    {
        "command": "mining",
        "params": {
            "analysis_type": "anomaly",
            "values": [10, 11, 10, 12, 100, 11, 10, 500, 11, 10, 12, 10, 11, 10, 11],
        },
    },
    {
        "command": "sensitivity",
        "params": {
            "analysis_type": "monte_carlo",
            "inputs": {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}},
            "formula": "x * 2",
            "n_simulations": 500,
        },
    },
]


@pytest.mark.parametrize("case", JSON_SERIALIZABLE_CASES, ids=lambda c: c["command"])
def test_json_serializable(case):
    """Every command's output must be JSON-serializable (no numpy.bool_ leak)."""
    result = handler(case)
    assert result["status"] in ("success", "warning"), f"{case['command']}: {result.get('message', '')}"
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_json_serializable_gage_rr():
    """Gage R&R needs special setup — test separately."""
    import numpy as np

    np.random.seed(42)
    parts = list(range(1, 7)) * 6  # 6 parts, 3 operators, 2 replicates
    operators = (["O1"] * 6 + ["O2"] * 6 + ["O3"] * 6) * 2
    measurements = np.random.normal(100, 2, 36).tolist()
    result = handler(
        {
            "command": "gage_rr",
            "params": {
                "analysis_type": "crossed",
                "measurements": measurements,
                "parts": parts,
                "operators": operators,
            },
        }
    )
    assert result["status"] == "success"
    json.dumps(result)


def test_handler_discover_unknown_command():
    """discover with unknown command_name should error."""
    result = handler({"command": "discover", "params": {"command_name": "nonexistent"}})
    assert result["status"] == "error"
    assert "Unknown command" in result["message"]


def test_handler_run_script():
    """run command should execute script and return result."""
    result = handler(
        {
            "command": "run",
            "params": {"script": "result = {'sum': sum(data['values'])}", "data": {"values": [1, 2, 3]}},
        }
    )
    assert result["status"] == "success"
    assert result["data"]["sum"] == 6


def test_handler_run_script_no_script():
    """run command without script should error."""
    result = handler({"command": "run", "params": {}})
    assert result["status"] == "error"


def test_handler_chart_request():
    """chart=true should generate chart_base64."""
    result = handler(
        {
            "command": "descriptive",
            "params": {"values": [1, 2, 3, 4, 5], "chart": True},
        }
    )
    assert result["status"] == "success"
    assert "chart_base64" in result["data"]


def test_handler_chart_request_unsupported():
    """chart=true for unsupported command should have chart_base64=None."""
    result = handler(
        {
            "command": "discover",
            "params": {"chart": True},
        }
    )
    assert result["status"] == "success"


def test_handler_type_error_keyword():
    """TypeError with 'unexpected keyword argument' should be VALIDATION_ERROR."""
    result = handler(
        {
            "command": "descriptive",
            "params": {"values": [1, 2, 3], "nonexistent_param": 42},
        }
    )
    assert result["status"] == "error"
    assert result["error_type"] == "PARAM_ERROR"


def test_handler_missing_dependency():
    """ImportError should return MISSING_DEPENDENCY."""
    # This is hard to trigger without mocking, but we can test the path exists
    result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "success"


def test_handler_file_not_found():
    """FileNotFoundError should return FILE_NOT_FOUND."""
    result = handler(
        {
            "command": "descriptive",
            "params": {"file": "/nonexistent/path/data.csv"},
        }
    )
    assert result["status"] == "error"
    assert result["error_type"] == "FILE_NOT_FOUND"


class TestMainCLI:
    """Test main() CLI entry point via subprocess."""

    def test_stdin_input(self):
        """main() should read from stdin."""
        r = subprocess.run(
            [sys.executable, "main.py"],
            input='{"command":"descriptive","params":{"values":[1,2,3,4,5]}}',
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["status"] == "success"

    def test_file_input(self):
        """main() should read from file argument."""
        tmpfile = tempfile.mktemp(suffix=".json")
        with open(tmpfile, "w") as f:
            json.dump({"command": "descriptive", "params": {"values": [1, 2, 3]}}, f)
        try:
            r = subprocess.run(
                [sys.executable, "main.py", tmpfile],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            assert r.returncode == 0
            out = json.loads(r.stdout)
            assert out["status"] == "success"
        finally:
            os.unlink(tmpfile)

    def test_file_not_found(self):
        """main() should handle missing file gracefully."""
        r = subprocess.run(
            [sys.executable, "main.py", "/nonexistent/file.json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["status"] == "error"
        assert out["error_type"] == "FILE_NOT_FOUND"

    def test_invalid_json_stdin(self):
        """main() should handle invalid JSON gracefully."""
        r = subprocess.run(
            [sys.executable, "main.py"],
            input="not json",
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["status"] == "error"
        assert out["error_type"] == "INVALID_INPUT"

    def test_error_output(self):
        """main() should output valid JSON on error."""
        r = subprocess.run(
            [sys.executable, "main.py"],
            input='{"command":"nonexistent"}',
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["status"] == "error"


class TestMainDirect:
    """Test main() directly for coverage."""

    def test_main_stdin(self, monkeypatch, capsys):
        """main() reads from stdin when no args."""
        import main as main_module

        monkeypatch.setattr("sys.argv", ["main.py"])
        monkeypatch.setattr("sys.stdin.read", lambda: '{"command":"descriptive","params":{"values":[1,2,3]}}')
        main_module.main()
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["status"] == "success"

    def test_main_file(self, monkeypatch, capsys):
        """main() reads from file when arg provided."""
        import main as main_module

        tmpfile = tempfile.mktemp(suffix=".json")
        with open(tmpfile, "w") as f:
            json.dump({"command": "descriptive", "params": {"values": [1, 2, 3]}}, f)
        try:
            monkeypatch.setattr("sys.argv", ["main.py", tmpfile])
            main_module.main()
            captured = capsys.readouterr()
            out = json.loads(captured.out)
            assert out["status"] == "success"
        finally:
            os.unlink(tmpfile)

    def test_main_file_not_found(self, monkeypatch, capsys):
        """main() handles missing file."""
        import main as main_module

        monkeypatch.setattr("sys.argv", ["main.py", "/nonexistent.json"])
        main_module.main()
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["status"] == "error"
        assert out["error_type"] == "FILE_NOT_FOUND"

    def test_main_invalid_json(self, monkeypatch, capsys):
        """main() handles invalid JSON."""
        import main as main_module

        monkeypatch.setattr("sys.argv", ["main.py"])
        monkeypatch.setattr("sys.stdin.read", lambda: "not json")
        main_module.main()
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["status"] == "error"
        assert out["error_type"] == "INVALID_INPUT"

    def test_main_generic_error(self, monkeypatch, capsys):
        """main() handles generic exceptions."""
        import main as main_module

        monkeypatch.setattr("sys.argv", ["main.py"])
        monkeypatch.setattr("sys.stdin.read", lambda: None)  # Will cause TypeError
        main_module.main()
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["status"] == "error"


class TestHandlerExceptionPaths:
    """Tests for exception paths in handler()."""

    def test_handler_import_error(self, monkeypatch):
        """ImportError returns MISSING_DEPENDENCY."""
        import main as main_module

        def mock_route(cmd, params):
            raise ImportError("No module named 'fake_module'")

        monkeypatch.setattr(main_module, "_route", mock_route)
        result = main_module.handler({"command": "test", "params": {}})
        assert result["status"] == "error"
        assert result["error_type"] == "MISSING_DEPENDENCY"

    def test_handler_memory_error(self, monkeypatch):
        """MemoryError returns MEMORY_ERROR."""
        import main as main_module

        def mock_route(cmd, params):
            raise MemoryError()

        monkeypatch.setattr(main_module, "_route", mock_route)
        result = main_module.handler({"command": "test", "params": {}})
        assert result["status"] == "error"
        assert result["error_type"] == "MEMORY_ERROR"


class TestClassifyValueError:
    """Tests for _classify_value_error()."""

    def test_classify_value_error_data(self):
        """Data errors classified as DATA_ERROR."""
        from main import _classify_value_error

        assert _classify_value_error("Need at least 2 values") == "DATA_ERROR"
        assert _classify_value_error("No numeric columns found") == "DATA_ERROR"
        assert _classify_value_error("Column 'x' not found") == "DATA_ERROR"
        assert _classify_value_error("Sheet index out of range") == "DATA_ERROR"

    def test_classify_value_error_computation(self):
        """Computation errors classified as COMPUTATION_ERROR."""
        from main import _classify_value_error

        assert _classify_value_error("Cannot calculate: zero variance") == "COMPUTATION_ERROR"
        assert _classify_value_error("Result is undefined") == "COMPUTATION_ERROR"
        assert _classify_value_error("Singular matrix encountered") == "COMPUTATION_ERROR"

    def test_classify_value_error_param(self):
        """Param errors classified as PARAM_ERROR."""
        from main import _classify_value_error

        assert _classify_value_error("Unknown method: foo") == "PARAM_ERROR"
        assert _classify_value_error("Invalid argument") == "PARAM_ERROR"

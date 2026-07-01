"""Additional tests for main.py to improve coverage from 86% to 97%."""

from unittest.mock import MagicMock, patch

from main import handler


class TestHandlerExceptionPaths:
    """Test exception handling paths in handler()."""

    def test_handler_permission_error(self):
        """PermissionError returns PARAM_ERROR with suggestion."""
        with patch("main._route", side_effect=PermissionError("Access denied")):
            result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
            assert result["status"] == "error"
            assert result["error_type"] == "PARAM_ERROR"
            assert "Permission denied" in result["message"]

    def test_handler_type_error_generic(self):
        """Generic TypeError (not keyword/missing) returns PARAM_ERROR."""
        with patch("main._route", side_effect=TypeError("conversion failed")):
            result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
            assert result["status"] == "error"
            assert result["error_type"] == "PARAM_ERROR"
            assert "Invalid parameter type" in result["message"]

    def test_handler_type_error_unexpected_keyword(self):
        """TypeError with 'unexpected keyword' returns PARAM_ERROR."""
        with patch("main._route", side_effect=TypeError("unexpected keyword argument 'foo'")):
            result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
            assert result["status"] == "error"
            assert result["error_type"] == "PARAM_ERROR"

    def test_handler_type_error_missing_required(self):
        """TypeError with 'missing required' returns PARAM_ERROR."""
        with patch("main._route", side_effect=TypeError("missing required argument: 'values'")):
            result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
            assert result["status"] == "error"
            assert result["error_type"] == "PARAM_ERROR"
            assert "Missing required parameter" in result["message"]


class TestTwoColumnFileLoading:
    """Test two-column file loading for correlation/regression."""

    def test_correlation_with_file_columns(self, tmp_path):
        """Correlation command with file+x_column+y_column loads data."""
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for i in range(20):
                writer.writerow([i, i * 2 + 1])

        result = handler(
            {
                "command": "correlation",
                "params": {
                    "file": str(csv_file),
                    "x_column": "x",
                    "y_column": "y",
                },
            }
        )
        assert result["status"] == "success"
        assert "correlation" in result["data"]

    def test_regression_with_file_columns(self, tmp_path):
        """Regression command with file+x_column+y_column loads data."""
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for i in range(20):
                writer.writerow([i, i * 2 + 1])

        result = handler(
            {
                "command": "regression",
                "params": {
                    "file": str(csv_file),
                    "x_column": "x",
                    "y_column": "y",
                    "reg_type": "linear",
                },
            }
        )
        assert result["status"] == "success"
        assert "r_squared" in result["data"]


class TestRunScriptSecurity:
    """Test _run_script security checks (AST-based)."""

    def test_run_script_blocked_dunder(self):
        """Script with dunder access is blocked."""
        result = handler(
            {
                "command": "run",
                "params": {"script": "x = data.__class__"},
            }
        )
        assert result["status"] == "error"
        assert "dunder" in result["message"].lower()

    def test_run_script_blocked_import(self):
        """Script with import statement is blocked."""
        result = handler(
            {
                "command": "run",
                "params": {"script": "import os"},
            }
        )
        assert result["status"] == "error"
        assert "import" in result["message"].lower()

    def test_run_script_blocked_from_import(self):
        """Script with 'from X import Y' is blocked."""
        result = handler(
            {
                "command": "run",
                "params": {"script": "from os import listdir"},
            }
        )
        assert result["status"] == "error"
        assert "import" in result["message"].lower()

    def test_run_script_blocked_dunder_name(self):
        """Script with __builtins__ reference is blocked."""
        result = handler(
            {
                "command": "run",
                "params": {"script": "x = __builtins__"},
            }
        )
        assert result["status"] == "error"
        assert "dunder" in result["message"].lower()

    def test_run_script_safe_arithmetic(self):
        """Safe arithmetic script works."""
        result = handler(
            {
                "command": "run",
                "params": {
                    "script": 'result = sum(data["values"]) / len(data["values"])',
                    "data": {"values": [1, 2, 3, 4, 5]},
                },
            }
        )
        assert result["status"] == "success"
        assert result["data"] == 3.0

    def test_run_script_safe_list_comp(self):
        """Safe list comprehension works."""
        result = handler(
            {
                "command": "run",
                "params": {"script": 'result = [x**2 for x in data["values"]]', "data": {"values": [1, 2, 3]}},
            }
        )
        assert result["status"] == "success"
        assert result["data"] == [1, 4, 9]


class TestRunScriptExecution:
    """Test _run_script execution paths."""

    def test_run_script_normal_execution(self):
        """Normal script executes and returns result."""
        result = handler(
            {
                "command": "run",
                "params": {
                    "script": "result = {'sum': sum(data['values'])}",
                    "data": {"values": [1, 2, 3, 4, 5]},
                },
            }
        )
        assert result["status"] == "success"
        assert result["data"]["sum"] == 15

    def test_run_script_runtime_error(self):
        """Script with runtime error returns error."""
        result = handler(
            {
                "command": "run",
                "params": {
                    "script": "x = 1 / 0",
                    "data": {},
                },
            }
        )
        assert result["status"] == "error"
        assert "division by zero" in result["message"].lower() or "ZeroDivisionError" in result["message"]

    def test_run_script_no_script(self):
        """Missing script parameter returns error."""
        result = handler(
            {
                "command": "run",
                "params": {"data": {}},
            }
        )
        assert result["status"] == "error"

    def test_run_script_timeout(self):
        """Script timeout returns error."""
        result = handler(
            {
                "command": "run",
                "params": {
                    "script": "while True: pass",
                    "timeout": 1,
                },
            }
        )
        assert result["status"] == "error"
        assert "timed out" in result["message"].lower()


class TestGenerateChart:
    """Test _generate_chart error handling."""

    def test_generate_chart_exception_handled(self):
        """Chart generation exception is caught and returns None."""
        with (
            patch("main._route", return_value={"mean": 10.0}),
            patch(
                "stats_engine.chart_handlers.CHART_HANDLERS",
                {"descriptive": MagicMock(side_effect=Exception("chart error"))},
            ),
        ):
            result = handler(
                {
                    "command": "descriptive",
                    "params": {"values": [1, 2, 3, 4, 5]},
                    "chart": True,
                }
            )
            # Should not crash, chart error is handled
            assert result["status"] == "success"


class TestMainEntryPoint:
    """Test main() entry point."""

    def test_main_broken_pipe(self, capsys):
        """BrokenPipeError in print is handled silently."""
        with patch("main.sys") as mock_sys:
            mock_sys.argv = ["main.py"]
            mock_sys.stdin.read.return_value = '{"command": "descriptive", "params": {"values": [1,2,3]}}'
            with patch("main.print", side_effect=BrokenPipeError):
                # Should not raise
                from main import main

                main()


class TestWarningMarker:
    """Test _warning marker handling."""

    def test_handler_warning_marker(self):
        """Command returning _warning marker becomes warning status."""
        with patch("main._route") as mock_route:
            mock_route.return_value = {"result": "ok", "_warning": "some warning"}
            result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
            assert result["status"] == "warning"

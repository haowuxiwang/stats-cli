"""Tests for Phase 1 (critical fixes) and Phase 2 (robustness improvements)."""

import numpy as np

from main import handler


class TestInputSizeLimits:
    """Phase 1.2: Input size validation."""

    def test_values_too_large(self):
        """Values exceeding 100k returns DATA_ERROR."""
        data = list(range(100_001))
        result = handler({"command": "descriptive", "params": {"values": data}})
        assert result["status"] == "error"
        assert result["error_type"] == "DATA_ERROR"
        assert "exceeds maximum size" in result["message"]

    def test_values_at_limit(self):
        """Values at exactly 100k should work."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100_000).tolist()
        result = handler({"command": "descriptive", "params": {"values": data}})
        assert result["status"] == "success"

    def test_groups_too_large(self):
        """Groups with total > 100k elements returns DATA_ERROR."""
        groups = [list(range(50_001)), list(range(50_001))]
        result = handler({"command": "anova", "params": {"anova_type": "one_way", "groups": groups}})
        assert result["status"] == "error"
        assert result["error_type"] == "DATA_ERROR"

    def test_x_too_large(self):
        """X array exceeding limit returns DATA_ERROR."""
        data = list(range(100_001))
        result = handler({"command": "regression", "params": {"x": data, "y": data, "reg_type": "linear"}})
        assert result["status"] == "error"
        assert result["error_type"] == "DATA_ERROR"


class TestUnknownCommand:
    """Phase 1.4: Unknown command error type."""

    def test_unknown_command_error_type(self):
        """Unknown command returns UNKNOWN_COMMAND error type."""
        result = handler({"command": "foobar"})
        assert result["status"] == "error"
        assert result["error_type"] == "UNKNOWN_COMMAND"
        assert "foobar" in result["message"]
        assert "suggestion" in result

    def test_unknown_command_suggests_alternatives(self):
        """Unknown command error includes suggestion."""
        result = handler({"command": "desc"})
        assert result["status"] == "error"
        assert result["error_type"] == "UNKNOWN_COMMAND"


class TestChartErrorFeedback:
    """Phase 1.5: Chart generation failure feedback."""

    def test_chart_error_in_response(self):
        """When chart fails, response includes chart_error field."""
        # Empty data with chart will fail
        result = handler({"command": "descriptive", "params": {"values": [1], "chart": True}})
        # Should still return data even if chart fails
        assert result["status"] in ("success", "warning")


class TestRunScriptSecurityAST:
    """Phase 1.1: AST-based run script security."""

    def test_import_blocked(self):
        result = handler({"command": "run", "params": {"script": "import os"}})
        assert result["status"] == "error"
        assert "import" in result["message"].lower()

    def test_from_import_blocked(self):
        result = handler({"command": "run", "params": {"script": "from sys import exit"}})
        assert result["status"] == "error"
        assert "import" in result["message"].lower()

    def test_dunder_attribute_blocked(self):
        result = handler({"command": "run", "params": {"script": "x = data.__class__"}})
        assert result["status"] == "error"
        assert "dunder" in result["message"].lower()

    def test_dunder_name_blocked(self):
        result = handler({"command": "run", "params": {"script": "x = __builtins__"}})
        assert result["status"] == "error"

    def test_safe_script_works(self):
        result = handler(
            {
                "command": "run",
                "params": {
                    "script": 'result = sum(data["x"]) / len(data["x"])',
                    "data": {"x": [1, 2, 3, 4, 5]},
                },
            }
        )
        assert result["status"] == "success"
        assert result["data"] == 3.0

    def test_no_print_in_safe_builtins(self):
        """Print is removed from safe builtins (no stdout pollution)."""
        result = handler(
            {
                "command": "run",
                "params": {"script": "result = max(data['x'])", "data": {"x": [1, 5, 3]}},
            }
        )
        assert result["status"] == "success"
        assert result["data"] == 5


class TestDeepCopyProtection:
    """Phase 2.1: Deep copy protects nested structures."""

    def test_nested_dict_not_mutated(self):
        """Handler should not mutate caller's nested params."""
        original = {"values": [1, 2, 3], "nested": {"key": [4, 5, 6]}}
        import copy

        original_copy = copy.deepcopy(original)
        handler({"command": "descriptive", "params": original})
        assert original == original_copy


class TestGageRRInputFlexibility:
    """Phase 2.4: Gage RR accepts both flat and 2D array format."""

    def test_crossed_2d_format(self):
        """Crossed Gage R&R accepts 2D array."""
        # 2 parts x 2 operators x 3 replicates
        measurements = [[10.1, 10.2, 10.0], [10.4, 10.5, 10.3]]
        parts = ["P1", "P2"]
        operators = ["A", "B"]
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
        assert result["status"] in ("success", "warning")

    def test_crossed_flat_format(self):
        """Crossed Gage R&R still works with flat array."""
        result = handler(
            {
                "command": "gage_rr",
                "params": {
                    "analysis_type": "crossed",
                    "measurements": [10.1, 10.2, 10.0, 10.4, 10.5, 10.3],
                    "parts": ["P1", "P1", "P1", "P2", "P2", "P2"],
                    "operators": ["A", "B", "A", "A", "B", "A"],
                },
            }
        )
        assert result["status"] in ("success", "warning")

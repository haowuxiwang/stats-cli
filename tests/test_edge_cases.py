"""Edge case tests for boundary conditions."""

from main import handler


class TestAllNaN:
    """Test commands with all-NaN input."""

    def test_descriptive_all_nan(self):
        result = handler({"command": "descriptive", "params": {"values": [float("nan"), float("nan")]}})
        assert result["status"] == "error"

    def test_normality_all_nan(self):
        result = handler({"command": "normality", "params": {"values": [float("nan"), float("nan")]}})
        assert result["status"] == "error"

    def test_correlation_all_nan(self):
        result = handler({"command": "correlation", "params": {"x": [float("nan")], "y": [float("nan")]}})
        assert result["status"] == "error"

    def test_ttest_all_nan(self):
        result = handler({"command": "ttest", "params": {"values": [float("nan"), float("nan")]}})
        assert result["status"] == "error"


class TestZeroVariance:
    """Test commands with zero variance (identical values)."""

    def test_correlation_identical(self):
        result = handler({"command": "correlation", "params": {"x": [5, 5, 5], "y": [5, 5, 5]}})
        assert result["status"] in ["error", "warning"]

    def test_homogeneity_identical(self):
        result = handler({"command": "homogeneity", "params": {"test_type": "levene", "groups": [[5, 5, 5], [5, 5, 5]]}})
        # Should either succeed, warning, or return error, not crash
        assert result["status"] in ["success", "error", "warning"]

    def test_regression_identical_x(self):
        result = handler({"command": "regression", "params": {"x": [5, 5, 5, 5, 5], "y": [1, 2, 3, 4, 5]}})
        assert result["status"] == "error"


class TestEmptyInput:
    """Test commands with empty input."""

    def test_descriptive_empty(self):
        result = handler({"command": "descriptive", "params": {"values": []}})
        assert result["status"] == "error"

    def test_ttest_empty(self):
        result = handler({"command": "ttest", "params": {"values": []}})
        assert result["status"] == "error"

    def test_anova_empty(self):
        result = handler({"command": "anova", "params": {"groups": []}})
        assert result["status"] == "error"

    def test_workflow_empty_steps(self):
        result = handler({"command": "workflow", "params": {"steps": [], "values": [1, 2, 3]}})
        # Empty steps may return success with empty results
        assert result["status"] in ["success", "error"]


class TestSingleValue:
    """Test commands with single value."""

    def test_descriptive_single(self):
        result = handler({"command": "descriptive", "params": {"values": [5.0]}})
        # Single value may return warning due to insufficient data for CI
        assert result["status"] in ["success", "warning"]
        assert result["data"]["mean"] == 5.0

    def test_ttest_single(self):
        result = handler({"command": "ttest", "params": {"values": [5.0]}})
        assert result["status"] == "error"


class TestMissingParams:
    """Test commands with missing required parameters."""

    def test_ttest_missing_values2(self):
        result = handler({"command": "ttest", "params": {"test_type": "two_sample", "values": [1, 2, 3]}})
        assert result["status"] == "error"
        assert result["error_type"] == "PARAM_ERROR"

    def test_capability_missing_limits(self):
        result = handler({"command": "capability", "params": {"values": [1, 2, 3]}})
        assert result["status"] == "error"

    def test_regression_missing_y(self):
        result = handler({"command": "regression", "params": {"x": [1, 2, 3]}})
        assert result["status"] == "error"

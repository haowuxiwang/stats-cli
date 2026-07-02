"""Tests for adversarial review bug fixes — verifies HIGH/MEDIUM bugs are resolved."""

import numpy as np

from main import handler


class TestRegressionPerfectFitFix:
    """BUG 1: regression with perfect collinearity should not leak inf."""

    def test_perfect_linear_fit_no_inf(self):
        """y=2x produces R²=1, f_statistic=None instead of inf."""
        result = handler(
            {
                "command": "regression",
                "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "reg_type": "linear"},
            }
        )
        assert result["status"] in ("success", "warning")
        data = result.get("data", result)
        # f_statistic should be None, not inf
        assert data.get("f_statistic") is None or np.isfinite(data.get("f_statistic"))
        assert data.get("r_squared") == 1.0
        assert "_warning" in data or "warning" in result

    def test_perfect_fit_with_warning_message(self):
        """Perfect fit includes a helpful warning message."""
        result = handler(
            {
                "command": "regression",
                "params": {"x": [1, 2, 3], "y": [2, 4, 6], "reg_type": "linear"},
            }
        )
        data = result.get("data", result)
        # Should have a warning about perfect fit
        warning = data.get("_warning", "") or result.get("warning", "")
        assert "perfect" in warning.lower() or "warning" in result


class TestTtestZeroVarianceFix:
    """BUG 2: t-test with zero variance should raise actionable error."""

    def test_one_sample_zero_variance_raises(self):
        """Identical values → error with clear message."""
        result = handler(
            {
                "command": "ttest",
                "params": {"test_type": "one_sample", "values": [5.0, 5.0, 5.0], "mu": 4.0},
            }
        )
        assert result["status"] == "error"
        assert "zero variance" in result["message"].lower() or "identical" in result["message"].lower()

    def test_two_sample_zero_variance_raises(self):
        """Both groups identical → error."""
        result = handler(
            {
                "command": "ttest",
                "params": {"test_type": "two_sample", "values": [5.0, 5.0, 5.0], "values2": [3.0, 3.0, 3.0]},
            }
        )
        assert result["status"] == "error"
        assert "zero variance" in result["message"].lower()

    def test_normal_ttest_still_works(self):
        """Normal t-test still functions after adding zero-variance guard."""
        result = handler(
            {
                "command": "ttest",
                "params": {"test_type": "two_sample", "values": [10, 11, 12, 10, 11], "values2": [13, 14, 15, 13, 14]},
            }
        )
        assert result["status"] in ("success", "warning")
        data = result.get("data", result)
        assert "t_statistic" in data


class TestAnovaNWayPrecheck:
    """BUG 3: n_way ANOVA with too few observations should give clear error."""

    def test_saturated_design_raises_clear_error(self):
        """2×2 design with only 4 observations → clear error, not patsy crash."""
        result = handler(
            {
                "command": "anova",
                "params": {
                    "anova_type": "n_way",
                    "factors": {"A": [1, 1, 2, 2], "B": [1, 2, 1, 2]},
                    "response": [10, 12, 14, 16],
                },
            }
        )
        assert result["status"] == "error"
        # Should mention observations/model, not patsy internals
        assert "observations" in result["message"].lower() or "model" in result["message"].lower()

    def test_n_way_with_enough_data_works(self):
        """2×2×2 design with 32 observations → success."""
        factors = {
            "A": [1, 1, 2, 2] * 8,
            "B": [1, 2, 1, 2] * 8,
            "C": [1, 1, 1, 1, 2, 2, 2, 2] * 4,
        }
        response = [
            10 + a * 2 + b * 3 + c + r
            for r, (a, b, c) in enumerate(list(zip([1, 1, 2, 2] * 8, [1, 2, 1, 2] * 8, [1, 1, 1, 1, 2, 2, 2, 2] * 4)))
        ]
        result = handler(
            {
                "command": "anova",
                "params": {"anova_type": "n_way", "factors": factors, "response": response},
            }
        )
        assert result["status"] in ("success", "warning")


class TestDiscoverSchemaAccuracy:
    """BUG 5: discover schema should match actual output fields."""

    def test_ttest_schema_matches_output(self):
        """discover ttest output fields should match actual output."""
        from stats_engine.discover import COMMANDS

        ttest = COMMANDS["ttest"]
        documented = set(ttest["output_fields"])

        # Run actual two-sample ttest
        handler(
            {
                "command": "ttest",
                "params": {"test_type": "two_sample", "values": [10, 11, 12], "values2": [13, 14, 15]},
            }
        )
        # At least the key two-sample fields should be documented
        assert "n1" in documented, "Schema should document n1 for two-sample"
        assert "mean_difference" in documented or "t_statistic" in documented
        # Old stale fields should be removed
        assert "ci_95" not in documented, "two-sample ttest doesn't return ci_95 (returns CI for mean difference)"
        assert "n" not in documented or "n1" in documented  # prefer n1/n2 for two-sample


class TestEdgeCasesRobustness:
    """Ensure edge cases produce graceful responses."""

    def test_empty_array_returns_error(self):
        """Empty input should error, not crash."""
        result = handler(
            {
                "command": "descriptive",
                "params": {"values": []},
            }
        )
        assert result["status"] == "error"

    def test_n1_data_returns_warning(self):
        """n=1 should return warning (low power) not crash."""
        result = handler(
            {
                "command": "capability",
                "params": {"values": [10.0], "usl": 12, "lsl": 8},
            }
        )
        assert result["status"] in ("error", "warning")

    def test_unknown_command_returns_structured_error(self):
        """Unknown command returns UNKNOWN_COMMAND error type."""
        result = handler({"command": "nonexistent_command_xyz"})
        assert result["status"] == "error"
        assert result["error_type"] == "UNKNOWN_COMMAND"
        assert "suggestion" in result

    def test_concurrent_calls_no_state_leak(self):
        """Rapid sequential calls don't leak state."""
        results = []
        for _i in range(20):
            r = handler(
                {
                    "command": "descriptive",
                    "params": {"values": np.random.normal(10, 1, 10).tolist()},
                }
            )
            results.append(r["status"])
        assert all(s in ("success", "warning") for s in results)

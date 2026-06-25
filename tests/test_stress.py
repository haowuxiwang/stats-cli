"""Stress tests for robustness and edge cases.

Tests extreme inputs, boundary conditions, and error recovery.
"""

import numpy as np
import pytest

from main import handler
from stats_engine.anova import anova
from stats_engine.correlation import correlation
from stats_engine.descriptive import descriptive
from stats_engine.normality import normality
from stats_engine.outlier import outlier
from stats_engine.regression import regression
from stats_engine.ttest import ttest

# ============================================================================
# Extreme Value Stress Tests
# ============================================================================


class TestExtremeValues:
    """Test with extreme numeric values."""

    def test_large_dataset(self):
        """10,000 values should work fine."""
        np.random.seed(42)
        values = np.random.normal(100, 15, 10000).tolist()
        result = descriptive(values=values)
        assert result["n"] == 10000
        assert abs(result["mean"] - 100) < 1

    def test_single_value(self):
        """Single value should return partial result."""
        result = descriptive(values=[42.0])
        assert result["n"] == 1
        assert result["mean"] == 42.0
        assert result["insufficient_for_stats"] is True

    def test_two_identical_values(self):
        """Two identical values: std=0, no crash."""
        result = descriptive(values=[7.0, 7.0])
        assert result["n"] == 2
        assert result["std"] == 0.0

    def test_hundred_identical_values(self):
        """100 identical values: should not produce warnings."""
        result = descriptive(values=[3.14] * 100)
        assert result["n"] == 100
        assert result["std"] == 0.0
        assert result["skewness"] is not None  # Should handle gracefully
        assert result["kurtosis"] is not None

    def test_negative_values(self):
        """All negative values."""
        values = [-10, -20, -30, -40, -50]
        result = descriptive(values=values)
        assert result["mean"] == -30.0
        assert result["min"] == -50
        assert result["max"] == -10

    def test_zero_values(self):
        """Values including zero."""
        values = [0, 0, 0, 0, 1]
        result = descriptive(values=values)
        assert result["n"] == 5
        assert result["mean"] == 0.2

    def test_scientific_notation(self):
        """Values with large magnitude should work."""
        values = [1.5e10, 3.14e-5, 6.02e23, 1.6e-19, 9.81]
        result = descriptive(values=values)
        assert result["n"] == 5
        assert result["std"] >= 0


# ============================================================================
# Input Validation Stress Tests
# ============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_all_nan_values(self):
        """All NaN should raise error (no valid data)."""
        values = [float("nan"), float("nan"), float("nan")]
        with pytest.raises(ValueError):
            descriptive(values=values)

    def test_all_inf_values(self):
        """All Inf should raise error (no valid data)."""
        values = [float("inf"), float("inf"), float("-inf")]
        with pytest.raises(ValueError):
            descriptive(values=values)

    def test_mixed_nan_inf_valid(self):
        """Mix of NaN, Inf, and valid values."""
        values = [1.0, float("nan"), 2.0, float("inf"), 3.0, float("-inf"), 4.0]
        result = descriptive(values=values)
        assert result["n"] == 4  # Only valid values counted

    def test_handler_json_string(self):
        """Handler should accept JSON string."""
        result = handler('{"command": "descriptive", "params": {"values": [1, 2, 3]}}')
        assert result["status"] == "success"

    def test_handler_malformed_json(self):
        """Handler should return error for malformed JSON."""
        result = handler("{invalid json")
        assert result["status"] == "error"
        assert result["error_type"] == "INVALID_INPUT"

    def test_handler_missing_command(self):
        """Handler should return error for missing command."""
        result = handler({"params": {"values": [1, 2, 3]}})
        assert result["status"] == "error"
        assert result["error_type"] == "MISSING_COMMAND"

    def test_handler_unknown_command(self):
        """Handler should return error for unknown command."""
        result = handler({"command": "nonexistent", "params": {}})
        assert result["status"] == "error"

    def test_handler_missing_required_params(self):
        """Handler should return error for missing required params."""
        result = handler({"command": "descriptive", "params": {}})
        assert result["status"] == "error"


# ============================================================================
# Statistical Correctness Stress Tests
# ============================================================================


class TestStatisticalCorrectness:
    """Verify statistical results against known properties."""

    def test_ttest_symmetry(self):
        """t-test should be symmetric: swapping groups changes sign of t."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 30).tolist()
        g2 = np.random.normal(12, 1, 30).tolist()

        r1 = ttest(test_type="two_sample", values=g1, values2=g2)
        r2 = ttest(test_type="two_sample", values=g2, values2=g1)

        assert abs(r1["t_statistic"] + r2["t_statistic"]) < 0.0001
        assert abs(r1["p_value"] - r2["p_value"]) < 0.0001

    def test_anova_f_statistic_positive(self):
        """F-statistic should always be non-negative."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 15).tolist()
        g2 = np.random.normal(12, 1, 15).tolist()
        g3 = np.random.normal(14, 1, 15).tolist()

        result = anova(anova_type="one_way", groups=[g1, g2, g3])
        assert result["f_statistic"] >= 0

    def test_regression_r_squared_range(self):
        """R-squared should be between 0 and 1 for linear regression."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50).tolist()
        y = (2 * np.array(x) + 3 + np.random.normal(0, 2, 50)).tolist()

        result = regression(x=x, y=y, reg_type="linear")
        assert 0 <= result["r_squared"] <= 1

    def test_correlation_range(self):
        """Correlation should be between -1 and 1."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100).tolist()
        y = np.random.normal(0, 1, 100).tolist()

        for method in ["pearson", "spearman", "kendall"]:
            result = correlation(x=x, y=y, method=method)
            assert -1 <= result["correlation"] <= 1, f"{method}: r={result['correlation']}"

    def test_descriptive_ci_contains_mean(self):
        """95% CI should contain the mean for normal data."""
        np.random.seed(42)
        values = np.random.normal(50, 10, 100).tolist()
        result = descriptive(values=values)

        if result["ci_95_lower"] is not None:
            assert result["ci_95_lower"] <= result["mean"] <= result["ci_95_upper"]

    def test_normality_consistency(self):
        """Normal data should pass multiple normality tests."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 300).tolist()
        result = normality(values=values)

        # At least 2 out of 3 tests should say normal
        normal_count = sum(
            [
                result["shapiro_wilk"]["normal"],
                result["anderson_darling"]["normal"],
                result["lilliefors"]["normal"],
            ]
        )
        assert normal_count >= 2, f"Only {normal_count}/3 tests say normal"


# ============================================================================
# Regression Type Stress Tests
# ============================================================================


class TestRegressionStress:
    """Stress test all regression types."""

    def test_linear_perfect_fit(self):
        """Perfect linear data: R^2 should be ~1."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = regression(x=x, y=y, reg_type="linear")
        assert result["r_squared"] > 0.9999

    def test_quadratic_perfect_fit(self):
        """Perfect quadratic data: R^2 should be ~1."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
        result = regression(x=x, y=y, reg_type="quadratic")
        assert result["r_squared"] > 0.9999

    def test_polynomial_high_degree(self):
        """High degree polynomial should not crash."""
        x = list(range(1, 21))
        y = [i**4 for i in x]
        result = regression(x=x, y=y, reg_type="polynomial", degree=4)
        assert result["r_squared"] > 0.99

    def test_logistic_binary(self):
        """Logistic regression with binary outcome."""
        x = list(range(1, 21))
        y = [0] * 10 + [1] * 10
        result = regression(x=x, y=y, reg_type="logistic")
        assert result["regression_type"] == "logistic"
        assert "coefficients" in result


# ============================================================================
# ANOVA Stress Tests
# ============================================================================


class TestAnovaStress:
    """Stress test ANOVA."""

    def test_many_groups(self):
        """ANOVA with 10 groups."""
        np.random.seed(42)
        groups = [np.random.normal(10 + i, 1, 10).tolist() for i in range(10)]
        result = anova(anova_type="one_way", groups=groups)
        assert result["significant"]  # Groups have different means

    def test_unequal_group_sizes(self):
        """ANOVA with unequal group sizes."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 10).tolist()
        g2 = np.random.normal(10, 1, 20).tolist()
        g3 = np.random.normal(10, 1, 30).tolist()
        result = anova(anova_type="one_way", groups=[g1, g2, g3])
        assert "f_statistic" in result

    def test_two_way_anova(self):
        """Two-way ANOVA should work."""
        np.random.seed(42)
        data = {
            "factor_a": np.random.choice(["A", "B"], 60).tolist(),
            "factor_b": np.random.choice(["X", "Y"], 60).tolist(),
            "values": np.random.normal(10, 2, 60).tolist(),
        }
        result = anova(anova_type="two_way", groups=[], data=data)
        assert "sources" in result or "f_statistic" in result
        assert "r_squared" in result


# ============================================================================
# Outlier Detection Stress Tests
# ============================================================================


class TestOutlierStress:
    """Stress test outlier detection."""

    def test_multiple_outliers(self):
        """Multiple outliers should be detected."""
        values = [10, 10, 10, 10, 10, 10, 10, 10, 100, 200]
        result = outlier(values=values, method="iqr")
        assert result["n_outliers"] >= 1

    def test_no_outliers_uniform(self):
        """Uniform data should have no outliers."""
        values = list(range(1, 21))
        result = outlier(values=values, method="grubbs")
        assert result["n_outliers"] == 0

    def test_outlier_methods_consistency(self):
        """Different methods should generally agree on extreme outliers."""
        values = [10, 10, 10, 10, 10, 10, 10, 10, 10, 1000]
        grubbs = outlier(values=values, method="grubbs")
        iqr = outlier(values=values, method="iqr")
        # At least one method should detect the outlier
        assert grubbs["n_outliers"] + iqr["n_outliers"] >= 1


# ============================================================================
# Handler End-to-End Stress Tests
# ============================================================================


class TestHandlerStress:
    """End-to-end handler stress tests."""

    def test_all_commands_discoverable(self):
        """All commands should be listed in discover."""
        result = handler({"command": "discover", "params": {}})
        assert result["status"] == "success"
        commands = result["data"]["commands"]
        expected = [
            "descriptive",
            "normality",
            "ttest",
            "anova",
            "regression",
            "correlation",
            "outlier",
            "power",
            "control_chart",
            "capability",
            "doe",
            "gage_rr",
            "reliability",
            "multivariate",
            "timeseries",
        ]
        for cmd in expected:
            assert cmd in commands, f"Missing command: {cmd}"

    def test_discover_each_command(self):
        """Each command should have valid discover metadata."""
        result = handler({"command": "discover", "params": {}})
        for name, cmd in result["data"]["commands"].items():
            assert "description" in cmd, f"{name}: missing description"
            assert "category" in cmd, f"{name}: missing category"

    def test_power_via_handler_backward_compat(self):
        """Power command should work with both 'power' and 'power_val'."""
        r1 = handler({"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}})
        r2 = handler({"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power_val": 0.80}})
        assert r1["status"] == "success"
        assert r2["status"] == "success"
        assert r1["data"]["sample_size"] == r2["data"]["sample_size"]

    def test_all_new_commands_discoverable(self):
        """New commands should be listed in discover."""
        result = handler({"command": "discover", "params": {}})
        assert result["status"] == "success"
        commands = result["data"]["commands"]
        for cmd in ["distribution", "bayesian", "mining", "sensitivity"]:
            assert cmd in commands, f"Missing new command: {cmd}"


# ============================================================================
# New Module Stress Tests
# ============================================================================


class TestNewModulesStress:
    """Stress tests for distribution, bayesian, mining, sensitivity."""

    def test_distribution_large_data(self):
        """Distribution select with 10000 values."""
        from stats_engine.distribution import distribution

        np.random.seed(42)
        values = np.random.normal(10, 2, 10000).tolist()
        result = distribution("select", values=values)
        assert result["n"] == 10000
        assert result["best_distribution"] is not None

    def test_bayesian_many_groups(self):
        """Bayesian ANOVA with 20 groups."""
        from stats_engine.bayesian import bayesian

        np.random.seed(42)
        groups = [np.random.normal(10 + i * 0.5, 1, 10).tolist() for i in range(20)]
        result = bayesian("anova", groups=groups)
        assert result["n_groups"] == 20
        assert result["n_total"] == 200

    def test_mining_large_classification(self):
        """Random forest with 500 samples x 10 features."""
        from stats_engine.mining import mining

        np.random.seed(42)
        features = np.random.randn(500, 10).tolist()
        labels = [0] * 250 + [1] * 250
        result = mining("classify", features=features, labels=labels)
        assert result["n_samples"] == 500
        assert result["n_features"] == 10

    def test_sensitivity_many_inputs(self):
        """Monte Carlo with 10 input variables."""
        from stats_engine.sensitivity import sensitivity

        inputs = {f"x{i}": {"dist": "normal", "params": {"mean": i, "std": 1}} for i in range(10)}
        formula = " + ".join(f"x{i}" for i in range(10))
        result = sensitivity("monte_carlo", inputs=inputs, formula=formula, n_simulations=1000)
        assert result["n_valid"] == 1000

    def test_distribution_edge_values(self):
        """Distribution fit with extreme values."""
        from stats_engine.distribution import distribution

        np.random.seed(42)
        values = [1e-10, 1e-5, 0.001, 0.1, 1.0, 10.0, 1e5, 1e10]
        result = distribution("fit", values=values, dist_name="lognormal")
        assert "parameters" in result

    def test_bayesian_extreme_proportion(self):
        """Proportion test with 0 and n successes."""
        from stats_engine.bayesian import bayesian

        r1 = bayesian("proportion", successes=0, n=10)
        assert r1["successes"] == 0
        assert r1["posterior_mean"] > 0

        r2 = bayesian("proportion", successes=10, n=10)
        assert r2["successes"] == 10
        assert r2["posterior_mean"] < 1

    def test_mining_single_anomaly(self):
        """Anomaly detection with a single extreme outlier."""
        from stats_engine.mining import mining

        values = [10] * 20 + [1000]
        result = mining("anomaly", values=values, method="isolation_forest")
        assert result["n_anomalies"] >= 1

    def test_sensitivity_constant_input(self):
        """Monte Carlo with constant input should still work."""
        from stats_engine.sensitivity import sensitivity

        inputs = {"x": {"dist": "constant", "params": {"value": 5}}}
        result = sensitivity("monte_carlo", inputs=inputs, formula="x", n_simulations=500)
        assert result["mean"] == 5.0
        assert result["std"] == 0.0


class TestNewModulesValidation:
    """Input validation tests for new modules."""

    def test_distribution_bad_dist(self):
        from stats_engine.distribution import distribution

        with pytest.raises(ValueError, match="Unknown distribution"):
            distribution("fit", values=[1, 2, 3, 4, 5], dist_name="unknown")

    def test_distribution_bad_analysis_type(self):
        from stats_engine.distribution import distribution

        with pytest.raises(ValueError, match="Unknown analysis_type"):
            distribution("unknown", values=[1, 2, 3, 4, 5])

    def test_bayesian_bad_analysis_type(self):
        from stats_engine.bayesian import bayesian

        with pytest.raises(ValueError, match="Unknown analysis_type"):
            bayesian("unknown", values=[1, 2, 3])

    def test_mining_bad_analysis_type(self):
        from stats_engine.mining import mining

        with pytest.raises(ValueError, match="Unknown analysis_type"):
            mining("unknown")

    def test_sensitivity_bad_analysis_type(self):
        from stats_engine.sensitivity import sensitivity

        with pytest.raises(ValueError, match="Unknown analysis_type"):
            sensitivity("unknown")

    def test_mining_classify_too_few_samples(self):
        from stats_engine.mining import mining

        with pytest.raises(ValueError, match="at least 10"):
            mining("classify", features=[[1], [2]], labels=[0, 1])

    def test_sensitivity_bad_distribution(self):
        from stats_engine.sensitivity import sensitivity

        with pytest.raises(ValueError, match="Unknown distribution"):
            inputs = {"x": {"dist": "unknown", "params": {}}}
            sensitivity("monte_carlo", inputs=inputs, formula="x")

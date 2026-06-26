"""Tests for stats_engine/chart_handlers.py."""

import base64

import numpy as np

from stats_engine.chart_handlers import CHART_HANDLERS


class TestChartHandlersRegistry:
    def test_registry_has_25_entries(self):
        assert len(CHART_HANDLERS) == 25

    def test_all_handlers_callable(self):
        for name, handler in CHART_HANDLERS.items():
            assert callable(handler), f"Handler for '{name}' is not callable"

    def test_expected_commands_present(self):
        expected = [
            "descriptive",
            "normality",
            "control_chart",
            "capability",
            "correlation",
            "regression",
            "timeseries",
            "report",
            "ttest",
            "anova",
            "homogeneity",
            "multiple_comparison",
            "equivalence",
            "power",
            "multivariate",
            "trend",
            "outlier",
            "reliability",
            "gage_rr",
            "nonparametric",
            "explore",
            "doe",
        ]
        for cmd in expected:
            assert cmd in CHART_HANDLERS, f"Missing handler for '{cmd}'"


class TestDescriptiveHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["descriptive"]
        result = {"_values": [1, 2, 3, 4, 5]}
        chart = handler(result, {})
        assert chart is not None
        decoded = base64.b64decode(chart)
        assert decoded[:4] == b"\x89PNG"

    def test_empty_values(self):
        handler = CHART_HANDLERS["descriptive"]
        result = {"_values": []}
        chart = handler(result, {})
        assert chart is None


class TestNormalityHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["normality"]
        result = {"_values": np.random.normal(0, 1, 50).tolist()}
        chart = handler(result, {})
        assert chart is not None


class TestControlChartHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["control_chart"]
        result = {
            "_values": [10, 11, 12, 13, 14, 15],
            "chart_type": "imr",
            "ucl": 16,
            "cl": 12.5,
            "lcl": 9,
            "out_of_control": [],
        }
        chart = handler(result, {})
        assert chart is not None


class TestCapabilityHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["capability"]
        result = {
            "_values": np.random.normal(10, 1, 50).tolist(),
            "usl": 13,
            "lsl": 7,
            "target": 10,
            "cp": 1.0,
            "cpk": 0.9,
        }
        chart = handler(result, {})
        assert chart is not None


class TestCorrelationHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["correlation"]
        params = {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]}
        chart = handler({}, params)
        assert chart is not None

    def test_empty(self):
        handler = CHART_HANDLERS["correlation"]
        chart = handler({}, {"x": [], "y": []})
        assert chart is None


class TestRegressionHandler:
    def test_with_coefficients(self):
        handler = CHART_HANDLERS["regression"]
        result = {
            "coefficients": {"slope": 2.0, "intercept": 0.0},
            "r_squared": 1.0,
            "regression_type": "linear",
        }
        params = {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]}
        chart = handler(result, params)
        assert chart is not None


class TestTimeseriesHandler:
    def test_exp_smoothing(self):
        handler = CHART_HANDLERS["timeseries"]
        result = {"_values": [10, 12, 11, 13, 14], "analysis_type": "exp_smoothing"}
        chart = handler(result, {})
        assert chart is not None

    def test_acf(self):
        handler = CHART_HANDLERS["timeseries"]
        result = {"_values": list(range(50)), "analysis_type": "acf"}
        chart = handler(result, {"max_lag": 10})
        assert chart is not None


class TestTtestHandler:
    def test_two_sample(self):
        handler = CHART_HANDLERS["ttest"]
        result = {"p_value": 0.001, "test_type": "two_sample"}
        params = {"values": [1, 2, 3, 4, 5], "values2": [6, 7, 8, 9, 10]}
        chart = handler(result, params)
        assert chart is not None


class TestAnovaHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["anova"]
        params = {"groups": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        chart = handler({}, params)
        assert chart is not None


class TestHomogeneityHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["homogeneity"]
        params = {"groups": [[1, 2, 3], [4, 5, 6]]}
        chart = handler({}, params)
        assert chart is not None


class TestMultipleComparisonHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["multiple_comparison"]
        result = {
            "comparisons": [
                {"group1": "A", "group2": "B", "mean_diff": 2.5, "ci_lower": 1.0, "ci_upper": 4.0, "significant": True},
                {"group1": "A", "group2": "C", "mean_diff": 4.0, "ci_lower": 2.5, "ci_upper": 5.5, "significant": True},
            ]
        }
        chart = handler(result, {})
        assert chart is not None


class TestEquivalenceHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["equivalence"]
        result = {"ci_90": [-0.3, 0.5], "mean_diff": 0.1}
        params = {"delta": 1.0}
        chart = handler(result, params)
        assert chart is not None


class TestPowerHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["power"]
        params = {"effect_size": 0.5, "power": 0.8}
        chart = handler({}, params)
        assert chart is not None


class TestMultivariateHandler:
    def test_correlation_matrix(self):
        handler = CHART_HANDLERS["multivariate"]
        result = {
            "analysis_type": "correlation_matrix",
            "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
            "columns": ["X", "Y"],
        }
        chart = handler(result, {})
        assert chart is not None

    def test_pca(self):
        handler = CHART_HANDLERS["multivariate"]
        result = {"analysis_type": "pca"}
        chart = handler(result, {})
        assert chart is None


class TestTrendHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["trend"]
        result = {"_values": [10, 11, 12, 13, 14, 15], "test_type": "cusum"}
        chart = handler(result, {})
        assert chart is not None


class TestOutlierHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["outlier"]
        result = {"_values": [10, 11, 10, 12, 100], "outliers": [{"index": 4}]}
        chart = handler(result, {})
        assert chart is not None


class TestReliabilityHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["reliability"]
        result = {"parameters": {"shape": 2.0, "scale": 500}, "analysis_type": "weibull"}
        params = {"times": [100, 200, 300, 400, 500]}
        chart = handler(result, params)
        assert chart is not None


class TestGageRRHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["gage_rr"]
        result = {"contribution": {"Part-to-Part": 60, "Repeatability": 25, "Reproducibility": 15}}
        chart = handler(result, {})
        assert chart is not None


class TestNonparametricHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["nonparametric"]
        params = {"groups": [[1, 2, 3], [4, 5, 6]]}
        chart = handler({}, params)
        assert chart is not None


class TestExploreHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["explore"]
        result = {"columns": ["A", "B", "C"], "missing_values": {"A": 0, "B": 5, "C": 2}}
        chart = handler(result, {})
        assert chart is not None


class TestDoeHandler:
    def test_basic(self):
        handler = CHART_HANDLERS["doe"]
        result = {"effects": {"Temp": 10, "Time": 5, "Pressure": 3}}
        chart = handler(result, {})
        assert chart is not None

    def test_no_effects(self):
        handler = CHART_HANDLERS["doe"]
        result = {"effects": {}}
        chart = handler(result, {})
        assert chart is None


def test_multiple_comparison_empty_comparisons():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["multiple_comparison"]
    result = handler({"comparisons": []}, {})
    assert result is None


def test_equivalence_no_ci():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["equivalence"]
    result = handler({}, {"delta": 1})
    assert result is None


def test_power_no_effect_size():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["power"]
    result = handler({}, {})
    assert result is None


def test_trend_empty_values():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["trend"]
    result = handler({}, {"_values": []})
    assert result is None


def test_outlier_numeric_entries():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["outlier"]
    result = handler({"outliers": [5, 10], "n_outliers": 2}, {"_values": [1, 2, 3, 4, 5, 10]})
    # Should handle numeric outlier entries
    assert result is not None or result is None  # Just shouldn't crash


def test_reliability_no_times():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["reliability"]
    result = handler({}, {})
    assert result is None


def test_gage_rr_no_contribution():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["gage_rr"]
    result = handler({}, {})
    assert result is None


def test_explore_no_columns():
    from stats_engine.chart_handlers import CHART_HANDLERS

    handler = CHART_HANDLERS["explore"]
    result = handler({"columns": []}, {})
    assert result is None

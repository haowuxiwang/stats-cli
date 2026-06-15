"""Tests for stats_engine/charts.py."""
import base64

import numpy as np

from stats_engine.charts import (
    acf_plot,
    boxplot,
    capability_plot,
    control_chart_plot,
    heatmap,
    histogram,
    means_comparison_plot,
    missing_values_plot,
    outlier_plot,
    pareto_plot,
    power_curve,
    qq_plot,
    residual_plot,
    scatter_with_regression,
    time_series_plot,
    tost_plot,
    ttest_plot,
    variance_component_plot,
    weibull_plot,
)


class TestHistogram:
    def test_basic(self):
        result = histogram([1, 2, 3, 4, 5], title="Test")
        assert isinstance(result, str)
        # Verify valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        # Should be PNG
        assert decoded[:4] == b'\x89PNG'

    def test_with_normal_curve(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 200).tolist()
        result = histogram(values, title="Normal", normal_curve=True)
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_without_normal_curve(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = histogram(values, title="No Curve", normal_curve=False)
        assert isinstance(result, str)

    def test_custom_bins(self):
        values = list(range(100))
        result = histogram(values, title="Custom Bins", bins=10)
        assert isinstance(result, str)


class TestControlChartPlot:
    def test_imr_basic(self):
        values = [10, 12, 11, 13, 10, 14, 12, 11, 13, 10]
        result = control_chart_plot(values, chart_type="imr", ucl=15, cl=11.5, lcl=8)
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_with_out_of_control(self):
        values = [10, 12, 11, 13, 10, 20, 12, 11, 13, 10]
        result = control_chart_plot(values, chart_type="imr", out_of_control=[5])
        assert isinstance(result, str)

    def test_no_limits(self):
        values = [10, 12, 11, 13, 10, 14, 12, 11, 13, 10]
        result = control_chart_plot(values, chart_type="xbar")
        assert isinstance(result, str)


class TestBoxplot:
    def test_single_group(self):
        result = boxplot([[1, 2, 3, 4, 5]], labels=["Group A"])
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_multiple_groups(self):
        groups = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [3, 4, 5, 6, 7]]
        result = boxplot(groups, labels=["A", "B", "C"])
        assert isinstance(result, str)


class TestScatterWithRegression:
    def test_basic_no_line(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = scatter_with_regression(x, y, title="Test")
        assert isinstance(result, str)

    def test_with_regression_line(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = scatter_with_regression(x, y, title="With Line", slope=2.0, intercept=0.0, r_squared=1.0)
        assert isinstance(result, str)
        base64.b64decode(result)


class TestQQPlot:
    def test_normal_data(self):
        np.random.seed(42)
        values = np.random.normal(0, 1, 100).tolist()
        result = qq_plot(values, title="Q-Q Normal")
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_small_sample(self):
        values = [1, 2, 3, 4, 5]
        result = qq_plot(values, title="Q-Q Small")
        assert isinstance(result, str)


class TestCapabilityPlot:
    def test_basic(self):
        np.random.seed(42)
        values = np.random.normal(10, 1, 100).tolist()
        result = capability_plot(values, usl=13, lsl=7, cp=1.0, cpk=0.9)
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_with_target(self):
        np.random.seed(42)
        values = np.random.normal(10, 0.5, 100).tolist()
        result = capability_plot(values, usl=12, lsl=8, target=10, cp=1.33, cpk=1.2)
        assert isinstance(result, str)


class TestTimeSeriesPlot:
    def test_basic(self):
        values = list(range(1, 51))
        result = time_series_plot(values, title="Time Series")
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_with_fitted(self):
        values = list(range(1, 21))
        fitted = [v * 0.9 + 0.5 for v in values]
        result = time_series_plot(values, title="With Fitted", fitted=fitted)
        assert isinstance(result, str)

    def test_with_forecast(self):
        values = list(range(1, 21))
        forecast = [21, 22, 23, 24, 25]
        result = time_series_plot(values, title="With Forecast", forecast=forecast)
        assert isinstance(result, str)


class TestHeatmap:
    def test_basic(self):
        matrix = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]]
        result = heatmap(matrix, labels=["X", "Y", "Z"])
        assert isinstance(result, str)
        base64.b64decode(result)

    def test_no_labels(self):
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        result = heatmap(matrix)
        assert isinstance(result, str)


class TestACFPlot:
    def test_basic(self):
        np.random.seed(42)
        values = np.cumsum(np.random.normal(0, 1, 100)).tolist()
        result = acf_plot(values, max_lag=20, title="ACF Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_small_sample(self):
        values = [1, 2, 3]
        result = acf_plot(values, max_lag=1)
        assert isinstance(result, str)

    def test_constant_values(self):
        values = [5.0] * 20
        result = acf_plot(values, max_lag=10)
        assert result is None  # Constant values have zero variance


class TestParetoPlot:
    def test_basic(self):
        names = ["A", "B", "C", "D"]
        effects = [10, 5, 3, 1]
        result = pareto_plot(effects, names, title="Pareto Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_negative_effects(self):
        names = ["A", "B", "C"]
        effects = [10, -5, 3]
        result = pareto_plot(effects, names)
        assert isinstance(result, str)


class TestOutlierPlot:
    def test_basic(self):
        values = [10, 11, 10, 12, 100, 11, 10, 12, 11, 10]
        result = outlier_plot(values, outlier_indices=[4], title="Outlier Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_no_outliers(self):
        values = [10, 11, 10, 12, 11, 10, 12, 11, 10, 11]
        result = outlier_plot(values, outlier_indices=[])
        assert isinstance(result, str)


class TestResidualPlot:
    def test_basic(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2.1, 4.0, 5.8, 8.1, 10.2, 11.9, 14.0, 16.1, 17.9, 20.0]
        result = residual_plot(x, y, slope=2.0, intercept=0.0, title="Residual Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_no_slope(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = residual_plot(x, y, title="No Slope")
        assert isinstance(result, str)


class TestWeibullPlot:
    def test_basic(self):
        times = [100, 200, 300, 400, 500, 600, 700, 800]
        result = weibull_plot(times, shape=2.0, scale=500, title="Weibull Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_no_params(self):
        times = [100, 200, 300, 400, 500]
        result = weibull_plot(times, title="No Params")
        assert isinstance(result, str)

    def test_too_few_values(self):
        times = [100, 200]
        result = weibull_plot(times)
        assert result is None


class TestVarianceComponentPlot:
    def test_basic(self):
        components = {"Part-to-Part": 60.5, "Repeatability": 25.3, "Reproducibility": 14.2}
        result = variance_component_plot(components, title="Variance Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_empty(self):
        result = variance_component_plot({})
        assert result is None


class TestTtestPlot:
    def test_two_groups(self):
        g1 = [10.2, 10.5, 10.3, 10.1, 10.4]
        g2 = [11.3, 11.5, 11.1, 11.4, 11.2]
        result = ttest_plot(g1, group2=g2, p_value=0.001, title="t-test Plot")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_one_group(self):
        g1 = [10.2, 10.5, 10.3, 10.1, 10.4]
        result = ttest_plot(g1, p_value=0.05, title="One Group")
        assert isinstance(result, str)


class TestMissingValuesPlot:
    def test_basic(self):
        columns = ["A", "B", "C", "D"]
        counts = [0, 5, 2, 0]
        result = missing_values_plot(columns, counts, title="Missing Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'


class TestMeansComparisonPlot:
    def test_basic(self):
        names = ["A vs B", "A vs C", "B vs C"]
        means = [2.5, 4.0, 1.5]
        ci_lowers = [1.0, 2.5, 0.0]
        ci_uppers = [4.0, 5.5, 3.0]
        result = means_comparison_plot(names, means, ci_lowers, ci_uppers, title="Means Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'


class TestTOSTPlot:
    def test_equivalent(self):
        result = tost_plot(0.1, -0.3, 0.5, delta=1.0, title="TOST Equivalent")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_not_equivalent(self):
        result = tost_plot(2.0, 1.0, 3.0, delta=0.5, title="TOST Not Equivalent")
        assert isinstance(result, str)


class TestPowerCurve:
    def test_basic(self):
        n_values = list(range(10, 101, 10))
        result = power_curve(0.5, n_values, target_power=0.8, title="Power Test")
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

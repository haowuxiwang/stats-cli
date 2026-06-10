"""Tests for stats_engine/charts.py."""
import base64

import numpy as np

from stats_engine.charts import (
    boxplot,
    capability_plot,
    control_chart_plot,
    heatmap,
    histogram,
    qq_plot,
    scatter_with_regression,
    time_series_plot,
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

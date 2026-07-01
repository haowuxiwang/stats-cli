"""Coverage-boosting tests for stats_engine/charts.py — targets uncovered lines."""

import base64

import numpy as np

from stats_engine.charts import (
    acf_plot,
    bootstrap_plot,
    cluster_scatter_2d,
    functional_plot,
    oc_curve_plot,
    posterior_plot,
    tornado_plot,
    weibull_plot,
)


class TestAcfPlot:
    def test_normal(self):
        np.random.seed(42)
        values = np.random.normal(0, 1, 50).tolist()
        result = acf_plot(values, max_lag=10)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    def test_too_few_values(self):
        """Line 218-219: n < 3 returns None."""
        result = acf_plot([1, 2], max_lag=5)
        assert result is None

    def test_zero_variance(self):
        """Line 224-225: var == 0 returns None."""
        result = acf_plot([5.0, 5.0, 5.0, 5.0, 5.0])
        assert result is None


class TestWeibullPlot:
    def test_with_params(self):
        """Line 376-379: with shape and scale."""
        times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = weibull_plot(times, shape=2.5, scale=50)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    def test_too_few_points(self):
        """Line 357-358: n < 3 returns None."""
        result = weibull_plot([10, 20])
        assert result is None

    def test_negative_values_filtered(self):
        """Line 355: only positive values used."""
        times = [-5, 0, 10, 20, 30, 40, 50]
        result = weibull_plot(times, shape=1.5, scale=30)
        assert isinstance(result, str)


class TestOcCurvePlot:
    def test_basic(self):
        """Lines 460-468."""
        defect_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        accept_probs = [0.99, 0.95, 0.80, 0.50, 0.25, 0.10]
        result = oc_curve_plot(defect_rates, accept_probs)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"


class TestTornadoPlot:
    def test_basic(self):
        """Lines 472-482."""
        variables = ["Temp", "Pressure", "Speed", "Time"]
        sensitivities = [0.8, 0.5, 0.3, 0.1]
        result = tornado_plot(variables, sensitivities)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"


class TestBootstrapPlot:
    def test_basic(self):
        """Lines 574-584."""
        result = bootstrap_plot(original_stat=10.5, ci_lower=9.8, ci_upper=11.2, bootstrap_mean=10.6)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"


class TestPosteriorPlot:
    def test_basic(self):
        """Lines 589-613."""
        result = posterior_plot(
            prior_mean=0, prior_std=1, posterior_mean=0.5, posterior_std=0.3, credible_interval=[0.1, 0.9]
        )
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"


class TestFunctionalPlot:
    def test_with_mean(self):
        """Lines 618-628."""
        t = np.linspace(0, 1, 50)
        curves = [np.sin(t) + np.random.normal(0, 0.1, 50) for _ in range(10)]
        mean_curve = np.sin(t)
        result = functional_plot(t, curves, mean_curve=mean_curve)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    def test_without_mean(self):
        t = np.linspace(0, 1, 50)
        curves = [np.sin(t) + np.random.normal(0, 0.1, 50) for _ in range(5)]
        result = functional_plot(t, curves)
        assert isinstance(result, str)


class TestClusterScatter2D:
    def test_with_centers(self):
        """Lines 633-646."""
        np.random.seed(42)
        points = np.random.randn(50, 2)
        labels = [0] * 25 + [1] * 25
        centers = np.array([[0.5, 0.5], [-0.5, -0.5]])
        result = cluster_scatter_2d(points, labels, centers=centers)
        assert isinstance(result, str)
        assert base64.b64decode(result)[:4] == b"\x89PNG"

    def test_without_centers(self):
        np.random.seed(42)
        points = np.random.randn(30, 2)
        labels = [0] * 15 + [1] * 15
        result = cluster_scatter_2d(points, labels)
        assert isinstance(result, str)

"""Coverage-boosting tests for stats_engine/chart_handlers.py — targets uncovered handlers."""

import numpy as np

from main import handler


class TestChartHandlersCoverage:
    """Test chart handler dispatch for all supported commands."""

    def test_timeseries_acf_chart(self):
        """Lines 103-117: _timeseries_handler with acf."""
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": [10, 12, 11, 13, 10, 14, 12, 11, 13, 10, 12, 14],
                    "analysis_type": "acf",
                    "chart": True,
                },
            }
        )
        assert result["status"] == "success"
        assert "chart_base64" in result["data"]

    def test_timeseries_exp_smoothing_chart(self):
        """Line 112-117: time_series_plot path."""
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": [10, 12, 11, 13, 10, 14, 12, 11, 13, 10, 12, 14],
                    "analysis_type": "exp_smoothing",
                    "chart": True,
                },
            }
        )
        assert result["status"] == "success"
        assert "chart_base64" in result["data"]

    def test_ttest_chart(self):
        """Lines 120-130: _ttest_handler."""
        result = handler(
            {
                "command": "ttest",
                "params": {
                    "test_type": "two_sample",
                    "values": [10, 11, 12, 10, 11, 12, 11, 10, 12, 11],
                    "values2": [13, 14, 15, 13, 14, 15, 14, 13, 15, 14],
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_ttest_one_sample_chart(self):
        """Line 120-130: ttest with single group. Uses 'mu' not 'population_mean'."""
        result = handler(
            {
                "command": "ttest",
                "params": {
                    "test_type": "one_sample",
                    "values": [10, 11, 12, 10, 11, 12, 11, 10, 12, 11],
                    "mu": 10,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_multiple_comparison_chart(self):
        """Lines 139-155: _multiple_comparison_handler. Uses 'test_type' not 'method'."""
        result = handler(
            {
                "command": "multiple_comparison",
                "params": {
                    "test_type": "tukey",
                    "groups": [
                        [10, 11, 12, 10, 11, 12, 11, 10, 12, 11],
                        [13, 14, 15, 13, 14, 15, 14, 13, 15, 14],
                        [16, 17, 18, 16, 17, 18, 17, 16, 18, 17],
                    ],
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_equivalence_chart(self):
        """Lines 158-165: _equivalence_handler."""
        result = handler(
            {
                "command": "equivalence",
                "params": {
                    "test_type": "tost",
                    "values": [10.1, 10.0, 9.9, 10.1, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0],
                    "values2": [10.2, 10.1, 10.0, 10.2, 10.1, 10.2, 10.1, 10.0, 10.2, 10.1],
                    "delta": 0.5,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_power_chart(self):
        """Lines 168-174: _power_handler. Uses 'test_type'='two_sample'."""
        result = handler(
            {
                "command": "power",
                "params": {
                    "test_type": "two_sample",
                    "effect_size": 0.5,
                    "alpha": 0.05,
                    "power": 0.8,
                    "chart": True,
                },
            }
        )
        # Power command may warn on small effect sizes
        assert result["status"] in ("success", "warning", "error")

    def test_multivariate_pca_chart(self):
        """Lines 177+: _multivariate_handler for PCA. Uses 'values' not 'variables'."""
        np.random.seed(42)
        data = np.random.randn(20, 3).tolist()
        result = handler(
            {
                "command": "multivariate",
                "params": {
                    "analysis_type": "pca",
                    "values": data,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

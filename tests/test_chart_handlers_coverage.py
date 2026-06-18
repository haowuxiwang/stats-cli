"""Additional tests for stats_engine/chart_handlers.py to improve coverage."""

import pytest

from main import handler


class TestChartHandlers:
    """Test chart handler edge cases."""

    def test_descriptive_chart(self):
        """Descriptive command generates chart."""
        result = handler({
            "command": "descriptive",
            "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "chart": True,
        })
        assert result["status"] == "success"

    def test_anova_chart(self):
        """ANOVA command generates chart."""
        result = handler({
            "command": "anova",
            "params": {
                "groups": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_capability_chart(self):
        """Capability command generates chart."""
        result = handler({
            "command": "capability",
            "params": {
                "values": [9.8, 10.0, 10.2, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2] * 3,
                "usl": 12,
                "lsl": 8,
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_correlation_chart(self):
        """Correlation command generates chart."""
        result = handler({
            "command": "correlation",
            "params": {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 4, 5, 4, 5, 7, 8, 9, 10, 12],
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_regression_chart(self):
        """Regression command generates chart."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2, 4, 5, 4, 5, 7, 8, 9, 10, 12],
                "reg_type": "linear",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_outlier_chart(self):
        """Outlier command generates chart."""
        result = handler({
            "command": "outlier",
            "params": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
                "method": "grubbs",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_normality_chart(self):
        """Normality command generates chart."""
        result = handler({
            "command": "normality",
            "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "chart": True,
        })
        assert result["status"] == "success"

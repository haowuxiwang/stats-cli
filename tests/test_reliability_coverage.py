"""Additional tests for stats_engine/reliability.py to improve coverage."""

import numpy as np

from main import handler


class TestReliabilityEdgeCases:
    """Test reliability edge cases."""

    def test_weibull_basic(self):
        """Basic Weibull analysis."""
        times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = handler({
            "command": "reliability",
            "params": {
                "times": times,
                "analysis_type": "weibull",
            },
        })
        assert result["status"] == "success"
        assert "beta" in result["data"]
        assert "eta" in result["data"]

    def test_kaplan_meier_basic(self):
        """Basic Kaplan-Meier analysis."""
        times = [10, 20, 30, 40, 50]
        result = handler({
            "command": "reliability",
            "params": {
                "times": times,
                "analysis_type": "kaplan_meier",
            },
        })
        assert result["status"] == "success"

    def test_kaplan_meier_with_status(self):
        """Kaplan-Meier with censored data."""
        times = [10, 20, 30, 40, 50]
        status = [1, 1, 0, 1, 0]  # 0 = censored
        result = handler({
            "command": "reliability",
            "params": {
                "times": times,
                "status": status,
                "analysis_type": "kaplan_meier",
            },
        })
        assert result["status"] == "success"

    def test_distribution_fit_basic(self):
        """Distribution fitting."""
        np.random.seed(42)
        times = np.random.weibull(2, 50) * 100
        result = handler({
            "command": "reliability",
            "params": {
                "times": times.tolist(),
                "analysis_type": "distribution",
            },
        })
        assert result["status"] == "success"

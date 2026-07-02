"""Coverage-boosting tests for chart_handlers.py — targets uncovered handler dispatchers."""

import numpy as np

from main import handler


class TestChartHandlersCoverage:
    """Test chart handler dispatch for commands with uncovered handlers."""

    def test_doe_pareto_chart(self):
        """Lines 244-251: _doe_handler generates Pareto chart."""
        result = handler(
            {
                "command": "doe",
                "params": {
                    "doe_type": "full_factorial",
                    "factors": [{"name": "A", "levels": [1, 2]}, {"name": "B", "levels": [1, 2]}],
                    "responses": [10, 14, 12, 18],
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")
        assert "chart_base64" in result.get("data", {})

    def test_distribution_histogram_chart(self):
        """Lines 254-261: _distribution_handler generates histogram."""
        result = handler(
            {
                "command": "distribution",
                "params": {
                    "analysis_type": "fit",
                    "values": [10.1, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2] * 5,
                    "distribution": "normal",
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_acceptance_sampling_oc_chart(self):
        """Lines 264-271: _acceptance_sampling_handler generates OC curve."""
        result = handler(
            {
                "command": "acceptance_sampling",
                "params": {
                    "analysis_type": "oc_curve",
                    "n": 50,
                    "c": 2,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_sensitivity_monte_carlo_chart(self):
        """Lines 274-283: _sensitivity_handler generates tornado/dot chart."""
        result = handler(
            {
                "command": "sensitivity",
                "params": {
                    "analysis_type": "monte_carlo",
                    "inputs": {
                        "T": {"dist": "normal", "params": {"mean": 100, "std": 10}},
                        "S": {"dist": "normal", "params": {"mean": 50, "std": 5}},
                    },
                    "formula": "T + S",
                    "n_simulations": 100,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_advanced_bootstrap_chart(self):
        """Lines 286-295: _advanced_handler generates bootstrap plot."""
        result = handler(
            {
                "command": "advanced",
                "params": {
                    "analysis_type": "bootstrap",
                    "values": [10.1, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.3] * 4,
                    "statistic": "mean",
                    "n_bootstrap": 100,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_bayesian_estimate_chart(self):
        """Lines 298-308: _bayesian_handler generates posterior plot."""
        result = handler(
            {
                "command": "bayesian",
                "params": {
                    "analysis_type": "estimate",
                    "values": [10.1, 10.2, 10.0, 10.3, 10.1],
                    "prior_mean": 10,
                    "prior_std": 1,
                    "credible_level": 0.95,
                    "chart": True,
                },
            }
        )
        assert result["status"] in ("success", "warning")

    def test_multivariate_pca_chart(self):
        """Line 342+: _multivariate_handler for PCA scatter."""
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

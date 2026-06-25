"""Additional tests for stats_engine/reliability.py to improve coverage."""

from unittest.mock import MagicMock, patch

import numpy as np

from main import handler


class TestReliabilityEdgeCases:
    """Test reliability edge cases."""

    def test_weibull_basic(self):
        """Basic Weibull analysis."""
        times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = handler(
            {
                "command": "reliability",
                "params": {
                    "times": times,
                    "analysis_type": "weibull",
                },
            }
        )
        assert result["status"] == "success"
        assert "beta" in result["data"]
        assert "eta" in result["data"]

    def test_kaplan_meier_basic(self):
        """Basic Kaplan-Meier analysis."""
        times = [10, 20, 30, 40, 50]
        result = handler(
            {
                "command": "reliability",
                "params": {
                    "times": times,
                    "analysis_type": "kaplan_meier",
                },
            }
        )
        assert result["status"] == "success"

    def test_kaplan_meier_with_status(self):
        """Kaplan-Meier with censored data."""
        times = [10, 20, 30, 40, 50]
        status = [1, 1, 0, 1, 0]  # 0 = censored
        result = handler(
            {
                "command": "reliability",
                "params": {
                    "times": times,
                    "status": status,
                    "analysis_type": "kaplan_meier",
                },
            }
        )
        assert result["status"] == "success"

    def test_distribution_fit_basic(self):
        """Distribution fitting."""
        np.random.seed(42)
        times = np.random.weibull(2, 50) * 100
        result = handler(
            {
                "command": "reliability",
                "params": {
                    "times": times.tolist(),
                    "analysis_type": "distribution",
                },
            }
        )
        assert result["status"] == "success"


class TestReliabilityExceptionBranches:
    """Test exception branches in reliability module."""

    def test_weibull_neg_log_likelihood_invalid_params(self):
        """neg_log_likelihood returns 1e10 for beta/eta <= 0 (line 47)."""
        from stats_engine.reliability import _weibull

        times = np.array([10.0, 20.0, 30.0])
        status = np.array([1.0, 1.0, 1.0])

        # Mock minimize to capture the objective function and call it with invalid params
        def capture_objective(func, x0, bounds=None):
            # Call the objective with invalid params (beta=0, eta=-1)
            val_zero_beta = func([0, 10.0])
            val_neg_eta = func([1.0, -1.0])
            assert val_zero_beta == 1e10
            assert val_neg_eta == 1e10
            # Return a fake result so the rest of the function works
            mock_result = MagicMock()
            mock_result.x = [1.0, 20.0]
            return mock_result

        with patch("scipy.optimize.minimize", side_effect=capture_objective):
            result = _weibull(times, status)

        assert "beta" in result
        assert "eta" in result

    def test_distribution_fit_lognormal_error(self):
        """_distribution_fit handles lognormal fit failure (lines 174-175)."""
        from stats_engine.reliability import _distribution_fit

        # All zeros: log(times[times > 0]) is empty -> norm.fit fails
        times = [0.0, 0.0, 0.0]
        result = _distribution_fit(times)
        # Should still return a result, with lognormal having an error
        assert "distributions" in result
        # The lognormal fit may fail or succeed with edge case, but the function should not crash
        assert "lognormal" in result["distributions"]

    def test_distribution_fit_exponential_error(self):
        """_distribution_fit handles exponential fit failure (lines 184-185)."""
        from stats_engine.reliability import _distribution_fit

        # Mock np.mean to raise to trigger the exponential except block
        with patch("stats_engine.reliability.np.mean", side_effect=ValueError("mock error")):
            result = _distribution_fit([10.0, 20.0, 30.0])

        assert "distributions" in result
        assert "exponential" in result["distributions"]
        assert "error" in result["distributions"]["exponential"]

    def test_distribution_fit_normal_error(self):
        """_distribution_fit handles normal fit failure (lines 194-195)."""
        from stats_engine.reliability import _distribution_fit

        # Mock sp_stats.norm.fit to raise for the normal branch only
        call_count = {"n": 0}

        def mock_norm_fit(data, *args, **kwargs):
            call_count["n"] += 1
            # First call is for lognormal, second is for normal
            if call_count["n"] >= 2:
                raise ValueError("mock fit error")
            import scipy.stats as sp

            return sp.norm.fit(data, *args, **kwargs)

        with patch("stats_engine.reliability.sp_stats.norm.fit", side_effect=mock_norm_fit):
            result = _distribution_fit([10.0, 20.0, 30.0])

        assert "distributions" in result
        assert "normal" in result["distributions"]
        assert "error" in result["distributions"]["normal"]

    def test_distribution_fit_weibull_error(self):
        """_distribution_fit handles Weibull fit failure (lines 162-163)."""
        from stats_engine.reliability import _distribution_fit

        with patch("stats_engine.reliability._weibull", side_effect=ValueError("mock weibull error")):
            result = _distribution_fit([10.0, 20.0, 30.0])

        assert "distributions" in result
        assert "weibull" in result["distributions"]
        assert "error" in result["distributions"]["weibull"]

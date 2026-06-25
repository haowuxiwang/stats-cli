"""Tests for stats_engine/distribution.py."""

import numpy as np
import pytest

from stats_engine.distribution import distribution


class TestDistributionFit:
    def test_fit_normal(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("fit", values=values, dist_name="normal")
        assert result["analysis_type"] == "fit"
        assert result["distribution"] == "normal"
        assert "parameters" in result
        assert "log_likelihood" in result
        assert "aic" in result
        assert "bic" in result
        assert "ks_statistic" in result
        assert "ks_p_value" in result
        assert "interpretation" in result

    def test_fit_lognormal(self):
        np.random.seed(42)
        values = np.random.lognormal(2, 0.5, 50).tolist()
        result = distribution("fit", values=values, dist_name="lognormal")
        assert result["distribution"] == "lognormal"
        assert "parameters" in result

    def test_fit_exponential(self):
        np.random.seed(42)
        values = np.random.exponential(5, 50).tolist()
        result = distribution("fit", values=values, dist_name="exponential")
        assert result["distribution"] == "exponential"

    def test_fit_gamma(self):
        np.random.seed(42)
        values = np.random.gamma(2, 2, 50).tolist()
        result = distribution("fit", values=values, dist_name="gamma")
        assert result["distribution"] == "gamma"

    def test_fit_weibull(self):
        np.random.seed(42)
        values = np.random.weibull(2, 50).tolist()
        result = distribution("fit", values=values, dist_name="weibull")
        assert result["distribution"] == "weibull"

    def test_fit_mom_method(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("fit", values=values, dist_name="normal", method="mom")
        assert result["method"] == "mom"

    def test_fit_unknown_dist(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            distribution("fit", values=[1, 2, 3, 4, 5], dist_name="unknown")

    def test_fit_insufficient_data(self):
        with pytest.raises(ValueError):
            distribution("fit", values=[1, 2], dist_name="normal")


class TestDistributionGof:
    def test_gof_normal(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 100).tolist()
        result = distribution("gof", values=values, dist_name="normal")
        assert result["analysis_type"] == "gof"
        assert "ks_statistic" in result
        assert "ks_p_value" in result
        assert "chi2_statistic" in result
        assert "chi2_p_value" in result
        assert "anderson_darling" in result

    def test_gof_interpretation(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 100).tolist()
        result = distribution("gof", values=values, dist_name="normal")
        assert "interpretation" in result


class TestDistributionSelect:
    def test_select_default(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values)
        assert result["analysis_type"] == "select"
        assert "rankings" in result
        assert "best_distribution" in result
        assert "best_parameters" in result
        assert result["criterion"] == "aic"
        # Normal should rank high for normal data
        dist_names = [r["distribution"] for r in result["rankings"] if "error" not in r]
        assert "normal" in dist_names

    def test_select_bic(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values, criterion="bic")
        assert result["criterion"] == "bic"

    def test_select_subset(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values, distributions=["normal", "lognormal"])
        assert result["n_distributions"] == 2

    def test_select_rankings_sorted(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values)
        valid = [r for r in result["rankings"] if "error" not in r]
        aic_values = [r["aic"] for r in valid]
        assert aic_values == sorted(aic_values)

    def test_select_unknown_dist(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            distribution("select", values=[1, 2, 3, 4, 5], distributions=["unknown"])


class TestDistributionCoverage:
    """Coverage improvement tests for distribution.py."""

    def test_fit_mom_exponential(self):
        np.random.seed(42)
        values = np.random.exponential(5, 50).tolist()
        result = distribution("fit", values=values, dist_name="exponential", method="mom")
        assert result["method"] == "mom"
        assert result["distribution"] == "exponential"

    def test_fit_mom_lognormal(self):
        np.random.seed(42)
        values = np.random.lognormal(2, 0.5, 50).tolist()
        result = distribution("fit", values=values, dist_name="lognormal", method="mom")
        assert result["method"] == "mom"
        assert result["distribution"] == "lognormal"

    def test_fit_mom_gamma_fallback(self):
        np.random.seed(42)
        values = np.random.gamma(2, 2, 50).tolist()
        result = distribution("fit", values=values, dist_name="gamma", method="mom")
        assert result["method"] == "mom"
        assert result["distribution"] == "gamma"

    def test_gof_small_sample(self):
        """Small sample should trigger bin merging in chi-square test."""
        np.random.seed(42)
        values = np.random.normal(10, 1, 20).tolist()
        result = distribution("gof", values=values, dist_name="normal")
        assert "chi2_statistic" in result

    def test_gof_lognormal(self):
        np.random.seed(42)
        values = np.random.lognormal(2, 0.5, 100).tolist()
        result = distribution("gof", values=values, dist_name="lognormal")
        assert "ks_statistic" in result

    def test_select_with_failing_dist(self):
        """Select should handle distributions that fail to fit."""
        # Beta distribution requires values in (0, 1), so fitting to unbounded data may fail
        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values, distributions=["normal", "beta"])
        assert "rankings" in result
        # Should still have results (beta may fail but normal should work)
        valid = [r for r in result["rankings"] if "error" not in r]
        assert len(valid) >= 1

    def test_gof_chi2_large_sample(self):
        """Large sample should produce valid chi-square test."""
        np.random.seed(42)
        values = np.random.normal(10, 2, 200).tolist()
        result = distribution("gof", values=values, dist_name="normal")
        assert result["chi2_statistic"] is not None
        assert result["chi2_p_value"] is not None

    def test_fit_weibull_params(self):
        np.random.seed(42)
        values = (np.random.weibull(2, 100) * 10).tolist()
        result = distribution("fit", values=values, dist_name="weibull")
        assert result["distribution"] == "weibull"
        assert "c" in result["parameters"]

    def test_fit_beta(self):
        np.random.seed(42)
        values = np.random.beta(2, 5, 100).tolist()
        result = distribution("fit", values=values, dist_name="beta")
        assert result["distribution"] == "beta"
        assert "a" in result["parameters"]
        assert "b" in result["parameters"]

    def test_fit_logistic(self):
        np.random.seed(42)
        values = np.random.logistic(10, 1, 100).tolist()
        result = distribution("fit", values=values, dist_name="logistic")
        assert result["distribution"] == "logistic"

    def test_fit_gumbel(self):
        np.random.seed(42)
        values = np.random.gumbel(10, 1, 100).tolist()
        result = distribution("fit", values=values, dist_name="gumbel")
        assert result["distribution"] == "gumbel"


class TestDistributionJsonSerializable:
    def test_fit_serializable(self):
        import json

        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("fit", values=values)
        json.dumps(result)  # Should not raise

    def test_select_serializable(self):
        import json

        np.random.seed(42)
        values = np.random.normal(10, 2, 50).tolist()
        result = distribution("select", values=values)
        json.dumps(result)  # Should not raise

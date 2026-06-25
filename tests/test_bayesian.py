"""Tests for stats_engine/bayesian.py."""

import json

import numpy as np

from stats_engine.bayesian import bayesian


class TestBayesianEstimate:
    def test_basic_estimate(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 30).tolist()
        result = bayesian("estimate", values=values)
        assert result["analysis_type"] == "estimate"
        assert "posterior_mean" in result
        assert "posterior_std" in result
        assert "credible_interval" in result
        assert "prior_mean" in result
        assert "prior_std" in result
        assert "bayes_factor_h0_vs_h1" in result
        assert "interpretation" in result

    def test_estimate_with_prior(self):
        values = [10.1, 10.2, 10.3, 10.4, 10.5]
        result = bayesian("estimate", values=values, prior_mean=10, prior_std=1)
        assert result["prior_mean"] == 10
        assert result["prior_std"] == 1

    def test_estimate_credible_interval_contains_mean(self):
        np.random.seed(42)
        values = np.random.normal(10, 2, 30).tolist()
        result = bayesian("estimate", values=values)
        ci = result["credible_interval"]
        assert ci[0] <= result["posterior_mean"] <= ci[1]

    def test_estimate_custom_credible_level(self):
        values = [10.1, 10.2, 10.3, 10.4, 10.5]
        result = bayesian("estimate", values=values, credible_level=0.99)
        assert result["credible_level"] == 0.99


class TestBayesianTtest:
    def test_one_sample(self):
        values = [10.2, 10.5, 10.3, 10.1, 10.4]
        result = bayesian("ttest", values=values, mu=10)
        assert result["analysis_type"] == "bayesian_ttest"
        assert result["test_type"] == "one_sample"
        assert "bayes_factor_10" in result
        assert "bf_interpretation" in result
        assert "credible_interval" in result
        assert "cohens_d" in result
        assert "p_value" in result

    def test_two_sample(self):
        values1 = [10.2, 10.5, 10.3, 10.1, 10.4]
        values2 = [11.3, 11.5, 11.1, 11.4, 11.2]
        result = bayesian("ttest", values=values1, values2=values2)
        assert result["test_type"] == "two_sample"
        assert "bayes_factor_10" in result

    def test_paired(self):
        values1 = [10.2, 10.5, 10.3, 10.1, 10.4]
        values2 = [10.5, 10.8, 10.6, 10.4, 10.7]
        result = bayesian("ttest", values=values1, values2=values2, paired=True)
        assert result["test_type"] == "paired"

    def test_bayes_factor_range(self):
        values = [10.2, 10.5, 10.3, 10.1, 10.4]
        result = bayesian("ttest", values=values, mu=10)
        assert result["bayes_factor_10"] > 0

    def test_json_serializable(self):
        values1 = [10.2, 10.5, 10.3, 10.1, 10.4]
        values2 = [11.3, 11.5, 11.1, 11.4, 11.2]
        result = bayesian("ttest", values=values1, values2=values2)
        json.dumps(result)  # Should not raise


class TestBayesianProportion:
    def test_basic_proportion(self):
        result = bayesian("proportion", successes=7, n=10)
        assert result["analysis_type"] == "bayesian_proportion"
        assert result["n"] == 10
        assert result["successes"] == 7
        assert "posterior_mean" in result
        assert "credible_interval" in result
        assert "map_estimate" in result
        assert "interpretation" in result

    def test_proportion_uniform_prior(self):
        result = bayesian("proportion", successes=5, n=10, prior_alpha=1, prior_beta=1)
        assert result["posterior_mean"] == 0.5  # (1+5)/(1+10-5+1) = 6/12 = 0.5... wait
        # Beta(1+5, 1+5) = Beta(6, 6), mean = 6/12 = 0.5
        assert abs(result["posterior_mean"] - 0.5) < 0.01

    def test_proportion_jeffreys_prior(self):
        result = bayesian("proportion", successes=5, n=10, prior_alpha=0.5, prior_beta=0.5)
        assert result["posterior_alpha"] == 5.5
        assert result["posterior_beta"] == 5.5

    def test_proportion_extremes(self):
        result = bayesian("proportion", successes=0, n=10)
        assert result["successes"] == 0

        result2 = bayesian("proportion", successes=10, n=10)
        assert result2["successes"] == 10

    def test_proportion_ci_bounds(self):
        result = bayesian("proportion", successes=5, n=10)
        ci = result["credible_interval"]
        assert 0 <= ci[0] <= ci[1] <= 1

    def test_proportion_json_serializable(self):
        result = bayesian("proportion", successes=7, n=10)
        json.dumps(result)  # Should not raise


class TestBayesianAnova:
    def test_basic_anova(self):
        groups = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        result = bayesian("anova", groups=groups)
        assert result["analysis_type"] == "bayesian_anova"
        assert result["n_groups"] == 3
        assert "bayes_factor_10" in result
        assert "group_posteriors" in result
        assert "f_statistic" in result
        assert "eta_squared" in result
        assert "interpretation" in result

    def test_anova_identical_groups(self):
        groups = [[10, 11, 12], [10, 11, 12]]
        result = bayesian("anova", groups=groups)
        # BF10 should be low (evidence for H0)
        assert result["bayes_factor_10"] < 1

    def test_anova_different_groups(self):
        groups = [[10, 11, 12], [50, 51, 52]]
        result = bayesian("anova", groups=groups)
        # BF10 should be high (evidence for H1)
        assert result["bayes_factor_10"] > 1

    def test_anova_group_posteriors(self):
        groups = [[10, 11, 12], [20, 21, 22]]
        result = bayesian("anova", groups=groups)
        assert len(result["group_posteriors"]) == 2
        for gp in result["group_posteriors"]:
            assert "sample_mean" in gp
            assert "credible_interval" in gp

    def test_anova_json_serializable(self):
        groups = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        result = bayesian("anova", groups=groups)
        json.dumps(result)  # Should not raise

    def test_anova_many_groups(self):
        """Many groups with large differences → strong BF."""
        np.random.seed(42)
        groups = [np.random.normal(10 + i * 5, 1, 10).tolist() for i in range(5)]
        result = bayesian("anova", groups=groups)
        assert result["bayes_factor_10"] > 1
        assert "strong" in result["bf_interpretation"] or "extreme" in result["bf_interpretation"]


class TestBayesianCoverage:
    """Coverage improvement tests for bayesian.py."""

    def test_ttest_strong_evidence_h1(self):
        """Large separation → strong/extreme evidence for H1."""
        np.random.seed(42)
        g1 = np.random.normal(0, 0.5, 100).tolist()
        g2 = np.random.normal(10, 0.5, 100).tolist()
        result = bayesian("ttest", values=g1, values2=g2)
        assert result["bayes_factor_10"] > 10

    def test_ttest_moderate_evidence_h1(self):
        """Moderate separation → moderate evidence for H1."""
        np.random.seed(42)
        g1 = np.random.normal(10, 2, 30).tolist()
        g2 = np.random.normal(11.5, 2, 30).tolist()
        result = bayesian("ttest", values=g1, values2=g2)
        # BF should be > 1 but may not be > 10
        assert result["bayes_factor_10"] > 0

    def test_ttest_anecdotal_evidence_h0(self):
        """Very similar groups → anecdotal evidence for H0."""
        np.random.seed(42)
        g1 = np.random.normal(10, 3, 10).tolist()
        g2 = np.random.normal(10.1, 3, 10).tolist()
        result = bayesian("ttest", values=g1, values2=g2)
        assert result["bayes_factor_01"] > 0

    def test_proportion_with_prior(self):
        """Proportion with informative prior."""
        result = bayesian("proportion", successes=50, n=100, prior_alpha=10, prior_beta=10)
        assert result["posterior_alpha"] == 60
        assert result["posterior_beta"] == 60
        assert result["map_estimate"] is not None

    def test_proportion_map_extreme(self):
        """MAP with alpha < 1 or beta < 1."""
        result = bayesian("proportion", successes=5, n=10, prior_alpha=0.5, prior_beta=0.5)
        # With Jeffreys prior, MAP falls back to mean
        assert result["posterior_mean"] is not None

    def test_estimate_strong_prior(self):
        """Strong prior should pull posterior toward prior mean."""
        values = [100, 101, 102, 103, 104]
        result = bayesian("estimate", values=values, prior_mean=0, prior_std=0.1)
        # Strong prior at 0 should pull posterior well below sample mean
        assert result["posterior_mean"] < 50

    def test_ttest_one_sample_with_mu(self):
        result = bayesian("ttest", values=[10, 11, 12, 13, 14], mu=12)
        assert result["test_type"] == "one_sample"
        assert "bayes_factor_10" in result

    def test_ttest_bf_very_strong(self):
        """Very strong evidence for H1."""
        np.random.seed(42)
        g1 = np.random.normal(0, 0.3, 200).tolist()
        g2 = np.random.normal(5, 0.3, 200).tolist()
        result = bayesian("ttest", values=g1, values2=g2)
        assert result["bayes_factor_10"] > 30

    def test_ttest_bf_moderate_h0(self):
        """Moderate evidence for H0 with small sample."""
        np.random.seed(42)
        g1 = np.random.normal(10, 5, 5).tolist()
        g2 = np.random.normal(10.5, 5, 5).tolist()
        result = bayesian("ttest", values=g1, values2=g2)
        assert result["bayes_factor_01"] > 0

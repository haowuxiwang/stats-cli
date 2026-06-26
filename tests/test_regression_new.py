"""Tests for PLS and GLM regression types."""

import json

import numpy as np
import pytest

from stats_engine.regression import regression


class TestPLSRegression:
    """Tests for reg_type='pls'."""

    def test_pls_basic(self):
        """PLS regression returns expected keys."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(30) * 0.1

        result = regression(x=X, y=y, reg_type="pls", n_components=2)

        assert result["regression_type"] == "pls"
        assert result["n"] == 30
        assert result["n_components"] == 2
        assert "coefficients" in result
        assert "vip_scores" in result
        assert "r_squared" in result
        assert "x_scores" in result
        assert "y_scores" in result

    def test_pls_r_squared_reasonable(self):
        """R-squared should be between 0 and 1 for well-specified data."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 4))
        y = 3 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + rng.standard_normal(50) * 0.2

        result = regression(x=X, y=y, reg_type="pls", n_components=3)
        assert 0.0 <= result["r_squared"] <= 1.0
        assert result["r_squared"] > 0.5  # strong signal

    def test_pls_vip_scores_positive(self):
        """VIP scores must be non-negative."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((40, 5))
        y = X[:, 0] + 0.3 * X[:, 2] + rng.standard_normal(40) * 0.5

        result = regression(x=X, y=y, reg_type="pls", n_components=2)

        for key, val in result["vip_scores"].items():
            assert val >= 0, f"VIP score for {key} is negative: {val}"

    def test_pls_n_components_clamped(self):
        """n_components is clamped to min(n, p)."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((10, 3))
        y = X[:, 0] + rng.standard_normal(10) * 0.1

        result = regression(x=X, y=y, reg_type="pls", n_components=20)
        assert result["n_components"] == 3  # min(10, 3, 20) = 3

    def test_pls_single_predictor(self):
        """PLS with 1 predictor works like simple regression."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal(20)
        y = 1.5 * x + rng.standard_normal(20) * 0.1

        result = regression(x=x, y=y, reg_type="pls", n_components=1)
        assert result["regression_type"] == "pls"
        assert "x1" in result["vip_scores"]


class TestGLMRegression:
    """Tests for reg_type='poisson', 'gamma', 'negbin'."""

    def test_glm_poisson(self):
        """GLM Poisson with count data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        # Generate count data (Poisson-distributed)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)

        result = regression(x=x, y=y, reg_type="poisson")

        assert result["regression_type"] == "glm_poisson"
        assert result["family"] == "poisson"
        assert result["n"] == 100
        assert "coefficients" in result
        assert "deviance" in result
        assert "aic" in result
        assert "bic" in result
        assert "pseudo_r_squared" in result
        assert "summary_table" in result

    def test_glm_gamma(self):
        """GLM Gamma with positive continuous data."""
        rng = np.random.default_rng(13)
        x = rng.standard_normal(80)
        mu = np.exp(1.0 + 0.2 * x)
        y = rng.gamma(shape=2, scale=mu / 2)

        result = regression(x=x, y=y, reg_type="gamma")

        assert result["regression_type"] == "glm_gamma"
        assert result["family"] == "gamma"
        assert result["n"] == 80
        assert result["deviance"] > 0
        assert result["aic"] > 0

    def test_glm_negbin(self):
        """GLM Negative Binomial with overdispersed count data."""
        rng = np.random.default_rng(21)
        x = rng.standard_normal(60)
        mu = np.exp(0.8 + 0.4 * x)
        # NegBin via gamma-poisson mixture
        lam = rng.gamma(shape=2, scale=mu / 2)
        y = np.array([rng.poisson(lam_val) for lam_val in lam], dtype=float)

        result = regression(x=x, y=y, reg_type="negbin")

        assert result["regression_type"] == "glm_negbin"
        assert result["family"] == "negbin"
        assert "deviance" in result
        assert "aic" in result

    def test_glm_multiple_predictors(self):
        """GLM with multiple predictors."""
        rng = np.random.default_rng(33)
        X = rng.standard_normal((100, 3))
        mu = np.exp(0.5 + 0.3 * X[:, 0] + 0.1 * X[:, 1])
        y = rng.poisson(mu).astype(float)

        result = regression(x=X, y=y, reg_type="poisson")

        assert "x1" in result["coefficients"]
        assert "x2" in result["coefficients"]
        assert "x3" in result["coefficients"]
        assert "intercept" in result["coefficients"]

    def test_glm_invalid_family(self):
        """Invalid GLM family raises ValueError."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)

        with pytest.raises(ValueError, match="Unknown GLM family"):
            # Route through _glm directly since regression() won't match
            from stats_engine.regression import _glm

            _glm(x, y, "invalid_family")


class TestJSONSerialization:
    """Test that all new regression types produce JSON-serializable output."""

    def test_pls_json(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = X[:, 0] + rng.standard_normal(20) * 0.1

        result = regression(x=X, y=y, reg_type="pls", n_components=2)
        # Should not raise
        json_str = json.dumps(result)
        assert "pls" in json_str

    def test_glm_poisson_json(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)

        result = regression(x=x, y=y, reg_type="poisson")
        json_str = json.dumps(result)
        assert "glm_poisson" in json_str

    def test_glm_gamma_json(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.gamma(shape=2, scale=np.exp(1.0 + 0.2 * x) / 2)

        result = regression(x=x, y=y, reg_type="gamma")
        json_str = json.dumps(result)
        assert "glm_gamma" in json_str

    def test_glm_negbin_json(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(40)
        mu = np.exp(0.8 + 0.4 * x)
        lam = rng.gamma(shape=2, scale=mu / 2)
        y = np.array([rng.poisson(lam_val) for lam_val in lam], dtype=float)

        result = regression(x=x, y=y, reg_type="negbin")
        json_str = json.dumps(result)
        assert "glm_negbin" in json_str

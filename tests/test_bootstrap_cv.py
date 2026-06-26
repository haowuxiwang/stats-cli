"""Tests for bootstrap (advanced.py) and cross-validation (regression.py)."""

import json

import numpy as np
import pytest

from stats_engine.advanced import advanced
from stats_engine.regression import regression

# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------


class TestBootstrap:
    """Tests for advanced(analysis_type='bootstrap')."""

    def test_bootstrap_mean_ci_contains_true_mean(self):
        """Bootstrap 95% CI for mean should contain the true population mean."""
        np.random.seed(0)
        data = np.random.normal(loc=50, scale=5, size=100).tolist()
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="mean",
            n_bootstrap=5000,
            confidence_level=0.95,
            seed=42,
        )
        assert result["analysis_type"] == "bootstrap"
        assert result["statistic"] == "mean"
        assert result["ci_lower"] < 50 < result["ci_upper"]

    def test_bootstrap_median(self):
        """Bootstrap CI for median should contain true median."""
        np.random.seed(1)
        data = np.random.exponential(scale=2.0, size=200).tolist()
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="median",
            n_bootstrap=5000,
            confidence_level=0.95,
            seed=123,
        )
        true_median = float(np.median(data))
        assert result["ci_lower"] <= true_median <= result["ci_upper"]

    def test_bootstrap_known_data(self):
        """Bootstrap with known data should return reproducible results."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="mean",
            n_bootstrap=1000,
            confidence_level=0.95,
            seed=42,
        )
        assert result["original_statistic"] == 3.0
        assert result["n_bootstrap"] == 1000
        assert result["n_obs"] == 5
        assert result["seed"] == 42
        assert result["confidence_level"] == 0.95
        # CI should bracket the true mean of 3.0
        assert result["ci_lower"] <= 3.0 <= result["ci_upper"]
        # Bootstrap mean should be close to original
        assert abs(result["bootstrap_mean"] - 3.0) < 0.5

    def test_bootstrap_std(self):
        """Bootstrap CI for std."""
        data = [10.0, 12.0, 11.0, 13.0, 10.5, 12.5, 11.5, 13.5]
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="std",
            n_bootstrap=2000,
            confidence_level=0.90,
            seed=7,
        )
        true_std = float(np.std(data))
        assert result["statistic"] == "std"
        assert result["ci_lower"] <= true_std <= result["ci_upper"]

    def test_bootstrap_distribution_summary(self):
        """Bootstrap distribution summary should contain all expected percentiles."""
        data = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="mean",
            n_bootstrap=500,
            seed=0,
        )
        summary = result["bootstrap_distribution_summary"]
        expected_pct = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        for p in expected_pct:
            key = f"p{p}"
            assert key in summary
            assert isinstance(summary[key], float)

    def test_bootstrap_bias(self):
        """Bootstrap bias should be small for unbiased estimators like mean."""
        np.random.seed(99)
        data = np.random.normal(10, 1, 500).tolist()
        result = advanced(
            analysis_type="bootstrap",
            values=data,
            statistic="mean",
            n_bootstrap=5000,
            seed=42,
        )
        assert abs(result["bias"]) < 0.1

    def test_bootstrap_bad_values(self):
        """Bootstrap with < 2 valid points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            advanced(analysis_type="bootstrap", values=[1.0])

    def test_bootstrap_bad_confidence(self):
        """Invalid confidence_level should raise ValueError."""
        with pytest.raises(ValueError, match="confidence_level"):
            advanced(
                analysis_type="bootstrap",
                values=[1, 2, 3, 4, 5],
                confidence_level=1.5,
            )

    def test_bootstrap_bad_n_bootstrap(self):
        """n_bootstrap < 100 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            advanced(
                analysis_type="bootstrap",
                values=[1, 2, 3, 4, 5],
                n_bootstrap=50,
            )

    def test_bootstrap_bad_statistic(self):
        """Unknown statistic name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            advanced(
                analysis_type="bootstrap",
                values=[1, 2, 3, 4, 5],
                statistic="skewness",
            )

    def test_bootstrap_json_serializable(self):
        """Bootstrap result should be JSON-serializable."""
        result = advanced(
            analysis_type="bootstrap",
            values=[10.0, 20.0, 30.0, 40.0, 50.0],
            statistic="mean",
            n_bootstrap=500,
            seed=42,
        )
        # Should not raise
        json_str = json.dumps(result)
        assert "bootstrap" in json_str


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Tests for regression(reg_type='cross_validate') and cv parameter."""

    def test_cv_linear(self):
        """Cross-validation with linear regression."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 1, 50)
        result = regression(x=x.tolist(), y=y.tolist(), reg_type="cross_validate", cv=5)
        assert result["regression_type"] == "cross_validate"
        assert result["inner_type"] == "linear"
        assert result["cv_folds"] == 5
        assert result["scoring"] == "r2"
        assert len(result["scores"]) == 5
        assert result["mean_score"] > 0.8  # strong linear relationship

    def test_cv_5_folds(self):
        """Cross-validation with 5 folds returns correct number of scores."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = regression(x=x, y=y, reg_type="cross_validate", cv=5)
        assert len(result["scores"]) == 5
        assert result["min_score"] <= result["mean_score"] <= result["max_score"]

    def test_cv_scores_reasonable(self):
        """CV R^2 scores should be in [0, 1] for a good linear fit."""
        np.random.seed(0)
        x = np.arange(1, 21, dtype=float)
        y = 3 * x + 5 + np.random.normal(0, 0.5, 20)
        result = regression(x=x.tolist(), y=y.tolist(), reg_type="cross_validate", cv=5)
        for score in result["scores"]:
            assert 0.0 <= score <= 1.0

    def test_cv_via_parameter(self):
        """Using cv=5 parameter with reg_type='linear' should trigger CV mode."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 5, 8, 10, 11, 14, 16, 18, 20]
        result = regression(x=x, y=y, reg_type="linear", cv=5)
        assert result["regression_type"] == "cross_validate"
        assert result["inner_type"] == "linear"
        assert len(result["scores"]) == 5

    def test_cv_ridge(self):
        """Cross-validation with ridge regression via cv parameter."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 1.5 * x + 2 + np.random.normal(0, 0.5, 30)
        result = regression(x=x.tolist(), y=y.tolist(), reg_type="ridge", cv=3, scoring="r2")
        assert result["inner_type"] == "ridge"
        assert len(result["scores"]) == 3

    def test_cv_bad_inner_type(self):
        """CV with unsupported inner type should raise ValueError."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        # logistic is not in _CV_ESTIMATORS
        with pytest.raises(ValueError, match="Cross-validation not supported"):
            regression(
                x=x,
                y=y,
                reg_type="logistic",
                cv=2,
            )

    def test_cv_bad_folds(self):
        """cv < 2 should raise ValueError."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        with pytest.raises(ValueError, match="cv must be at least 2"):
            regression(x=x, y=y, reg_type="cross_validate", cv=1)

    def test_cv_too_many_folds(self):
        """cv > n should raise ValueError."""
        x = [1, 2, 3]
        y = [2, 4, 6]
        with pytest.raises(ValueError, match="cannot exceed"):
            regression(x=x, y=y, reg_type="cross_validate", cv=5)

    def test_cv_json_serializable(self):
        """CV result should be JSON-serializable."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = regression(x=x, y=y, reg_type="cross_validate", cv=5)
        json_str = json.dumps(result)
        assert "cross_validate" in json_str

    def test_cv_mse_scoring(self):
        """Cross-validation with neg_mean_squared_error scoring."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = regression(
            x=x,
            y=y,
            reg_type="cross_validate",
            cv=5,
            scoring="neg_mean_squared_error",
        )
        assert result["scoring"] == "neg_mean_squared_error"
        assert len(result["scores"]) == 5

    def test_cv_interpretation(self):
        """CV result should have an interpretation string."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = regression(x=x, y=y, reg_type="cross_validate", cv=5)
        assert "interpretation" in result
        assert "5-fold CV" in result["interpretation"]

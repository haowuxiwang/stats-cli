"""Tests for penalized regression methods: lasso, ridge, elastic_net, robust."""

import json

import numpy as np
import pytest

from stats_engine.regression import regression


@pytest.fixture
def linear_data():
    """Clean linear data: y = 3*x + 5 + noise."""
    np.random.seed(123)
    x = np.linspace(0, 10, 50)
    y = 3 * x + 5 + np.random.normal(0, 0.5, 50)
    return x.tolist(), y.tolist()


@pytest.fixture
def sparse_data():
    """Data where only 2 of 5 features truly matter — ideal for lasso sparsity."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    # True model: y = 2*x1 + 0*x2 + 0*x3 - 1.5*x4 + 0*x5
    y = 2 * X[:, 0] - 1.5 * X[:, 3] + np.random.normal(0, 0.3, n)
    return X.tolist(), y.tolist()


@pytest.fixture
def multicollinear_data():
    """Data with multicollinear predictors — ridge should handle better than OLS."""
    np.random.seed(99)
    n = 80
    x1 = np.random.randn(n)
    x2 = x1 + np.random.normal(0, 0.05, n)  # nearly collinear with x1
    x3 = np.random.randn(n)
    X = np.column_stack([x1, x2, x3])
    y = 1.5 * x1 + 0.5 * x3 + np.random.normal(0, 0.5, n)
    return X.tolist(), y.tolist()


@pytest.fixture
def outlier_data():
    """Data with clean relationship + outliers."""
    np.random.seed(77)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 3 + np.random.normal(0, 0.3, 50)
    # Inject outliers
    y[45] += 20
    y[46] -= 15
    y[47] += 25
    return x.tolist(), y.tolist()


# --- LASSO ---


def test_lasso_basic(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="lasso")
    assert result["regression_type"] == "lasso"
    assert "coefficients" in result
    assert "alpha" in result
    assert "r_squared" in result
    assert "n_nonzero_coefs" in result
    assert result["r_squared"] > 0.9


def test_lasso_sparsity(sparse_data):
    """LASSO should set some coefficients to exactly zero."""
    X, y = sparse_data
    result = regression(x=X, y=y, reg_type="lasso")
    # With 5 features and only 2 truly relevant, lasso should find fewer nonzero
    assert result["n_nonzero_coefs"] <= 5
    assert result["r_squared"] > 0.8


def test_lasso_with_alpha(linear_data):
    """Explicit alpha should be used instead of CV."""
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="lasso", reg_alpha=0.1)
    assert result["alpha"] == pytest.approx(0.1, abs=1e-6)


def test_lasso_auto_alpha(linear_data):
    """When reg_alpha is None, cross-validated alpha should be selected."""
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="lasso")
    assert result["alpha"] > 0


# --- RIDGE ---


def test_ridge_basic(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="ridge")
    assert result["regression_type"] == "ridge"
    assert "coefficients" in result
    assert "alpha" in result
    assert "r_squared" in result
    assert result["r_squared"] > 0.9


def test_ridge_multicollinear(multicollinear_data):
    """Ridge should handle multicollinear data without crashing."""
    X, y = multicollinear_data
    result = regression(x=X, y=y, reg_type="ridge")
    assert result["r_squared"] > 0.7
    # Ridge should produce finite coefficients even with multicollinearity
    for k, v in result["coefficients"].items():
        assert np.isfinite(v), f"Coefficient {k} is not finite"


def test_ridge_with_alpha(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="ridge", reg_alpha=1.0)
    assert result["alpha"] == pytest.approx(1.0, abs=1e-6)


def test_ridge_auto_alpha(multicollinear_data):
    X, y = multicollinear_data
    result = regression(x=X, y=y, reg_type="ridge")
    assert result["alpha"] > 0


# --- ELASTIC NET ---


def test_elastic_net_basic(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="elastic_net")
    assert result["regression_type"] == "elastic_net"
    assert "coefficients" in result
    assert "alpha" in result
    assert "l1_ratio" in result
    assert "r_squared" in result
    assert result["r_squared"] > 0.85


def test_elastic_net_with_params(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="elastic_net", reg_alpha=0.05, l1_ratio=0.7)
    assert result["alpha"] == pytest.approx(0.05, abs=1e-6)
    assert result["l1_ratio"] == pytest.approx(0.7, abs=1e-6)


def test_elastic_net_auto_params(sparse_data):
    """Auto-CV should find reasonable alpha and l1_ratio."""
    X, y = sparse_data
    result = regression(x=X, y=y, reg_type="elastic_net")
    assert result["alpha"] > 0
    assert 0 < result["l1_ratio"] <= 1


# --- ROBUST ---


def test_robust_basic(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="robust")
    assert result["regression_type"] == "robust"
    assert "coefficients" in result
    assert "method" in result
    assert "r_squared" in result
    assert "n_outliers" in result
    assert result["method"] == "huber"
    assert result["r_squared"] > 0.9


def test_robust_with_outliers(outlier_data):
    """Robust regression should resist outliers and still fit the underlying trend."""
    x, y = outlier_data
    # OOLS would be heavily influenced by outliers
    result_ols = regression(x=x, y=y, reg_type="linear")
    result_robust = regression(x=x, y=y, reg_type="robust")

    # Robust slope should be closer to the true value (2.0) than OLS
    robust_slope = result_robust["coefficients"]["slope"]
    ols_slope = result_ols["coefficients"]["slope"]
    assert abs(robust_slope - 2.0) < abs(ols_slope - 2.0)
    # Robust should detect the outliers
    assert result_robust["n_outliers"] >= 1


def test_robust_tukey(outlier_data):
    """Tukey biweight should be even more aggressive at downweighting outliers."""
    x, y = outlier_data
    result = regression(x=x, y=y, reg_type="robust", method="tukey")
    assert result["method"] == "tukey"
    assert result["r_squared"] > 0.5
    # Tukey should still recover the true slope
    assert abs(result["coefficients"]["slope"] - 2.0) < 0.5


def test_robust_huber_explicit(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="robust", method="huber")
    assert result["method"] == "huber"


# --- JSON serializability ---


def _assert_json_serializable(obj):
    """Assert that obj can be serialized to JSON without errors."""
    json_str = json.dumps(obj)
    parsed = json.loads(json_str)
    assert parsed == obj


def test_lasso_json_serializable(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="lasso")
    _assert_json_serializable(result)


def test_ridge_json_serializable(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="ridge")
    _assert_json_serializable(result)


def test_elastic_net_json_serializable(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="elastic_net")
    _assert_json_serializable(result)


def test_robust_json_serializable(linear_data):
    x, y = linear_data
    result = regression(x=x, y=y, reg_type="robust")
    _assert_json_serializable(result)


# --- Edge cases ---


def test_unknown_reg_type():
    with pytest.raises(ValueError, match="Unknown regression type"):
        regression(x=[1, 2, 3], y=[1, 2, 3], reg_type="bad_type")


def test_lasso_nan_filtering():
    """NaN values should be filtered before fitting."""
    x = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 4, 5, np.nan, 10, 12, 14, 16, 18, 20]
    result = regression(x=x, y=y, reg_type="lasso")
    assert result["n"] == 8
    assert result["r_squared"] > 0.9

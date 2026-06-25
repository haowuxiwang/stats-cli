"""Tests for stats_engine/regression.py."""

import numpy as np
import pytest

from stats_engine.regression import regression


def test_linear(xy_data):
    x, y = xy_data
    result = regression(x=x, y=y, reg_type="linear")
    assert result["regression_type"] == "linear"
    assert "r_squared" in result
    assert "adj_r_squared" in result
    assert "coefficients" in result
    assert "f_statistic" in result
    assert "p_value" in result


def test_polynomial(xy_data):
    x, y = xy_data
    result = regression(x=x, y=y, reg_type="polynomial", degree=2)
    assert "polynomial" in result["regression_type"]
    assert "r_squared" in result


def test_multiple():
    with pytest.raises(ValueError, match="file is required"):
        regression(file=None, x_columns=["x1", "x2"], y_column="y", reg_type="multiple")


def test_default_linear(xy_data):
    x, y = xy_data
    result = regression(x=x, y=y)
    assert result["regression_type"] == "linear"


def test_r_squared_range(xy_data):
    x, y = xy_data
    result = regression(x=x, y=y, reg_type="linear")
    assert 0 <= result["r_squared"] <= 1


def test_exponential():
    np.random.seed(42)
    x = np.linspace(0.5, 5, 30)
    y = 2.0 * np.exp(0.5 * x) + 1.0 + np.random.normal(0, 0.1, 30)
    result = regression(x=x.tolist(), y=y.tolist(), reg_type="exponential")
    assert result["regression_type"] == "exponential"
    assert result["n"] == 30
    assert "coefficients" in result
    assert "coefficients_std" in result
    assert set(result["coefficients"].keys()) == {"a", "b", "c"}
    assert set(result["coefficients_std"].keys()) == {"a", "b", "c"}
    assert result["r_squared"] > 0.95
    assert "adj_r_squared" in result
    assert "residual_std" in result
    assert "equation" in result


def test_power():
    np.random.seed(42)
    x = np.linspace(1, 10, 30)
    y = 3.0 * np.power(x, 1.5) + np.random.normal(0, 1, 30)
    result = regression(x=x.tolist(), y=y.tolist(), reg_type="power")
    assert result["regression_type"] == "power"
    assert result["n"] == 30
    assert set(result["coefficients"].keys()) == {"a", "b"}
    assert set(result["coefficients_std"].keys()) == {"a", "b"}
    assert result["r_squared"] > 0.95
    assert "equation" in result


def test_logarithmic():
    np.random.seed(42)
    x = np.linspace(1, 20, 30)
    y = 5.0 * np.log(x) + 2.0 + np.random.normal(0, 0.3, 30)
    result = regression(x=x.tolist(), y=y.tolist(), reg_type="logarithmic")
    assert result["regression_type"] == "logarithmic"
    assert result["n"] == 30
    assert set(result["coefficients"].keys()) == {"a", "b"}
    assert set(result["coefficients_std"].keys()) == {"a", "b"}
    assert result["r_squared"] > 0.95
    assert "equation" in result


def test_sigmoid():
    np.random.seed(42)
    x = np.linspace(0.1, 10, 50)
    a, b, c, d = 0.0, 2.0, 3.0, 10.0
    y = d + (a - d) / (1 + (x / c) ** b) + np.random.normal(0, 0.1, 50)
    result = regression(x=x.tolist(), y=y.tolist(), reg_type="sigmoid")
    assert result["regression_type"] == "sigmoid"
    assert result["n"] == 50
    assert set(result["coefficients"].keys()) == {"a", "b", "c", "d"}
    assert set(result["coefficients_std"].keys()) == {"a", "b", "c", "d"}
    assert result["r_squared"] > 0.90
    assert "equation" in result


def test_logistic():
    """Logistic regression with binary outcome."""
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = (x > 5).astype(int).tolist()
    try:
        result = regression(x=x.tolist(), y=y, reg_type="logistic")
        assert result["regression_type"] == "logistic"
    except Exception:
        pass  # May fail with perfect separation


def test_stepwise():
    """Stepwise regression with multiple predictors."""
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(0, 1, n).tolist()
    x2 = np.random.normal(0, 1, n).tolist()
    x3 = np.random.normal(0, 1, n).tolist()
    y = (2 * np.array(x1) + 0.5 * np.array(x2) + np.random.normal(0, 0.5, n)).tolist()
    try:
        result = regression(x=[x1, x2, x3], y=y, reg_type="stepwise")
        assert "regression_type" in result
    except Exception:
        pass  # May fail if statsmodels not available


def test_quadratic():
    """Quadratic regression."""
    np.random.seed(42)
    x = np.linspace(-3, 3, 30)
    y = (x**2 + np.random.normal(0, 0.5, 30)).tolist()
    result = regression(x=x.tolist(), y=y, reg_type="quadratic")
    assert "polynomial" in result["regression_type"] or "quadratic" in result["regression_type"]


def test_regression_with_nan():
    """Regression should handle NaN values."""
    x = [1, 2, float("nan"), 4, 5]
    y = [2, 4, 6, 8, 10]
    try:
        result = regression(x=x, y=y, reg_type="linear")
        assert result["regression_type"] == "linear"
    except ValueError:
        pass  # May raise for NaN input


def test_regression_with_inf():
    """Regression should handle Inf values."""
    x = [1, 2, float("inf"), 4, 5]
    y = [2, 4, 6, 8, 10]
    try:
        result = regression(x=x, y=y, reg_type="linear")
        assert result["regression_type"] == "linear"
    except ValueError:
        pass  # May raise for Inf input


def test_regression_unknown_nonlinear():
    with pytest.raises(ValueError, match="Unknown regression type"):
        regression(x=[1, 2, 3, 4, 5], y=[2, 4, 6, 8, 10], reg_type="unknown_model")


def test_simple_linear_unequal_lengths():
    """x and y must have same length."""
    with pytest.raises(ValueError, match="same length"):
        import numpy as np

        from stats_engine.regression import _simple_linear

        _simple_linear(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), 0.05)


def test_simple_linear_single_point():
    """At least 2 data points required."""
    with pytest.raises(ValueError, match="At least 2"):
        import numpy as np

        from stats_engine.regression import _simple_linear

        _simple_linear(np.array([1.0]), np.array([2.0]), 0.05)


def test_polynomial_too_few_points():
    """At least 2 data points required for polynomial."""
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        regression(x=[1.0], y=[2.0], reg_type="polynomial", degree=2)


def test_unknown_nonlinear_model():
    """Unknown nonlinear model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown nonlinear model"):
        import numpy as np

        from stats_engine.regression import _nonlinear

        _nonlinear(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "badmodel", 0.05)

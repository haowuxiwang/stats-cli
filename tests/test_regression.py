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

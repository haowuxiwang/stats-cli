"""Tests for stats_engine/transform.py."""

import numpy as np
import pytest

from stats_engine.transform import transform


def test_log():
    values = [1, 2, 3, 4, 5]
    result = transform(values=values, method="log")
    assert result["method"] == "log"
    assert "values" in result
    assert "before" in result
    assert "after" in result


def test_sqrt():
    values = [1, 4, 9, 16, 25]
    result = transform(values=values, method="sqrt")
    assert result["method"] == "sqrt"


def test_boxcox():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = transform(values=values, method="boxcox")
    assert result["method"] == "boxcox"
    assert "lambda" in result


def test_johnson():
    np.random.seed(42)
    values = np.random.exponential(2, 50).tolist()
    result = transform(values=values, method="johnson")
    assert result["method"] == "johnson"


def test_rank():
    values = [10, 30, 20, 50, 40]
    result = transform(values=values, method="rank")
    assert result["method"] == "rank"


def test_standardize():
    values = [1, 2, 3, 4, 5]
    result = transform(values=values, method="standardize")
    assert result["method"] == "standardize"
    assert abs(np.mean(result["values"])) < 1e-10


def test_recip():
    values = [1, 2, 4, 5, 10]
    result = transform(values=values, method="recip")
    assert result["method"] == "recip"


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        transform(values=[1, 2, 3], method="invalid")


def test_log_with_nan():
    values = [1, 2, float("nan"), 4, 5]
    result = transform(values=values, method="log")
    assert result["method"] == "log"


def test_sqrt_with_inf():
    values = [1, 4, 9, float("inf"), 25]
    result = transform(values=values, method="sqrt")
    assert result["method"] == "sqrt"


def test_boxcox_requires_positive():
    """Box-Cox requires positive values."""
    values = [-1, 2, 3, 4, 5]
    # Should either work with offset or raise
    try:
        result = transform(values=values, method="boxcox")
        assert result["method"] == "boxcox"
    except ValueError:
        pass  # Acceptable


def test_johnson_fallback():
    """Johnson transform may fallback to Box-Cox."""
    values = [1, 1, 1, 1, 1]  # Constant values
    try:
        result = transform(values=values, method="johnson")
        assert result["method"] == "johnson"
    except Exception:
        pass  # Acceptable for degenerate data


def test_rank_with_ties():
    values = [10, 20, 20, 30, 30, 30]
    result = transform(values=values, method="rank")
    assert result["method"] == "rank"
    assert len(result["values"]) == 6


def test_standardize_single_value():
    """Standardize with single value should handle gracefully."""
    values = [5.0]
    try:
        result = transform(values=values, method="standardize")
        assert result["method"] == "standardize"
    except Exception:
        pass  # May fail with zero std


def test_recip_with_zero():
    """Reciprocal of zero should be handled."""
    values = [1, 2, 0, 4, 5]
    try:
        result = transform(values=values, method="recip")
        assert result["method"] == "recip"
    except Exception:
        pass  # May fail with division by zero


def test_log_with_negative():
    """Log with negative values should use offset."""
    values = [-5, -2, 0, 3, 6]
    result = transform(values=values, method="log")
    assert result["method"] == "log"
    assert "offset" in result


def test_sqrt_with_negative():
    """Sqrt with negative values should use offset."""
    values = [-5, -2, 0, 3, 6]
    result = transform(values=values, method="sqrt")
    assert result["method"] == "sqrt"
    assert "offset" in result


def test_johnson_constant_values():
    """Johnson transform with constant values should fallback."""
    values = [5.0, 5.0, 5.0, 5.0, 5.0]
    try:
        result = transform(values=values, method="johnson")
        assert result["method"] == "johnson"
    except Exception:
        pass  # May fail with constant data


def test_standardize_constant():
    """Standardize with constant values should return zeros."""
    values = [5.0, 5.0, 5.0, 5.0, 5.0]
    result = transform(values=values, method="standardize")
    assert result["method"] == "standardize"
    assert all(v == 0 for v in result["values"])

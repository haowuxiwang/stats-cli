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

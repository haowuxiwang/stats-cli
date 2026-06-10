"""Tests for stats_engine/trend.py."""

import numpy as np
import pytest

from stats_engine.trend import trend


def test_cusum(sample_values):
    result = trend(values=sample_values, test_type="cusum")
    assert result["test_type"] == "cusum"
    assert "alarm_points" in result
    assert "stable" in result
    assert "interpretation" in result


def test_ewma(sample_values):
    result = trend(values=sample_values, test_type="ewma")
    assert result["test_type"] == "ewma"
    assert "alarm_points" in result


def test_runs(sample_values):
    result = trend(values=sample_values, test_type="runs")
    assert result["test_type"] == "runs"
    assert "p_value" in result


def test_cusum_with_target():
    np.random.seed(42)
    values = np.random.normal(100, 2, 50).tolist()
    result = trend(values=values, test_type="cusum", target=100)
    assert "alarm_points" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown"):
        trend(values=[1, 2, 3], test_type="invalid")

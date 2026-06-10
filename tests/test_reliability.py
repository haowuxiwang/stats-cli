"""Tests for stats_engine/reliability.py."""

import numpy as np
import pytest

from stats_engine.reliability import reliability


def test_weibull():
    times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    status = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["analysis_type"] == "weibull"
    assert "beta" in result
    assert "eta" in result
    assert "b_lives" in result
    assert "mttf" in result


def test_kaplan_meier():
    times = [100, 200, 300, 400, 500]
    status = [1, 0, 1, 1, 0]
    result = reliability(analysis_type="kaplan_meier", times=times, status=status)
    assert result["analysis_type"] == "kaplan_meier"
    assert "median_survival" in result
    assert "km_table" in result


def test_distribution():
    np.random.seed(42)
    values = np.random.weibull(2, 50) * 100
    result = reliability(analysis_type="distribution", values=values.tolist())
    assert result["analysis_type"] == "distribution_fit"
    assert "best_fit" in result


def test_stability():
    times = [0, 3, 6, 9, 12]
    values = [100, 99.5, 99.0, 98.5, 98.0]
    result = reliability(analysis_type="stability", times=times, values=values)
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        reliability(analysis_type="invalid", times=[1, 2], status=[1, 1])


def test_weibull_shape_positive():
    """Weibull shape parameter (beta) should be positive."""
    times = [100, 200, 300, 400, 500]
    status = [1, 1, 1, 1, 1]
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["beta"] > 0, f"beta={result['beta']}"


def test_kaplan_meier_survival_decreasing():
    """Survival probability should be non-increasing."""
    times = [10, 20, 30, 40, 50]
    status = [1, 1, 1, 1, 1]
    result = reliability(analysis_type="kaplan_meier", times=times, status=status)
    survival = [row["survival"] for row in result["km_table"]]
    for i in range(1, len(survival)):
        assert survival[i] <= survival[i - 1] + 0.001, (
            f"Survival increased at step {i}: {survival[i - 1]} -> {survival[i]}"
        )

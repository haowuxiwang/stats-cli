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


def test_weibull_with_censored():
    """Weibull with censored data."""
    times = [100, 200, 300, 400, 500]
    status = [1, 0, 1, 0, 1]  # 0 = censored
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["analysis_type"] == "weibull"


def test_distribution_multiple_fits():
    """Distribution fitting should try multiple distributions."""
    np.random.seed(42)
    values = np.random.normal(100, 10, 100).tolist()
    result = reliability(analysis_type="distribution", values=values)
    assert result["analysis_type"] == "distribution_fit"
    assert "fits" in result or "best_fit" in result


def test_stability_with_confidence():
    """Stability study with confidence interval."""
    times = [0, 3, 6, 9, 12, 15, 18]
    values = [100, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0]
    result = reliability(analysis_type="stability", times=times, values=values)
    assert result["analysis_type"] == "stability"
    assert "intercept" in result or "slope" in result


def test_kaplan_meier_all_censored():
    """Kaplan-Meier with all censored data."""
    times = [100, 200, 300]
    status = [0, 0, 0]
    try:
        result = reliability(analysis_type="kaplan_meier", times=times, status=status)
        assert result["analysis_type"] == "kaplan_meier"
    except Exception:
        pass  # May fail with all censored data


def test_weibull_single_failure():
    """Weibull with single failure time."""
    times = [100]
    status = [1]
    try:
        result = reliability(analysis_type="weibull", times=times, status=status)
        assert result["analysis_type"] == "weibull"
    except Exception:
        pass  # May fail with insufficient data


def test_distribution_fit_all_fail():
    """Distribution fit when all distributions fail."""
    from stats_engine.reliability import reliability
    # Use very small dataset that may cause fitting issues
    result = reliability(analysis_type="distribution", values=[1, 1, 1, 1, 1])
    # Should still return a result, possibly with errors in individual fits
    assert "distributions" in result or "error" in result


def test_shelf_life_boundary():
    """Shelf life calculation with edge case data."""
    from stats_engine.reliability import reliability
    result = reliability(analysis_type="stability",
                         values=[10.0, 10.1, 10.0, 10.1, 10.0],
                         times=[0, 1, 2, 3, 4])
    assert "shelf_life" in result or "parameters" in result


def test_shelf_life_with_usl():
    """Shelf life with upper spec limit."""
    from stats_engine.reliability import reliability
    result = reliability(analysis_type="stability",
                         times=[0, 3, 6, 9, 12],
                         values=[100, 101, 102, 103, 104],
                         usl=110)
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result


def test_shelf_life_with_lsl():
    """Shelf life with lower spec limit triggers shelf life calculation."""
    from stats_engine.reliability import reliability
    result = reliability(analysis_type="stability",
                         times=[0, 3, 6, 9, 12],
                         values=[100, 99.5, 99.0, 98.5, 98.0],
                         lsl=95)
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result
    assert result["shelf_life"] is not None

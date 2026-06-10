"""Tests for stats_engine/ttest.py."""

import numpy as np
import pytest

from stats_engine.ttest import ttest


def test_one_sample():
    np.random.seed(42)
    values = np.random.normal(10, 1, 30).tolist()
    result = ttest(test_type="one_sample", values=values, mu=10)
    assert result["test_type"] == "one_sample"
    assert "t_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert "ci_95" in result


def test_two_sample(two_groups):
    g1, g2 = two_groups
    result = ttest(test_type="two_sample", values=g1, values2=g2)
    assert result["test_type"] == "two_sample"
    assert "t_statistic" in result
    assert "p_value" in result
    assert "significant" in result


def test_paired():
    np.random.seed(42)
    before = np.random.normal(10, 1, 20).tolist()
    after = (np.array(before) + np.random.normal(0.5, 0.3, 20)).tolist()
    result = ttest(test_type="paired", values=before, values2=after)
    assert result["test_type"] == "paired"
    assert "p_value" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown test_type"):
        ttest(test_type="invalid", values=[1, 2, 3])


def test_interpretation(two_groups):
    g1, g2 = two_groups
    result = ttest(test_type="two_sample", values=g1, values2=g2)
    assert "interpretation" in result


def test_one_sample_t_statistic_range():
    """t-statistic should be reasonable for data near mu."""
    np.random.seed(42)
    values = np.random.normal(10, 1, 30).tolist()
    result = ttest(test_type="one_sample", values=values, mu=10)
    assert -3 < result["t_statistic"] < 3, f"t={result['t_statistic']}"
    assert 0 < result["p_value"] <= 1


def test_two_sample_different_means():
    """Large mean difference should give significant result."""
    np.random.seed(42)
    g1 = np.random.normal(100, 10, 50).tolist()
    g2 = np.random.normal(130, 10, 50).tolist()
    result = ttest(test_type="two_sample", values=g1, values2=g2)
    assert bool(result["significant"]) is True
    assert result["p_value"] < 0.001


def test_paired_effect_direction():
    """Paired test on shifted data should detect the shift."""
    np.random.seed(42)
    before = np.random.normal(50, 5, 30).tolist()
    after = (np.array(before) + 10).tolist()
    result = ttest(test_type="paired", values=before, values2=after)
    assert result["t_statistic"] < -10  # large negative t for increase

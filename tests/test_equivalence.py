"""Tests for stats_engine/equivalence.py."""

import numpy as np
import pytest

from stats_engine.equivalence import equivalence


def test_tost_two_sample():
    np.random.seed(42)
    x = np.random.normal(10, 1, 30).tolist()
    y = np.random.normal(10.2, 1, 30).tolist()
    result = equivalence(test_type="tost", values=x, values2=y, delta=0.5)
    assert result["test_type"] == "tost"
    assert "p_value" in result
    assert "equivalent" in result
    assert "ci_90" in result
    assert "interpretation" in result


def test_one_sample_tost():
    np.random.seed(42)
    values = np.random.normal(10, 1, 30).tolist()
    result = equivalence(test_type="one_sample_tost", values=values, mu=10, delta=0.5)
    assert result["test_type"] == "one_sample_tost"
    assert "p_value" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown test_type"):
        equivalence(test_type="invalid", values=[1, 2], delta=0.5)


def test_tost_equivalent():
    """TOST with very similar samples should be equivalent."""
    from stats_engine.equivalence import equivalence

    result = equivalence(
        test_type="tost", values=[10.0, 10.1, 10.0, 10.1, 10.0], values2=[10.05, 10.15, 10.05, 10.15, 10.05], delta=0.5
    )
    assert result["equivalent"] is True
    assert "ci_90" in result
    assert len(result["ci_90"]) == 2


def test_tost_not_equivalent():
    """TOST with very different samples should not be equivalent."""
    from stats_engine.equivalence import equivalence

    result = equivalence(test_type="tost", values=[10, 10, 10, 10, 10], values2=[20, 20, 20, 20, 20], delta=0.5)
    assert result["equivalent"] is False


def test_one_sample_tost_fields():
    """One-sample TOST returns all expected fields."""
    from stats_engine.equivalence import equivalence

    result = equivalence(test_type="one_sample_tost", values=[10.0, 10.1, 10.2, 10.0, 10.1], mu=10.0, delta=0.5)
    assert "test_type" in result
    assert "p_value" in result
    assert "equivalent" in result
    assert "ci_90" in result
    assert "delta" in result
    assert "alpha" in result


def test_tost_custom_alpha():
    """TOST with custom alpha."""
    from stats_engine.equivalence import equivalence

    result = equivalence(
        test_type="tost", values=[10, 10.1, 10.2], values2=[10.05, 10.15, 10.25], delta=0.5, alpha=0.10
    )
    assert result["alpha"] == 0.10

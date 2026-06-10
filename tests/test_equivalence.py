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

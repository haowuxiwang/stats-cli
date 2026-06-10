"""Tests for stats_engine/doe.py."""

import pytest

from stats_engine.doe import doe


def test_full_factorial():
    factors = [{"name": "Temp", "levels": 3}, {"name": "Time", "levels": 2}]
    result = doe(doe_type="full_factorial", factors=factors)
    assert result["doe_type"] == "full_factorial"
    assert "design_matrix" in result
    assert "n_runs" in result
    assert result["n_runs"] == 6  # 3 * 2


def test_fractional_factorial():
    factors = [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}, {"name": "C", "levels": 2}]
    result = doe(doe_type="fractional_factorial", factors=factors)
    assert result["doe_type"] == "fractional_factorial"
    assert "design_matrix" in result


def test_response_surface():
    factors = [{"name": "A", "levels": 3}, {"name": "B", "levels": 3}]
    result = doe(doe_type="response_surface", factors=factors)
    assert result["doe_type"] == "response_surface"
    assert "design_matrix" in result


def test_taguchi():
    factors = [{"name": "A", "levels": 3}, {"name": "B", "levels": 3}, {"name": "C", "levels": 3}]
    result = doe(doe_type="taguchi", factors=factors)
    assert result["doe_type"] == "taguchi"
    assert "design_matrix" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown doe_type"):
        doe(doe_type="invalid", factors=[{"name": "A", "levels": 2}])


def test_full_factorial_list_levels():
    """Factors with explicit level values."""
    factors = [{"name": "A", "levels": [10, 20, 30]}, {"name": "B", "levels": [100, 200]}]
    result = doe(doe_type="full_factorial", factors=factors)
    assert result["n_runs"] == 6
    assert result["design_matrix"][0] == {"A": 10, "B": 100}
    assert result["design_matrix"][-1] == {"A": 30, "B": 200}


def test_full_factorial_low_high():
    """Factors with low/high specification."""
    factors = [{"name": "A", "low": 1, "high": 2}, {"name": "B", "low": 10, "high": 30}]
    result = doe(doe_type="full_factorial", factors=factors)
    assert result["n_runs"] == 4
    assert result["design_matrix"][0]["A"] == 1
    assert result["design_matrix"][0]["B"] == 10


def test_taguchi_list_levels():
    """Taguchi with explicit level values."""
    factors = [{"name": "A", "levels": [5, 10, 15]}, {"name": "B", "levels": [100, 200, 300]}]
    result = doe(doe_type="taguchi", factors=factors)
    assert result["doe_type"] == "taguchi"
    assert "design_matrix" in result


def test_fractional_factorial_list_levels():
    """Fractional factorial with list levels."""
    factors = [{"name": "A", "levels": [0, 1]}, {"name": "B", "levels": [0, 1]}, {"name": "C", "levels": [0, 1]}]
    result = doe(doe_type="fractional_factorial", factors=factors)
    assert result["doe_type"] == "fractional_factorial"

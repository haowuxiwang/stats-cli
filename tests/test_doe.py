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


def test_factors_not_list():
    """Factors must be a list."""
    with pytest.raises(ValueError, match="factors must be a list"):
        doe(doe_type="full_factorial", factors="invalid")


def test_empty_factors():
    """Empty factors should raise error."""
    with pytest.raises(ValueError, match="At least one factor"):
        doe(doe_type="full_factorial", factors=[])


def test_factor_levels_too_small():
    """Factor with levels < 2 should raise error."""
    with pytest.raises(ValueError, match="levels.*must be >= 2"):
        doe(doe_type="full_factorial", factors=[{"name": "A", "levels": 1}])


def test_factor_list_levels_too_small():
    """Factor with < 2 level values should raise error."""
    with pytest.raises(ValueError, match="need at least 2 level values"):
        doe(doe_type="full_factorial", factors=[{"name": "A", "levels": [10]}])


def test_factor_invalid_levels_type():
    """Factor with invalid levels type should raise error."""
    with pytest.raises(ValueError, match="levels.*must be int or list"):
        doe(doe_type="full_factorial", factors=[{"name": "A", "levels": "invalid"}])


def test_factor_no_levels():
    """Factor without levels or low/high should raise error."""
    with pytest.raises(ValueError, match="must have.*levels"):
        doe(doe_type="full_factorial", factors=[{"name": "A"}])


def test_full_factorial_3_factors():
    """Full factorial with 3 factors."""
    factors = [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}, {"name": "C", "levels": 2}]
    result = doe(doe_type="full_factorial", factors=factors)
    assert result["n_runs"] == 8


def test_response_surface_with_responses():
    """Response surface with response data."""
    factors = [{"name": "A", "levels": 3}, {"name": "B", "levels": 3}]
    # Generate design first to get correct number of runs
    design_result = doe(doe_type="response_surface", factors=factors)
    n_runs = design_result["n_runs"]
    # Create responses matching design matrix size
    responses = list(range(1, n_runs + 1))
    result = doe(doe_type="response_surface", factors=factors, responses=responses)
    assert result["doe_type"] == "response_surface"
    assert "design_matrix" in result


def test_taguchi_with_array():
    """Taguchi with specified orthogonal array."""
    factors = [{"name": "A", "levels": 3}, {"name": "B", "levels": 3}]
    result = doe(doe_type="taguchi", factors=factors, orthogonal_array="L9")
    assert result["doe_type"] == "taguchi"


def test_fractional_factorial_with_resolution():
    """Fractional factorial with specified resolution."""
    factors = [
        {"name": "A", "levels": 2},
        {"name": "B", "levels": 2},
        {"name": "C", "levels": 2},
        {"name": "D", "levels": 2},
    ]
    result = doe(doe_type="fractional_factorial", factors=factors, resolution=3)
    assert result["doe_type"] == "fractional_factorial"

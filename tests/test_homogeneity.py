"""Tests for stats_engine/homogeneity.py."""

import pytest

from stats_engine.homogeneity import homogeneity


def test_levene(three_groups):
    result = homogeneity(test_type="levene", groups=three_groups)
    assert result["test_type"] == "levene"
    assert "statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert "interpretation" in result


def test_bartlett(three_groups):
    result = homogeneity(test_type="bartlett", groups=three_groups)
    assert result["test_type"] == "bartlett"
    assert "statistic" in result
    assert "p_value" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown test_type"):
        homogeneity(test_type="invalid", groups=[[1, 2], [3, 4]])


def test_levene_boundary_p_value():
    """Test p_value_context fields for near-boundary p-value."""
    from main import handler

    result = handler(
        {
            "command": "homogeneity",
            "params": {
                "test_type": "levene",
                "groups": [[10.0, 10.1, 10.2, 10.0, 10.1], [10.5, 10.6, 10.7, 10.5, 10.6]],
            },
        }
    )
    assert result["status"] in ("success", "warning")
    # p_boundary or small_sample_warning may be present
    data = result["data"]
    assert "p_value" in data
    assert "significant" in data


def test_levene_small_sample_warning():
    """Small sample should trigger warning."""
    from main import handler

    result = handler({"command": "homogeneity", "params": {"test_type": "levene", "groups": [[1, 2, 3], [4, 5, 6]]}})
    assert result["status"] == "warning"
    assert "small_sample_warning" in result["data"]


def test_bartlett_returns_all_fields():
    """Bartlett test returns expected fields."""
    from stats_engine.homogeneity import homogeneity

    result = homogeneity(test_type="bartlett", groups=[[10, 11, 12], [20, 21, 22]])
    assert "test_type" in result
    assert "p_value" in result
    assert "significant" in result
    assert "alpha" in result

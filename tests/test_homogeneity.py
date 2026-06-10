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

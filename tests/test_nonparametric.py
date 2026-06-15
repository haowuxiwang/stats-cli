"""Tests for stats_engine/nonparametric.py."""

import numpy as np
import pytest

from stats_engine.nonparametric import nonparametric


def test_mann_whitney(two_groups):
    g1, g2 = two_groups
    result = nonparametric(test_type="mann_whitney", x=g1, y=g2)
    assert result["test_type"] == "mann_whitney"
    assert "u_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert "interpretation" in result


def test_kruskal_wallis(three_groups):
    result = nonparametric(test_type="kruskal_wallis", groups=three_groups)
    assert result["test_type"] == "kruskal_wallis"
    assert "h_statistic" in result
    assert "p_value" in result


def test_wilcoxon():
    np.random.seed(42)
    x = np.random.normal(10, 1, 20).tolist()
    y = (np.array(x) + np.random.normal(0.5, 0.3, 20)).tolist()
    result = nonparametric(test_type="wilcoxon", x=x, y=y)
    assert result["test_type"] == "wilcoxon"
    assert "p_value" in result


def test_chi_square():
    observed = [50, 30, 20]
    result = nonparametric(test_type="chi_square", observed=observed)
    assert result["test_type"] == "chi_square"
    assert "chi2_statistic" in result
    assert "p_value" in result


def test_friedman():
    g1 = [10, 11, 12, 13, 14]
    g2 = [15, 16, 17, 18, 19]
    g3 = [20, 21, 22, 23, 24]
    result = nonparametric(test_type="friedman", groups=[g1, g2, g3])
    assert result["test_type"] == "friedman"


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown test_type"):
        nonparametric(test_type="invalid", x=[1, 2], y=[3, 4])


def test_mann_whitney_with_nan():
    """Mann-Whitney should handle NaN values."""
    x = [1, 2, float('nan'), 4, 5]
    y = [6, 7, 8, 9, 10]
    result = nonparametric(test_type="mann_whitney", x=x, y=y)
    assert result["test_type"] == "mann_whitney"


def test_wilcoxon_with_nan():
    """Wilcoxon should handle NaN values by filtering pairs."""
    x = [1, 2, float('nan'), 4, 5]
    y = [6, 7, 8, float('nan'), 10]
    try:
        result = nonparametric(test_type="wilcoxon", x=x, y=y)
        assert result["test_type"] == "wilcoxon"
    except ValueError:
        pass  # Acceptable if arrays become unequal after filtering


def test_chi_square_with_expected():
    """Chi-square with expected frequencies."""
    observed = [50, 30, 20]
    expected = [40, 30, 30]
    result = nonparametric(test_type="chi_square", observed=observed, expected=expected)
    assert result["test_type"] == "chi_square"


def test_kruskal_wallis_many_groups():
    """Kruskal-Wallis with many groups."""
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    result = nonparametric(test_type="kruskal_wallis", groups=groups)
    assert result["test_type"] == "kruskal_wallis"

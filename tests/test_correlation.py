"""Tests for stats_engine/correlation.py."""

from stats_engine.correlation import correlation


def test_pearson(xy_data):
    x, y = xy_data
    result = correlation(x=x, y=y, method="pearson")
    assert result["method"] == "pearson"
    assert "correlation" in result
    assert "p_value" in result
    assert "r_squared" in result
    assert "interpretation" in result
    assert -1 <= result["correlation"] <= 1


def test_spearman(xy_data):
    x, y = xy_data
    result = correlation(x=x, y=y, method="spearman")
    assert result["method"] == "spearman"
    assert "correlation" in result


def test_kendall(xy_data):
    x, y = xy_data
    result = correlation(x=x, y=y, method="kendall")
    assert result["method"] == "kendall"
    assert "correlation" in result


def test_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    result = correlation(x=x, y=y, method="pearson")
    assert result["correlation"] > 0.99


def test_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    result = correlation(x=x, y=y, method="pearson")
    assert result["correlation"] < -0.99

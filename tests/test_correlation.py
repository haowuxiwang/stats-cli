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


def test_unequal_lengths():
    """Correlation with unequal lengths should raise error."""
    x = [1, 2, 3]
    y = [1, 2, 3, 4]
    try:
        correlation(x=x, y=y)
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_insufficient_data():
    """Correlation with insufficient data should raise error."""
    x = [float("nan"), float("nan")]
    y = [float("nan"), float("nan")]
    try:
        correlation(x=x, y=y)
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_unknown_method():
    """Correlation with unknown method should raise error."""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    try:
        correlation(x=x, y=y, method="invalid")
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_moderate_correlation():
    """Test moderate correlation strength."""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 4, 5, 4, 5, 6, 7, 8, 9, 10]
    result = correlation(x=x, y=y, method="pearson")
    assert result["strength"] in ["moderate", "strong", "very strong"]


def test_weak_correlation():
    """Test weak correlation strength."""
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 3, 8, 2, 7, 4, 9, 1, 6, 10]
    result = correlation(x=x, y=y, method="pearson")
    assert result["strength"] in ["weak", "very weak", "moderate"]

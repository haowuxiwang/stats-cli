"""Tests for stats_engine/descriptive.py."""


from stats_engine.descriptive import descriptive


def test_basic(sample_values):
    result = descriptive(values=sample_values)
    assert "mean" in result
    assert "median" in result
    assert "std" in result
    assert "rsd_percent" in result
    assert "min" in result
    assert "max" in result
    assert "range" in result
    assert "q1" in result
    assert "q3" in result
    assert "iqr" in result
    assert "ci_95_lower" in result
    assert "ci_95_upper" in result
    assert "skewness" in result
    assert "kurtosis" in result
    assert "interpretation" in result


def test_values_accuracy():
    result = descriptive(values=[1, 2, 3, 4, 5])
    assert result["n"] == 5
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["range"] == 4.0


def test_single_value():
    result = descriptive(values=[5.0])
    assert result["n"] == 1
    assert result["mean"] == 5.0
    assert result["std"] is None
    assert result["insufficient_for_stats"] is True


def test_ci_contains_mean(sample_values):
    result = descriptive(values=sample_values)
    assert result["ci_95_lower"] <= result["mean"] <= result["ci_95_upper"]

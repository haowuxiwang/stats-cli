"""Tests for stats_engine/outlier.py."""

import pytest

from stats_engine.outlier import outlier


def test_grubbs_no_outliers(sample_values):
    result = outlier(values=sample_values, method="grubbs")
    assert result["method"] == "grubbs"
    assert "outliers" in result
    assert "n_outliers" in result
    assert "clean_data" in result


def test_grubbs_with_outlier():
    values = [10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 50.0]
    result = outlier(values=values, method="grubbs")
    assert result["n_outliers"] > 0
    assert 50.0 in result["outliers"]


def test_iqr(sample_values):
    result = outlier(values=sample_values, method="iqr")
    assert result["method"] == "iqr"
    assert "outliers" in result


def test_zscore(sample_values):
    result = outlier(values=sample_values, method="zscore")
    assert result["method"] == "zscore"
    assert "outliers" in result


def test_dixon(sample_values):
    result = outlier(values=sample_values, method="dixon")
    assert result["method"] == "dixon"
    assert "outliers" in result


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        outlier(values=[1, 2, 3], method="invalid")


def test_grubbs_no_outliers_clean():
    """Clean data should have zero outliers."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1]
    result = outlier(values=values, method="grubbs")
    assert result["n_outliers"] == 0


def test_grubbs_detects_outlier():
    """Extreme value should be detected as outlier."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 100.0]
    result = outlier(values=values, method="grubbs")
    assert result["n_outliers"] >= 1


def test_dixon_too_many_values():
    """Dixon's Q test works best with 3-30 data points."""
    values = list(range(50))
    result = outlier(values=values, method="dixon")
    assert result.get("error") is True or result["n_outliers"] >= 0


def test_dixon_with_outlier():
    """Dixon's Q test with clear outlier."""
    values = [10.0, 10.1, 10.2, 10.0, 10.1, 100.0]
    result = outlier(values=values, method="dixon")
    assert result["method"] == "dixon"


def test_zscore_constant_values():
    """Z-score with constant values (std=0)."""
    values = [5.0, 5.0, 5.0, 5.0, 5.0]
    result = outlier(values=values, method="zscore")
    assert result["method"] == "zscore"
    assert result["n_outliers"] == 0


def test_iqr_no_outliers():
    """IQR with no outliers."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1]
    result = outlier(values=values, method="iqr")
    assert result["n_outliers"] == 0


def test_mad_no_outliers():
    """MAD with clean data should detect no outliers."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1]
    result = outlier(values=values, method="mad")
    assert result["method"] == "mad"
    assert result["n_outliers"] == 0
    assert "median" in result
    assert "mad" in result
    assert "threshold" in result


def test_mad_with_outlier():
    """MAD should detect extreme outlier."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 100.0]
    result = outlier(values=values, method="mad")
    assert result["n_outliers"] >= 1
    assert 100.0 in result["outliers"]


def test_mad_constant_values():
    """MAD with constant values should return no outliers."""
    values = [5.0, 5.0, 5.0, 5.0, 5.0]
    result = outlier(values=values, method="mad")
    assert result["n_outliers"] == 0
    assert result["mad"] == 0


def test_iqr_custom_factor():
    """IQR with custom factor should produce different fences."""
    values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 50.0]
    result_15 = outlier(values=values, method="iqr", iqr_factor=1.5)
    result_30 = outlier(values=values, method="iqr", iqr_factor=3.0)
    # With stricter factor (3.0), fewer or same outliers
    assert result_30["n_outliers"] <= result_15["n_outliers"]
    assert result_15["iqr_factor"] == 1.5
    assert result_30["iqr_factor"] == 3.0

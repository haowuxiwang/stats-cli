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

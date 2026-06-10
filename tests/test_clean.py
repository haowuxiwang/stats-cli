"""Tests for stats_engine/clean.py."""

import pytest

from stats_engine.clean import clean


def test_drop():
    values = [1, 2, float("nan"), 4, 5]
    result = clean(values=values, method="drop")
    assert result["method"] == "drop"
    assert result["n_clean"] == 4
    assert "before" in result
    assert "after" in result


def test_impute_mean():
    values = [1, 2, float("nan"), 4, 5]
    result = clean(values=values, method="impute_mean")
    assert result["method"] == "impute_mean"
    assert result["n_clean"] == 5


def test_impute_median():
    values = [1, 2, float("nan"), 4, 5]
    result = clean(values=values, method="impute_median")
    assert result["method"] == "impute_median"
    assert result["n_clean"] == 5


def test_winsorize():
    values = [1, 2, 3, 4, 5, 100]
    result = clean(values=values, method="winsorize")
    assert result["method"] == "winsorize"


def test_clip():
    values = [1, 2, 3, 4, 5, 100]
    result = clean(values=values, method="clip")
    assert result["method"] == "clip"


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        clean(values=[1, 2, 3], method="invalid")

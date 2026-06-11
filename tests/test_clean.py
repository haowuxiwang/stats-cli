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


def test_impute_mode():
    """Impute with mode value."""
    values = [1, 2, 2, 2, 3, float("nan")]
    result = clean(values=values, method="impute_mode")
    assert result["method"] == "impute_mode"
    assert result["n_clean"] == 6


def test_impute_mode_all_nan():
    """All NaN values should raise error for mode imputation."""
    values = [float("nan"), float("nan"), float("nan")]
    with pytest.raises(ValueError, match="All values are NaN"):
        clean(values=values, method="impute_mode")


def test_winsorize_custom_limits():
    """Winsorize with custom limits."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    result = clean(values=values, method="winsorize", limits=(0.1, 0.1))
    assert result["method"] == "winsorize"


def test_clip_custom_bounds():
    """Clip with custom bounds."""
    values = [1, 2, 3, 4, 5, 100]
    result = clean(values=values, method="clip", lower=2, upper=10)
    assert result["method"] == "clip"
    assert all(2 <= v <= 10 for v in result["values"])

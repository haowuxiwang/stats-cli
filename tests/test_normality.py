"""Tests for stats_engine/normality.py."""

import numpy as np

from stats_engine.normality import normality


def test_normal_data():
    np.random.seed(42)
    values = np.random.normal(10, 1, 100).tolist()
    result = normality(values=values)
    assert "shapiro_wilk" in result
    assert "anderson_darling" in result
    assert "lilliefors" in result
    assert "is_normal" in result
    # np.bool_ is a subclass of bool in newer numpy, but just check truthiness
    assert bool(result["is_normal"]) is True


def test_non_normal_data():
    np.random.seed(42)
    values = np.random.exponential(2, 100).tolist()
    result = normality(values=values)
    assert "is_normal" in result


def test_shapiro_keys():
    np.random.seed(42)
    values = np.random.normal(0, 1, 50).tolist()
    result = normality(values=values)
    assert "statistic" in result["shapiro_wilk"]
    assert "p_value" in result["shapiro_wilk"]


def test_interpretation_present():
    np.random.seed(42)
    values = np.random.normal(0, 1, 50).tolist()
    result = normality(values=values)
    assert "interpretation" in result


def test_non_normal_exponential():
    """Exponential data should fail normality test."""
    np.random.seed(42)
    values = np.random.exponential(1, 200).tolist()
    result = normality(values=values)
    assert bool(result["is_normal"]) is False, "Exponential data incorrectly classified as normal"


def test_normal_gaussian():
    """Large normal sample should pass."""
    np.random.seed(42)
    values = np.random.normal(0, 1, 300).tolist()
    result = normality(values=values)
    assert bool(result["is_normal"]) is True
    assert result["shapiro_wilk"]["p_value"] > 0.05


def test_normality_large_sample():
    """Shapiro-Wilk subsamples when n > 5000."""
    np.random.seed(42)
    values = np.random.normal(10, 2, 6000).tolist()
    result = normality(values=values)
    assert "shapiro_wilk" in result


def test_normality_small_sample_warning():
    """Small sample (n < 10) should include warning (line 25)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normality(values=values)
    assert "n=5" in result.get("_warning", "")

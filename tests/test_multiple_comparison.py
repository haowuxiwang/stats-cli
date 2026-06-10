"""Tests for stats_engine/multiple_comparison.py."""

import pytest

from stats_engine.multiple_comparison import multiple_comparison


def test_tukey(three_groups):
    result = multiple_comparison(test_type="tukey", groups=three_groups)
    assert result["test_type"] == "tukey"
    assert "comparisons" in result
    assert "significant_pairs" in result
    assert "interpretation" in result


def test_bonferroni(three_groups):
    result = multiple_comparison(test_type="bonferroni", groups=three_groups)
    assert result["test_type"] == "bonferroni"
    assert "comparisons" in result


def test_scheffe(three_groups):
    result = multiple_comparison(test_type="scheffe", groups=three_groups)
    assert result["test_type"] == "scheffe"
    assert "comparisons" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown test_type"):
        multiple_comparison(test_type="invalid", groups=[[1, 2], [3, 4]])

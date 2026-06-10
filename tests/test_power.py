"""Tests for stats_engine/power.py."""

import pytest

from stats_engine.power import power


def test_t_test():
    result = power(analysis_type="t_test", effect_size=0.5, power_val=0.80)
    assert result["analysis_type"] == "t_test"
    assert "sample_size" in result
    assert "power" in result
    assert "interpretation" in result


def test_anova():
    result = power(analysis_type="anova", effect_size=0.25, power_val=0.80)
    assert result["analysis_type"] == "anova"
    assert "sample_size" in result


def test_proportion():
    result = power(analysis_type="proportion", effect_size=0.1, power_val=0.80)
    assert result["analysis_type"] == "proportion"


def test_correlation():
    result = power(analysis_type="correlation", effect_size=0.5, power_val=0.80)
    assert result["analysis_type"] == "correlation"


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        power(analysis_type="invalid", effect_size=0.5, power_val=0.80)

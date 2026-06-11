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


def test_effect_size_negative():
    """Negative effect size should raise error."""
    with pytest.raises(ValueError, match="effect_size must be non-negative"):
        power(analysis_type="t_test", effect_size=-0.5, n=30)


def test_effect_size_zero():
    """Zero effect size should return power = alpha."""
    result = power(analysis_type="t_test", effect_size=0, n=30)
    assert result["effect_size"] == 0
    assert result["power"] == 0.05
    assert "warning" in result


def test_t_test_power_with_n():
    """Calculate power given n."""
    result = power(analysis_type="t_test", effect_size=0.5, n=34)
    assert result["analysis_type"] == "t_test"
    assert "power" in result
    assert 0 < result["power"] < 1


def test_anova_power_with_n():
    """Calculate ANOVA power given n."""
    result = power(analysis_type="anova", effect_size=0.25, n=100, k_groups=3)
    assert result["analysis_type"] == "anova"
    assert "power" in result


def test_chi_square_power():
    """Chi-square power analysis."""
    result = power(analysis_type="chi_square", effect_size=0.3, n=100, df=1)
    assert result["analysis_type"] == "chi_square"
    assert "power" in result


def test_chi_square_sample_size():
    """Chi-square sample size calculation."""
    result = power(analysis_type="chi_square", effect_size=0.3, power_val=0.80, df=1)
    assert result["analysis_type"] == "chi_square"
    assert "sample_size" in result


def test_correlation_power_with_n():
    """Correlation power given n."""
    result = power(analysis_type="correlation", effect_size=0.5, n=30)
    assert result["analysis_type"] == "correlation"
    assert "power" in result


def test_proportion_power_with_n():
    """Proportion power given n."""
    result = power(analysis_type="proportion", effect_size=0.2, n=100)
    assert result["analysis_type"] == "proportion"
    assert "power" in result


def test_t_test_both_n_and_power():
    """Both n and power provided should raise error."""
    with pytest.raises(ValueError, match="Provide either"):
        power(analysis_type="t_test", effect_size=0.5, n=30, power_val=0.80)


def test_anova_both_n_and_power():
    """Both n and power provided should raise error."""
    with pytest.raises(ValueError, match="Provide either"):
        power(analysis_type="anova", effect_size=0.25, n=100, power_val=0.80)


def test_chi_square_both_n_and_power():
    """Both n and power provided should raise error."""
    with pytest.raises(ValueError, match="Provide either"):
        power(analysis_type="chi_square", effect_size=0.3, n=100, power_val=0.80)


def test_proportion_both_n_and_power():
    """Both n and power provided should raise error."""
    with pytest.raises(ValueError, match="Provide either"):
        power(analysis_type="proportion", effect_size=0.2, n=100, power_val=0.80)


def test_correlation_both_n_and_power():
    """Both n and power provided should raise error."""
    with pytest.raises(ValueError, match="Provide either"):
        power(analysis_type="correlation", effect_size=0.5, n=30, power_val=0.80)

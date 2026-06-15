"""Tests for stats_engine/advanced.py."""

import numpy as np
import pytest

from stats_engine.advanced import advanced


def test_exact_test():
    observed = [[10, 5], [3, 12]]
    result = advanced(analysis_type="exact_test", observed=observed)
    assert result["analysis_type"] == "exact_test"
    assert "odds_ratio" in result
    assert "p_value" in result
    assert "significant" in result
    assert "interpretation" in result


def test_exact_test_symmetric():
    observed = [[5, 5], [5, 5]]
    result = advanced(analysis_type="exact_test", observed=observed)
    assert result["odds_ratio"] == 1.0


def test_mcnemar():
    observed = [[10, 5], [15, 20]]
    result = advanced(analysis_type="mcnemar", observed=observed)
    assert result["analysis_type"] == "mcnemar"
    assert "p_value" in result
    assert "discordant_pairs" in result
    assert result["discordant_pairs"] == 20  # 5 + 15


def test_mcnemar_no_discordant():
    observed = [[10, 0], [0, 20]]
    result = advanced(analysis_type="mcnemar", observed=observed)
    assert result["p_value"] == 1.0


def test_cochran_q():
    np.random.seed(42)
    data = [
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    ]
    result = advanced(analysis_type="cochran_q", data=data)
    assert result["analysis_type"] == "cochran_q"
    assert "q_statistic" in result
    assert "p_value" in result
    assert result["n_treatments"] == 3
    assert result["n_subjects"] == 10


def test_mixed_effects():
    np.random.seed(42)
    groups = []
    group_ids = []
    for i in range(5):
        n = 10
        vals = np.random.normal(10 + i * 0.5, 1, n).tolist()
        groups.extend(vals)
        group_ids.extend([f"G{i}"] * n)
    result = advanced(analysis_type="mixed_effects", groups=groups, group_ids=group_ids)
    assert result["analysis_type"] == "mixed_effects"
    assert "fixed_effects" in result
    assert "variance_components" in result
    assert "icc" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        advanced(analysis_type="invalid")


def test_exact_test_bad_shape():
    with pytest.raises(ValueError, match="2x2"):
        advanced(analysis_type="exact_test", observed=[[1, 2, 3], [4, 5, 6]])


def test_mcnemar_bad_shape():
    """McNemar requires 2x2 table."""
    with pytest.raises(ValueError, match="2x2"):
        advanced(analysis_type="mcnemar", observed=[[1, 2, 3], [4, 5, 6]])


def test_mcnemar_chi_squared():
    """McNemar with many discordant pairs uses chi-squared."""
    observed = [[10, 50], [60, 20]]
    result = advanced(analysis_type="mcnemar", observed=observed)
    assert result["analysis_type"] == "mcnemar"
    assert result["discordant_pairs"] == 110
    assert result["chi2"] is not None


def test_cochran_q_bad_shape():
    """Cochran's Q requires 2D matrix."""
    with pytest.raises(ValueError, match="2D"):
        advanced(analysis_type="cochran_q", data=[1, 2, 3])


def test_cochran_q_zero_denominator():
    """Cochran's Q with zero denominator."""
    # All zeros or all ones
    data = [[0, 0, 0], [0, 0, 0]]
    result = advanced(analysis_type="cochran_q", data=data)
    assert result["q_statistic"] == 0.0
    assert result["p_value"] == 1.0

"""Tests for stats_engine/acceptance_sampling.py."""

import json

import pytest
from scipy.stats import binom

from stats_engine.acceptance_sampling import acceptance_sampling

# ---------------------------------------------------------------------------
# Single plan
# ---------------------------------------------------------------------------


def test_single_plan_basic():
    """Single plan with known binomial probability."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.05)
    expected = binom.cdf(2, 50, 0.05)
    assert result["analysis_type"] == "single_plan"
    assert abs(result["accept_prob"] - expected) < 1e-4
    assert abs(result["reject_prob"] - (1 - expected)) < 1e-4
    assert result["model"] == "binomial"


def test_single_plan_zero_defect():
    """Single plan with c=0 (zero-acceptance plan)."""
    result = acceptance_sampling("single_plan", n=20, c=0, defect_rate=0.01)
    expected = binom.cdf(0, 20, 0.01)
    assert abs(result["accept_prob"] - expected) < 1e-4
    assert result["reject_prob"] > 0


def test_single_plan_aoq_and_ati():
    """AOQ and ATI are computed for binomial model."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.05)
    pa = result["accept_prob"]
    # AOQ = Pa * p for infinite lot
    assert abs(result["aoq"] - pa * 0.05) < 1e-4
    # ATI = n for infinite lot
    assert result["ati"] == 50


def test_single_plan_hypergeometric():
    """Single plan with finite lot uses hypergeometric model."""
    result = acceptance_sampling("single_plan", n=20, c=1, defect_rate=0.05, lot_size=500)
    assert result["model"] == "hypergeometric"
    assert result["lot_size"] == 500
    assert 0 < result["accept_prob"] < 1
    # ATI should be between n and lot_size
    assert 20 <= result["ati"] <= 500


def test_single_plan_oc_curve_length():
    """OC curve has 50 data points by default."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.05)
    assert len(result["oc_curve"]) == 50


def test_single_plan_high_defect_rate():
    """High defect rate should give low acceptance probability."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.30)
    assert result["accept_prob"] < 0.01


def test_single_plan_low_defect_rate():
    """Low defect rate should give high acceptance probability."""
    result = acceptance_sampling("single_plan", n=50, c=5, defect_rate=0.01)
    assert result["accept_prob"] > 0.99


# ---------------------------------------------------------------------------
# Double plan
# ---------------------------------------------------------------------------


def test_double_plan_basic():
    """Double plan returns expected fields."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    assert result["analysis_type"] == "double_plan"
    assert "prob_accept_first" in result
    assert "prob_reject_first" in result
    assert "prob_continue" in result
    assert "prob_accept_second" in result
    assert "total_accept_prob" in result
    assert "asn" in result


def test_double_plan_probabilities_sum():
    """Accept + reject + continue probabilities sum to 1."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    total = result["prob_accept_first"] + result["prob_reject_first"] + result["prob_continue"]
    assert abs(total - 1.0) < 1e-6


def test_double_plan_asn_range():
    """ASN is between n1 and n1+n2."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    assert 50 <= result["asn"] <= 100


def test_double_plan_total_accept():
    """Total accept = accept_first + accept_second."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    expected = result["prob_accept_first"] + result["prob_accept_second"]
    assert abs(result["total_accept_prob"] - expected) < 1e-6


def test_double_plan_oc_curve():
    """Double plan generates OC curve with 50 points."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    assert len(result["oc_curve"]) == 50


# ---------------------------------------------------------------------------
# OC curve
# ---------------------------------------------------------------------------


def test_oc_curve_shape():
    """OC curve is monotonically decreasing."""
    result = acceptance_sampling("oc_curve", n=50, c=2)
    probs = [p["accept_prob"] for p in result["oc_curve"]]
    for i in range(1, len(probs)):
        assert probs[i] <= probs[i - 1] + 1e-10, f"OC curve not decreasing at index {i}: {probs[i - 1]} -> {probs[i]}"


def test_oc_curve_endpoints():
    """First point near 1.0, last point near 0.0."""
    result = acceptance_sampling("oc_curve", n=100, c=5)
    probs = [p["accept_prob"] for p in result["oc_curve"]]
    assert probs[0] > 0.99
    assert probs[-1] < 0.01


def test_oc_curve_custom_range():
    """OC curve respects custom defect_rate_range."""
    result = acceptance_sampling("oc_curve", n=50, c=2, defect_rate_range=[0.01, 0.10])
    rates = [p["defect_rate"] for p in result["oc_curve"]]
    assert rates[0] >= 0.01
    assert rates[-1] <= 0.10


def test_oc_curve_hypergeometric():
    """OC curve with lot_size uses hypergeometric model."""
    result = acceptance_sampling("oc_curve", n=20, c=1, lot_size=1000)
    assert result["model"] == "hypergeometric"
    assert len(result["oc_curve"]) == 50


# ---------------------------------------------------------------------------
# Find plan
# ---------------------------------------------------------------------------


def test_find_plan_known_values():
    """Find plan for classic AQL=0.01, LTPD=0.05."""
    result = acceptance_sampling("find_plan", AQL=0.01, LTPD=0.05, alpha=0.05, beta=0.10)
    assert result["analysis_type"] == "find_plan"
    assert result["n"] > 0
    assert result["c"] > 0
    # Verify constraints are met
    assert result["pa_at_aql"] >= 0.95
    assert result["pa_at_ltpd"] <= 0.10


def test_find_plan_stricter():
    """Stricter AQL/LTPD requires larger sample."""
    result_relaxed = acceptance_sampling("find_plan", AQL=0.02, LTPD=0.10, alpha=0.05, beta=0.10)
    result_strict = acceptance_sampling("find_plan", AQL=0.01, LTPD=0.05, alpha=0.05, beta=0.10)
    assert result_strict["n"] > result_relaxed["n"]


def test_find_plan_constraints_satisfied():
    """Returned plan satisfies both producer and consumer risk."""
    for aql, ltpd in [(0.005, 0.03), (0.01, 0.06), (0.02, 0.08)]:
        result = acceptance_sampling("find_plan", AQL=aql, LTPD=ltpd, alpha=0.05, beta=0.10)
        assert result["pa_at_aql"] >= 1 - 0.05 - 1e-6, f"Producer risk violated for AQL={aql}"
        assert result["pa_at_ltpd"] <= 0.10 + 1e-6, f"Consumer risk violated for LTPD={ltpd}"


# ---------------------------------------------------------------------------
# JSON serializability
# ---------------------------------------------------------------------------


def test_single_plan_json_serializable():
    """Single plan result is JSON serializable."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.05)
    json_str = json.dumps(result)
    assert json_str is not None
    parsed = json.loads(json_str)
    assert parsed["analysis_type"] == "single_plan"


def test_double_plan_json_serializable():
    """Double plan result is JSON serializable."""
    result = acceptance_sampling("double_plan", n1=50, c1=1, d1=4, n2=50, c2=4, defect_rate=0.05)
    json_str = json.dumps(result)
    assert json_str is not None


def test_find_plan_json_serializable():
    """Find plan result is JSON serializable."""
    result = acceptance_sampling("find_plan", AQL=0.01, LTPD=0.05, alpha=0.05, beta=0.10)
    json_str = json.dumps(result)
    assert json_str is not None


# ---------------------------------------------------------------------------
# Guard conditions — invalid inputs
# ---------------------------------------------------------------------------


def test_invalid_analysis_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        acceptance_sampling("invalid_type")


def test_single_plan_negative_n():
    with pytest.raises(ValueError, match="positive"):
        acceptance_sampling("single_plan", n=-1, c=0, defect_rate=0.05)


def test_single_plan_c_gte_n():
    with pytest.raises(ValueError, match="must be less than n"):
        acceptance_sampling("single_plan", n=10, c=10, defect_rate=0.05)


def test_single_plan_defect_rate_zero():
    with pytest.raises(ValueError, match="between 0 and 1"):
        acceptance_sampling("single_plan", n=10, c=1, defect_rate=0.0)


def test_single_plan_defect_rate_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        acceptance_sampling("single_plan", n=10, c=1, defect_rate=1.0)


def test_single_plan_lot_size_too_small():
    with pytest.raises(ValueError, match="must be >= n"):
        acceptance_sampling("single_plan", n=100, c=2, defect_rate=0.05, lot_size=50)


def test_double_plan_d1_le_c1():
    with pytest.raises(ValueError, match="must be > c1"):
        acceptance_sampling("double_plan", n1=50, c1=3, d1=3, n2=50, c2=5, defect_rate=0.05)


def test_double_plan_c2_lt_c1():
    with pytest.raises(ValueError, match="must be >= c1"):
        acceptance_sampling("double_plan", n1=50, c1=3, d1=6, n2=50, c2=2, defect_rate=0.05)


def test_find_plan_ltpd_le_aql():
    with pytest.raises(ValueError, match="LTPD.*must be > AQL"):
        acceptance_sampling("find_plan", AQL=0.05, LTPD=0.03, alpha=0.05, beta=0.10)


def test_find_plan_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        acceptance_sampling("find_plan", AQL=0.01, LTPD=0.05, alpha=0.0, beta=0.10)


def test_find_plan_beta_out_of_range():
    with pytest.raises(ValueError, match="beta"):
        acceptance_sampling("find_plan", AQL=0.01, LTPD=0.05, alpha=0.05, beta=1.0)


def test_oc_curve_c_gte_n():
    with pytest.raises(ValueError, match="must be less than n"):
        acceptance_sampling("oc_curve", n=5, c=5)


def test_single_plan_non_int_n():
    with pytest.raises(ValueError, match="positive integer"):
        acceptance_sampling("single_plan", n=5.5, c=1, defect_rate=0.05)


def test_single_plan_interpretation_present():
    """Interpretation string is always present."""
    result = acceptance_sampling("single_plan", n=50, c=2, defect_rate=0.05)
    assert "interpretation" in result
    assert len(result["interpretation"]) > 0

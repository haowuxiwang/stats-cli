"""Tests for p_value_context function in utils/output.py."""

from utils.output import p_value_context


def test_boundary_p_value():
    """p near alpha should set p_boundary and p_boundary_warning."""
    result = {}
    p_value_context(result, p_value=0.048, alpha=0.05, n=100)
    assert result.get("p_boundary") is True
    assert "p_boundary_warning" in result
    assert "0.048" in result["p_boundary_warning"]


def test_boundary_p_value_upper():
    """p just above alpha but within 20% margin."""
    result = {}
    p_value_context(result, p_value=0.055, alpha=0.05, n=100)
    assert result.get("p_boundary") is True
    assert "p_boundary_warning" in result


def test_non_boundary_p_value():
    """p far from alpha should NOT set p_boundary."""
    result = {}
    p_value_context(result, p_value=0.001, alpha=0.05, n=100)
    assert "p_boundary" not in result
    assert "p_boundary_warning" not in result


def test_small_sample_warning():
    """n < 10 should set small_sample_warning."""
    result = {}
    p_value_context(result, p_value=0.5, alpha=0.05, n=5)
    assert "small_sample_warning" in result
    assert "n=5" in result["small_sample_warning"]


def test_normal_sample_no_warning():
    """n >= 10 should NOT set small_sample_warning."""
    result = {}
    p_value_context(result, p_value=0.5, alpha=0.05, n=50)
    assert "small_sample_warning" not in result


def test_warning_marker_set():
    """When warnings exist, _warning should be set."""
    result = {}
    p_value_context(result, p_value=0.5, alpha=0.05, n=5)
    assert "_warning" in result
    assert "n=5" in result["_warning"]


def test_no_warning_marker_when_clean():
    """When no warnings, _warning should NOT be set."""
    result = {}
    p_value_context(result, p_value=0.5, alpha=0.05, n=100)
    assert "_warning" not in result


def test_both_warnings_combined():
    """Both boundary and small sample should combine in _warning."""
    result = {}
    p_value_context(result, p_value=0.048, alpha=0.05, n=5)
    assert "_warning" in result
    assert "0.048" in result["_warning"]
    assert "n=5" in result["_warning"]


def test_none_p_value():
    """None p_value should not crash or set boundary."""
    result = {}
    p_value_context(result, p_value=None, alpha=0.05, n=100)
    assert "p_boundary" not in result


def test_none_n():
    """None n should not crash or set small_sample_warning."""
    result = {}
    p_value_context(result, p_value=0.5, alpha=0.05, n=None)
    assert "small_sample_warning" not in result


def test_returns_result():
    """Function should return the result dict."""
    result = {"existing": "value"}
    ret = p_value_context(result, p_value=0.5, alpha=0.05, n=100)
    assert ret is result
    assert result["existing"] == "value"

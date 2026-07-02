"""Tests for stats_engine/gage_rr.py."""

import numpy as np
import pytest

from stats_engine.gage_rr import gage_rr


@pytest.fixture
def gage_data():
    np.random.seed(42)
    parts = list(range(1, 11)) * 6  # 10 parts, 3 operators, 2 replicates
    operators = (["O1"] * 10 + ["O2"] * 10 + ["O3"] * 10) * 2
    measurements = np.random.normal(100, 2, 60).tolist()
    return measurements, parts, operators


def test_crossed(gage_data):
    measurements, parts, operators = gage_data
    result = gage_rr(analysis_type="crossed", measurements=measurements, parts=parts, operators=operators)
    assert result["analysis_type"] == "crossed"
    assert "variance_components" in result
    assert "contribution" in result
    assert "study_variation" in result
    assert "ndc" in result
    assert "rating" in result


def test_nested(gage_data):
    measurements, parts, operators = gage_data
    result = gage_rr(analysis_type="nested", measurements=measurements, parts=parts, operators=operators)
    assert result["analysis_type"] == "nested"
    assert "variance_components" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        gage_rr(analysis_type="invalid", measurements=[1, 2], parts=[1, 2], operators=["A", "B"])


def test_attribute():
    """Attribute agreement analysis with pass/fail data."""
    # 10 samples, 2 operators, 2 replicates
    reference = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ratings = [
        [[1, 1], [1, 1]],  # Sample 1
        [[1, 1], [1, 0]],  # Sample 2
        [[1, 1], [1, 1]],  # Sample 3
        [[1, 0], [1, 1]],  # Sample 4
        [[1, 1], [1, 1]],  # Sample 5
        [[0, 0], [0, 0]],  # Sample 6
        [[0, 0], [0, 1]],  # Sample 7
        [[0, 0], [0, 0]],  # Sample 8
        [[0, 1], [0, 0]],  # Sample 9
        [[0, 0], [0, 0]],  # Sample 10
    ]
    result = gage_rr(analysis_type="attribute", reference=reference, ratings=ratings)
    assert result["analysis_type"] == "attribute_agreement"
    assert result["n_samples"] == 10
    assert result["n_operators"] == 2
    assert result["n_replicates"] == 2
    assert "overall" in result
    assert "operators" in result
    assert result["overall"]["agreement_pct"] > 0


def test_bias():
    """Bias study with known reference value."""
    np.random.seed(42)
    measurements = np.random.normal(10.5, 0.2, 20).tolist()
    result = gage_rr(analysis_type="bias", measurements=measurements, reference_value=10.0)
    assert result["analysis_type"] == "bias"
    assert result["n"] == 20
    assert result["reference_value"] == 10.0
    assert "bias" in result
    assert "bias_pct" in result
    assert "t_statistic" in result
    assert "p_value" in result
    assert "significant" in result


def test_bias_no_significant():
    """Bias study where bias is not significant."""
    np.random.seed(42)
    measurements = np.random.normal(10.0, 0.5, 30).tolist()
    result = gage_rr(analysis_type="bias", measurements=measurements, reference_value=10.0)
    assert result["analysis_type"] == "bias"
    assert abs(result["bias"]) < 0.5


def test_linearity():
    """Linearity study across reference values."""
    np.random.seed(42)
    reference_values = [10, 20, 30, 40, 50]
    measurements = []
    for ref in reference_values:
        measurements.extend(np.random.normal(ref, 0.5, 5).tolist())
    ref_expanded = []
    for ref in reference_values:
        ref_expanded.extend([ref] * 5)

    result = gage_rr(analysis_type="linearity", reference_values=ref_expanded, measurements=measurements)
    assert result["analysis_type"] == "linearity"
    assert "slope" in result
    assert "intercept" in result
    assert "r_squared" in result
    assert "linearity" in result
    assert abs(result["slope"] - 1.0) < 0.2


def test_stability():
    """Stability study with time series data."""
    np.random.seed(42)
    measurements = np.random.normal(100, 1, 30).tolist()
    time_points = list(range(30))
    result = gage_rr(analysis_type="stability", measurements=measurements, time_points=time_points)
    assert result["analysis_type"] == "stability"
    assert result["n"] == 30
    assert "mean" in result
    assert "std" in result
    assert "ucl" in result
    assert "lcl" in result
    assert "out_of_control" in result
    assert "trend_slope" in result
    assert "trend_p_value" in result


def test_stability_with_tolerance():
    """Stability study with tolerance parameter."""
    np.random.seed(42)
    measurements = np.random.normal(100, 1, 30).tolist()
    result = gage_rr(analysis_type="stability", measurements=measurements, tolerance=10.0)
    assert result["analysis_type"] == "stability"
    assert "pct_tolerance" in result


def test_stability_no_time():
    """Stability study without time points."""
    np.random.seed(42)
    measurements = np.random.normal(100, 1, 30).tolist()
    result = gage_rr(analysis_type="stability", measurements=measurements)
    assert result["analysis_type"] == "stability"
    assert result["trend_slope"] is None
    assert result["trend_p_value"] is None


def test_crossed_no_interaction():
    """Crossed Gage R&R with one replicate (no interaction)."""
    np.random.seed(42)
    # 5 parts, 2 operators, 1 replicate each = 10 measurements
    parts = list(range(1, 6)) * 2
    operators = ["O1"] * 5 + ["O2"] * 5
    measurements = np.random.normal(100, 2, 10).tolist()
    result = gage_rr(analysis_type="crossed", measurements=measurements, parts=parts, operators=operators)
    assert result["analysis_type"] == "crossed"
    assert "variance_components" in result


def test_crossed_with_tolerance():
    """Crossed Gage R&R with tolerance parameter."""
    np.random.seed(42)
    parts = list(range(1, 11)) * 6
    operators = (["O1"] * 10 + ["O2"] * 10 + ["O3"] * 10) * 2
    measurements = np.random.normal(100, 2, 60).tolist()
    result = gage_rr(
        analysis_type="crossed", measurements=measurements, parts=parts, operators=operators, tolerance=10.0
    )
    assert result["analysis_type"] == "crossed"
    assert "pct_tolerance" in result or "study_variation" in result


def test_crossed_2d_format_via_handler():
    """Crossed Gage R&R 2D format works through handler."""
    # 3 parts x 4 operators x 3 replicates (2D)
    np.random.seed(42)
    measurements = [[100 + np.random.normal(0, 1) for _ in range(3)] for _ in range(3)]
    result = gage_rr(
        analysis_type="crossed",
        measurements=measurements,
        parts=[f"P{i + 1}" for i in range(3)],
        operators=["O1", "O2", "O3"],
    )
    assert result["analysis_type"] == "crossed"
    assert "variance_components" in result


def test_linearity_analysis():
    """Linearity analysis returns slope and intercept."""
    reference = [1.0, 2.0, 3.0, 4.0, 5.0] * 3
    measurements = [1.1, 2.0, 3.2, 3.9, 5.1, 1.0, 2.1, 3.1, 4.0, 5.0, 1.2, 1.9, 3.0, 4.1, 5.2]
    result = gage_rr(analysis_type="linearity", measurements=measurements, reference_values=reference)
    assert result["analysis_type"] == "linearity"
    assert "slope" in result
    assert "intercept" in result


def test_crossed_2d_format():
    """Crossed Gage R&R accepts 2D array input (auto-detect flatten)."""
    # 2 parts x 2 operators x 3 replicates, 2D format
    measurements = [[10.1, 10.2, 10.0], [10.4, 10.5, 10.3]]
    parts = ["P1", "P2"]
    operators = ["A", "B", "C"]
    result = gage_rr(analysis_type="crossed", measurements=measurements, parts=parts, operators=operators)
    assert result["analysis_type"] == "crossed"
    assert "variance_components" in result

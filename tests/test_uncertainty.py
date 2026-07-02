"""Tests for stats_engine/uncertainty.py — GUM measurement uncertainty framework."""

import math

from stats_engine.uncertainty import UncertaintySource, measurement_uncertainty


class TestUncertaintySource:
    """Test individual uncertainty source calculations."""

    def test_type_a_from_values(self):
        """Type A source computes std from repeated measurements."""
        src = UncertaintySource(name="repeat", distribution="type_a", value=[10.0, 10.2, 9.8, 10.1, 9.9])
        uc = src.standard_uncertainty
        assert uc > 0
        assert abs(uc - 0.1581) < 0.01  # expected std ≈ 0.1581
        assert src.degrees_of_freedom == 4  # n - 1

    def test_rectangular_distribution(self):
        """Rectangular (uniform) distribution: u = a / sqrt(3)."""
        src = UncertaintySource(name="res", distribution="rectangular", value=0.01)
        expected = 0.01 / math.sqrt(3)
        assert abs(src.standard_uncertainty - expected) < 1e-10

    def test_triangular_distribution(self):
        """Triangular distribution: u = a / sqrt(6)."""
        src = UncertaintySource(name="env", distribution="triangular", value=0.02)
        expected = 0.02 / math.sqrt(6)
        assert abs(src.standard_uncertainty - expected) < 1e-10

    def test_normal_distribution(self):
        """Normal distribution: u = std directly."""
        src = UncertaintySource(name="cal", distribution="normal", value=0.005)
        assert src.standard_uncertainty == 0.005

    def test_u_shaped_distribution(self):
        """U-shaped distribution: u = a / sqrt(2)."""
        src = UncertaintySource(name="rf", distribution="u_shaped", value=0.03)
        expected = 0.03 / math.sqrt(2)
        assert abs(src.standard_uncertainty - expected) < 1e-10

    def test_sensitivity_coefficient(self):
        """Sensitivity coefficient scales standard uncertainty."""
        src = UncertaintySource(name="temp", distribution="rectangular", value=0.1, sensitivity_coefficient=2.0)
        expected_variance = (2.0 * 0.1 / math.sqrt(3)) ** 2
        assert abs(src.variance - expected_variance) < 1e-10


class TestMeasurementUncertainty:
    """Test GUM uncertainty evaluation."""

    def test_single_type_a_source(self):
        """Single Type A source produces valid uncertainty budget."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "repeatability", "values": [10.1, 10.2, 10.0, 10.1, 10.2, 9.9]}],
        )
        assert result["measurement_uncertainty"]["type_a_sources"] == 1
        assert result["measurement_uncertainty"]["type_b_sources"] == 0
        assert result["combined_standard_uncertainty"] > 0
        # Expanded = k * combined (allow rounding tolerance)
        assert (
            abs(result["expanded_uncertainty"] - result["coverage_factor"] * result["combined_standard_uncertainty"])
            < 0.001
        )
        assert len(result["uncertainty_budget"]) == 1

    def test_single_type_b_source(self):
        """Single Type B source produces valid uncertainty budget."""
        result = measurement_uncertainty(
            type_b_sources=[{"name": "resolution", "distribution": "rectangular", "half_width": 0.005}],
        )
        assert result["measurement_uncertainty"]["type_b_sources"] == 1
        expected_uc = 0.005 / math.sqrt(3)
        assert abs(result["combined_standard_uncertainty"] - expected_uc) < 0.001

    def test_mixed_sources(self):
        """Mixed Type A and Type B sources combine correctly."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "repeatability", "values": [10.0, 10.1, 9.9, 10.0, 10.1]}],
            type_b_sources=[
                {"name": "resolution", "distribution": "rectangular", "half_width": 0.005},
                {"name": "calibration", "distribution": "normal", "std": 0.002},
            ],
        )
        assert result["measurement_uncertainty"]["total_sources"] == 3
        assert len(result["uncertainty_budget"]) == 3
        # Budget sorted by contribution (largest first)
        assert (
            result["uncertainty_budget"][0]["variance_contribution"]
            >= result["uncertainty_budget"][-1]["variance_contribution"]
        )

    def test_expanded_uncertainty(self):
        """Expanded uncertainty = k * combined."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "test", "values": [1, 2, 3, 4, 5]}],
            coverage_factor=3.0,
        )
        assert result["coverage_factor"] == 3.0
        assert abs(result["expanded_uncertainty"] - 3.0 * result["combined_standard_uncertainty"]) < 0.001

    def test_sensitivity_coefficients(self):
        """Sensitivity coefficients correctly scale uncertainty."""
        result_no_sens = measurement_uncertainty(
            type_b_sources=[{"name": "temp", "distribution": "rectangular", "half_width": 0.1}],
            sensitivity_coefficients={"temp": 1.0},
        )
        result_with_sens = measurement_uncertainty(
            type_b_sources=[{"name": "temp", "distribution": "rectangular", "half_width": 0.1}],
            sensitivity_coefficients={"temp": 2.0},
        )
        # With sensitivity=2, variance should be 4x (2^2)
        ratio = result_with_sens["combined_standard_uncertainty"] / result_no_sens["combined_standard_uncertainty"]
        assert abs(ratio - 2.0) < 0.01

    def test_budget_percent_contributions_sum(self):
        """Percent contributions should sum to ~100%."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "A", "values": [10, 11, 10, 11, 10]}],
            type_b_sources=[
                {"name": "B1", "distribution": "rectangular", "half_width": 0.01},
                {"name": "B2", "distribution": "normal", "std": 0.005},
            ],
        )
        total_pct = sum(item["percent_contribution"] for item in result["uncertainty_budget"])
        assert abs(total_pct - 100.0) < 0.1

    def test_effective_degrees_of_freedom(self):
        """Effective DoF is computed via Welch-Satterthwaite."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "repeat", "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}],
            type_b_sources=[{"name": "cal", "distribution": "normal", "std": 0.01}],
        )
        dof = result["effective_degrees_of_freedom"]
        assert dof == "inf" or float(dof) > 0

    def test_no_sources_raises(self):
        """No sources raises ValueError."""
        from pytest import raises

        with raises(ValueError, match="At least one"):
            measurement_uncertainty()

    def test_via_handler(self):
        """Measurement uncertainty works through handler()."""
        from main import handler

        result = handler(
            {
                "command": "measurement_uncertainty",
                "params": {
                    "type_a_sources": [{"name": "repeatability", "values": [10.1, 10.2, 10.0, 10.1, 10.2]}],
                    "type_b_sources": [
                        {"name": "resolution", "distribution": "rectangular", "half_width": 0.005},
                        {"name": "calibration", "distribution": "normal", "std": 0.002},
                    ],
                    "coverage_factor": 2,
                },
            }
        )
        assert result["status"] == "success"
        assert "combined_standard_uncertainty" in result["data"]
        assert "expanded_uncertainty" in result["data"]
        assert "uncertainty_budget" in result["data"]

    def test_type_a_single_value_zero_uncertainty(self):
        """Type A with < 2 values gives zero uncertainty (edge case)."""
        src = UncertaintySource(name="single", distribution="type_a", value=[5.0])
        assert src.standard_uncertainty == 0.0

    def test_type_a_scalar_value(self):
        """Type A with scalar std input works."""
        src = UncertaintySource(name="known", distribution="type_a", value=0.05)
        assert src.standard_uncertainty == 0.05

    def test_empty_type_a_source_skipped(self):
        """Type A source with empty values is skipped."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "empty", "values": []}],
            type_b_sources=[{"name": "cal", "distribution": "normal", "std": 0.01}],
        )
        assert result["measurement_uncertainty"]["type_a_sources"] == 0
        assert result["measurement_uncertainty"]["type_b_sources"] == 1

    def test_type_b_missing_key_skipped(self):
        """Type B source with no recognizable value key is skipped."""
        result = measurement_uncertainty(
            type_b_sources=[
                {"name": "bad", "distribution": "rectangular"},  # no half_width/std/value
                {"name": "good", "distribution": "normal", "std": 0.01},
            ],
        )
        assert result["measurement_uncertainty"]["type_b_sources"] == 1
        assert result["uncertainty_budget"][0]["source"] == "good"

    def test_zero_combined_uncertainty(self):
        """Zero combined uncertainty gives infinite dof."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "constant", "values": [5.0, 5.0, 5.0, 5.0]}],
        )
        assert result["combined_standard_uncertainty"] == 0.0
        assert result["effective_degrees_of_freedom"] == "inf"

    def test_measured_value_for_relative(self):
        """Measured value is used for relative uncertainty."""
        result = measurement_uncertainty(
            type_a_sources=[{"name": "repeat", "values": [10.0, 10.1, 9.9]}],
            measured_value=10.0,
        )
        assert result["relative_uncertainty_percent"] is not None
        assert result["relative_uncertainty_percent"] > 0

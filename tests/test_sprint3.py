"""Tests for Sprint 3 advanced features: ALT, Crow-AMSAA, DSD, Z-MR chart."""

import json

import numpy as np
import pytest

from stats_engine.control_chart import control_chart
from stats_engine.discover import COMMANDS
from stats_engine.doe import doe
from stats_engine.reliability import reliability

# ============================================================
# ALT (Accelerated Life Testing)
# ============================================================


class TestALT:
    """Tests for Accelerated Life Testing."""

    def test_arrhenius_basic(self):
        """Test ALT with Arrhenius model and known data."""
        # Temperatures in Kelvin, failure times in hours
        # Arrhenius: ln(t) = a + b/T where b > 0 (activation energy)
        # Known: a=-2, b=5000, T=[350, 400, 450]
        # Lower temperature -> longer life (physical behavior)
        temps = [350, 400, 450]
        failure_times = [
            np.exp(-2 + 5000 / 350),
            np.exp(-2 + 5000 / 400),
            np.exp(-2 + 5000 / 450),
        ]
        result = reliability(
            analysis_type="alt",
            stress_levels=temps,
            failure_times=failure_times,
            stress_model="arrhenius",
            use_stress=300,
        )
        assert result["analysis_type"] == "alt"
        assert result["stress_model"] == "arrhenius"
        assert result["model_name"] == "Arrhenius"
        assert "model_params" in result
        assert "a" in result["model_params"]
        assert "b" in result["model_params"]
        assert "r_squared" in result
        assert result["r_squared"] > 0.99  # should be near-perfect fit
        assert "median_life_at_use" in result
        assert "acceleration_factor" in result
        assert result["acceleration_factor"] > 1  # use stress has longer life

    def test_inverse_power_law(self):
        """Test ALT with Inverse Power Law model."""
        stresses = [10, 20, 30, 40]
        failure_times = [
            np.exp(8 + 1.5 * np.log(10)),
            np.exp(8 + 1.5 * np.log(20)),
            np.exp(8 + 1.5 * np.log(30)),
            np.exp(8 + 1.5 * np.log(40)),
        ]
        result = reliability(
            analysis_type="alt",
            stress_levels=stresses,
            failure_times=failure_times,
            stress_model="inverse_power",
            use_stress=5,
        )
        assert result["analysis_type"] == "alt"
        assert result["stress_model"] == "inverse_power"
        assert result["model_name"] == "Inverse Power Law"
        assert result["r_squared"] > 0.99

    def test_log_linear(self):
        """Test ALT with Log-Linear model."""
        stresses = [1, 2, 3, 4]
        failure_times = [
            np.exp(6 - 0.5 * 1),
            np.exp(6 - 0.5 * 2),
            np.exp(6 - 0.5 * 3),
            np.exp(6 - 0.5 * 4),
        ]
        result = reliability(
            analysis_type="alt",
            stress_levels=stresses,
            failure_times=failure_times,
            stress_model="log_linear",
            use_stress=0.5,
        )
        assert result["analysis_type"] == "alt"
        assert result["stress_model"] == "log_linear"
        assert result["model_name"] == "Log-Linear"
        assert result["r_squared"] > 0.99

    def test_alt_without_use_stress(self):
        """Test ALT without use_stress (no extrapolation)."""
        temps = [350, 400, 450]
        failure_times = [1000, 500, 250]
        result = reliability(
            analysis_type="alt",
            stress_levels=temps,
            failure_times=failure_times,
            stress_model="arrhenius",
        )
        assert "median_life_at_use" not in result
        assert "acceleration_factor" not in result
        assert "model_params" in result

    def test_alt_invalid_model(self):
        """Test ALT with invalid stress model."""
        with pytest.raises(ValueError, match="Unknown stress_model"):
            reliability(
                analysis_type="alt",
                stress_levels=[100, 200],
                failure_times=[1000, 500],
                stress_model="invalid",
            )

    def test_alt_mismatched_lengths(self):
        """Test ALT with mismatched input lengths."""
        with pytest.raises(ValueError, match="same length"):
            reliability(
                analysis_type="alt",
                stress_levels=[100, 200],
                failure_times=[1000, 2000, 3000],
                stress_model="arrhenius",
            )

    def test_alt_insufficient_data(self):
        """Test ALT with fewer than 2 stress levels."""
        with pytest.raises(ValueError, match="at least 2"):
            reliability(
                analysis_type="alt",
                stress_levels=[100],
                failure_times=[1000],
                stress_model="arrhenius",
            )


# ============================================================
# Crow-AMSAA (Repairable Systems)
# ============================================================


class TestCrowAMSAA:
    """Tests for Crow-AMSAA NHPP model."""

    def test_improving_system(self):
        """Test Crow-AMSAA with improving system (beta < 1)."""
        # Simulate an improving system: failures spread out over time
        cumulative_times = [100, 250, 500, 900, 1500, 2500, 4000, 6000]
        cumulative_failures = [1, 2, 3, 4, 5, 6, 7, 8]
        result = reliability(
            analysis_type="crow",
            cumulative_times=cumulative_times,
            cumulative_failures=cumulative_failures,
        )
        assert result["analysis_type"] == "crow_amsaa"
        assert "beta" in result
        assert "rho" in result
        assert "current_mtbf" in result
        assert "growth_rate" in result
        assert "gof_p_value" in result
        assert "interpretation" in result
        # With spreading failure times, beta should be < 1
        assert result["beta"] < 1, f"Expected beta < 1 for improving system, got {result['beta']}"
        assert result["growth_rate"] > 0
        assert "Improving" in result["interpretation"]

    def test_deteriorating_system(self):
        """Test Crow-AMSAA with deteriorating system (beta > 1)."""
        # Simulate a deteriorating system: failures clustering closer together
        cumulative_times = [1000, 1800, 2400, 2800, 3100, 3300, 3450, 3550]
        cumulative_failures = [1, 2, 3, 4, 5, 6, 7, 8]
        result = reliability(
            analysis_type="crow",
            cumulative_times=cumulative_times,
            cumulative_failures=cumulative_failures,
        )
        assert result["analysis_type"] == "crow_amsaa"
        assert result["beta"] > 1, f"Expected beta > 1 for deteriorating system, got {result['beta']}"
        assert result["growth_rate"] < 0
        assert "Deteriorating" in result["interpretation"]

    def test_crow_basic_fields(self):
        """Test Crow-AMSAA returns all required fields."""
        cumulative_times = [100, 300, 600, 1000, 1500, 2100]
        cumulative_failures = [1, 2, 3, 4, 5, 6]
        result = reliability(
            analysis_type="crow",
            cumulative_times=cumulative_times,
            cumulative_failures=cumulative_failures,
        )
        assert "n_failures" in result
        assert "observation_time" in result
        assert "beta" in result
        assert "rho" in result
        assert "current_mtbf" in result
        assert "growth_rate" in result
        assert "laplace_statistic" in result
        assert "gof_p_value" in result
        assert "interpretation" in result

    def test_crow_mismatched_lengths(self):
        """Test Crow-AMSAA with mismatched input lengths."""
        with pytest.raises(ValueError, match="same length"):
            reliability(
                analysis_type="crow",
                cumulative_times=[100, 200, 300],
                cumulative_failures=[1, 2],
            )

    def test_crow_insufficient_data(self):
        """Test Crow-AMSAA with fewer than 2 data points."""
        with pytest.raises(ValueError, match="at least 2"):
            reliability(
                analysis_type="crow",
                cumulative_times=[100],
                cumulative_failures=[1],
            )


# ============================================================
# Definitive Screening Designs
# ============================================================


class TestDSD:
    """Tests for Definitive Screening Designs."""

    def test_dsd_generates_correct_runs(self):
        """Test DSD generates 2k+1 runs."""
        factors = [
            {"name": "A", "low": -1, "high": 1},
            {"name": "B", "low": -1, "high": 1},
            {"name": "C", "low": -1, "high": 1},
        ]
        result = doe(doe_type="definitive_screening", factors=factors)
        assert result["doe_type"] == "definitive_screening"
        assert result["n_factors"] == 3
        assert result["n_runs"] == 2 * 3 + 1  # 7 runs
        assert len(result["design_matrix"]) == 7

    def test_dsd_4_factors(self):
        """Test DSD with 4 factors."""
        factors = [
            {"name": "A", "levels": [-1, 0, 1]},
            {"name": "B", "levels": [-1, 0, 1]},
            {"name": "C", "levels": [-1, 0, 1]},
            {"name": "D", "levels": [-1, 0, 1]},
        ]
        result = doe(doe_type="definitive_screening", factors=factors)
        assert result["n_runs"] == 2 * 4 + 1  # 9 runs
        assert len(result["design_matrix"]) == 9

    def test_dsd_properties(self):
        """Test DSD has correct properties."""
        factors = [
            {"name": "A", "low": 0, "high": 100},
            {"name": "B", "low": 0, "high": 100},
            {"name": "C", "low": 0, "high": 100},
        ]
        result = doe(doe_type="definitive_screening", factors=factors)
        assert "aliasing_structure" in result
        assert "properties" in result
        assert result["properties"]["main_effects_estimable"] is True
        assert result["properties"]["quadratic_terms_estimable"] is True

    def test_dsd_too_few_factors(self):
        """Test DSD requires at least 3 factors."""
        factors = [
            {"name": "A", "low": -1, "high": 1},
            {"name": "B", "low": -1, "high": 1},
        ]
        with pytest.raises(ValueError, match="at least 3 factors"):
            doe(doe_type="definitive_screening", factors=factors)

    def test_dsd_design_matrix_values(self):
        """Test DSD design matrix contains valid factor levels."""
        factors = [
            {"name": "A", "levels": [10, 20, 30]},
            {"name": "B", "levels": [100, 200, 300]},
            {"name": "C", "levels": [1000, 2000, 3000]},
        ]
        result = doe(doe_type="definitive_screening", factors=factors)
        for row in result["design_matrix"]:
            assert row["A"] in [10, 20, 30]
            assert row["B"] in [100, 200, 300]
            assert row["C"] in [1000, 2000, 3000]


# ============================================================
# Z-MR Chart
# ============================================================


class TestZMRChart:
    """Tests for Z-MR (Short-run) chart."""

    def test_zmr_basic(self):
        """Test Z-MR chart basic functionality."""
        values = [10.1, 10.2, 10.0, 10.3, 10.1, 10.2, 10.0, 10.3, 10.1, 10.2]
        result = control_chart(chart_type="zmr", values=values, target=10.0, sigma=0.2)
        assert result["chart_type"] == "zmr"
        assert "z_values" in result
        assert "chart" in result
        assert "mr_chart" in result
        assert result["center"] == 0.0
        assert result["ucl"] == 3.0
        assert result["lcl"] == -3.0

    def test_zmr_standardization(self):
        """Test Z-MR chart standardization is correct."""
        values = [10.0, 10.5, 9.5, 10.0, 10.5]
        result = control_chart(chart_type="zmr", values=values, target=10.0, sigma=0.5)
        z_vals = result["z_values"]
        # Z_i = (x_i - target) / sigma = (x_i - 10.0) / 0.5
        assert z_vals[0] == 0.0  # (10.0 - 10.0) / 0.5
        assert z_vals[1] == 1.0  # (10.5 - 10.0) / 0.5
        assert z_vals[2] == -1.0  # (9.5 - 10.0) / 0.5

    def test_zmr_out_of_control(self):
        """Test Z-MR chart detects out-of-control points."""
        # Values with a clear outlier (Z > 3)
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0]
        result = control_chart(chart_type="zmr", values=values, target=10.0, sigma=0.5)
        # Last value: Z = (12.0 - 10.0) / 0.5 = 4.0, which is > 3
        assert len(result["chart"]["out_of_control_points"]) > 0
        assert result["summary"]["stable"] is False

    def test_zmr_no_target(self):
        """Test Z-MR chart with default target (mean)."""
        values = [10.1, 10.2, 10.0, 10.3, 10.1]
        result = control_chart(chart_type="zmr", values=values)
        assert "target" in result
        assert "sigma" in result

    def test_zmr_in_control(self):
        """Test Z-MR chart with all points in control."""
        # All values near target, within +/-3 sigma
        np.random.seed(42)
        values = (np.random.normal(100, 1, 20)).tolist()
        result = control_chart(chart_type="zmr", values=values, target=100.0, sigma=1.0)
        assert "chart" in result
        assert "summary" in result


# ============================================================
# Discover updates
# ============================================================


class TestDiscoverUpdates:
    """Test that discover.py includes all new features."""

    def test_reliability_includes_alt(self):
        """Test reliability discover entry includes ALT."""
        rel = COMMANDS["reliability"]
        assert "alt" in rel["description"]
        assert "alt" in rel["params"][0]["desc"]
        # Check new params exist
        param_names = [p["name"] for p in rel["params"]]
        assert "stress_levels" in param_names
        assert "failure_times" in param_names
        assert "stress_model" in param_names
        assert "use_stress" in param_names

    def test_reliability_includes_crow(self):
        """Test reliability discover entry includes Crow-AMSAA."""
        rel = COMMANDS["reliability"]
        assert "crow" in rel["description"]
        param_names = [p["name"] for p in rel["params"]]
        assert "cumulative_times" in param_names
        assert "cumulative_failures" in param_names

    def test_doe_includes_dsd(self):
        """Test DOE discover entry includes DSD."""
        d = COMMANDS["doe"]
        assert "definitive_screening" in d["description"]
        assert "definitive_screening" in d["params"][0]["desc"]

    def test_control_chart_includes_zmr(self):
        """Test control_chart discover entry includes Z-MR."""
        cc = COMMANDS["control_chart"]
        assert "zmr" in cc["description"]
        assert "zmr" in cc["params"][1]["desc"]
        param_names = [p["name"] for p in cc["params"]]
        assert "target" in param_names
        assert "sigma" in param_names


# ============================================================
# JSON Serializability
# ============================================================


class TestJSONSerializable:
    """Test that all new features produce JSON-serializable output."""

    def test_alt_json_serializable(self):
        """Test ALT output is JSON serializable."""
        temps = [350, 400, 450]
        failure_times = [1000, 500, 250]
        result = reliability(
            analysis_type="alt",
            stress_levels=temps,
            failure_times=failure_times,
            stress_model="arrhenius",
            use_stress=300,
        )
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_crow_json_serializable(self):
        """Test Crow-AMSAA output is JSON serializable."""
        result = reliability(
            analysis_type="crow",
            cumulative_times=[100, 250, 500, 900, 1500, 2500],
            cumulative_failures=[1, 2, 3, 4, 5, 6],
        )
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_dsd_json_serializable(self):
        """Test DSD output is JSON serializable."""
        factors = [
            {"name": "A", "low": -1, "high": 1},
            {"name": "B", "low": -1, "high": 1},
            {"name": "C", "low": -1, "high": 1},
        ]
        result = doe(doe_type="definitive_screening", factors=factors)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_zmr_json_serializable(self):
        """Test Z-MR chart output is JSON serializable."""
        values = [10.1, 10.2, 10.0, 10.3, 10.1, 10.2, 10.0, 10.3, 10.1, 10.2]
        result = control_chart(chart_type="zmr", values=values, target=10.0, sigma=0.2)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

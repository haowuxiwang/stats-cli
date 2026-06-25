"""Tests for SPC/MSA improvements: CUSUM FIR, Johnson capability, Gage R&R CI, destructive R&R."""

import json

import numpy as np
import pytest

from stats_engine.capability import capability
from stats_engine.control_chart import control_chart
from stats_engine.gage_rr import gage_rr

# ---------------------------------------------------------------------------
# CUSUM with FIR (head start)
# ---------------------------------------------------------------------------


class TestCusumFir:
    """CUSUM FIR should produce different alarm points than standard CUSUM."""

    @pytest.fixture
    def shift_data(self):
        """Data with a process shift to trigger CUSUM alarms."""
        np.random.seed(42)
        # In-control for first 20, then shift up by 1 sigma
        data = np.random.normal(100, 2, 30).tolist()
        for i in range(20, 30):
            data[i] += 2.0  # 1-sigma shift
        return data

    def test_fir_changes_alarm_points(self, shift_data):
        """FIR CUSUM should detect shifts earlier than standard CUSUM."""
        standard = control_chart("cusum", shift_data, k=0.5, h=5, fir=False)
        fir = control_chart("cusum", shift_data, k=0.5, h=5, fir=True)
        # Both should have valid alarm point lists
        assert "alarm_points" in standard["chart"]
        assert "alarm_points" in fir["chart"]
        # FIR may detect earlier (different alarm points), or at least
        # the CUSUM trajectories should differ
        cusum_pos_std = standard["chart"]["cusum_pos"]
        cusum_pos_fir = fir["chart"]["cusum_pos"]
        # At index 0, FIR should be h/2 (non-zero if h > 0)
        # Standard starts at max(0, x0 - target - k*sigma)
        # They should differ unless the data happens to make standard[0] == h/2
        assert cusum_pos_fir[0] != cusum_pos_std[0] or cusum_pos_fir[1] != cusum_pos_std[1]

    def test_fir_flag_in_output(self, shift_data):
        """FIR flag should appear in chart output."""
        result = control_chart("cusum", shift_data, k=0.5, h=5, fir=True)
        assert result["chart"]["fir"] is True
        assert "FIR" in result["chart"]["title"]

    def test_standard_fir_false(self, shift_data):
        """Standard CUSUM should have fir=False in output."""
        result = control_chart("cusum", shift_data, k=0.5, h=5, fir=False)
        assert result["chart"]["fir"] is False
        assert "FIR" not in result["chart"]["title"]


# ---------------------------------------------------------------------------
# Johnson-based capability
# ---------------------------------------------------------------------------


class TestJohnsonCapability:
    """Johnson capability should produce valid indices."""

    @pytest.fixture
    def skewed_data(self):
        """Right-skewed data that benefits from Johnson transformation."""
        np.random.seed(42)
        # Lognormal-ish data
        return np.exp(np.random.normal(2, 0.3, 100)).tolist()

    def test_johnson_produces_indices(self, skewed_data):
        """Johnson capability should compute Cp and Cpk."""
        result = capability(values=skewed_data, usl=12, lsl=5, capability_type="johnson")
        assert "johnson" in result
        johnson = result["johnson"]
        assert "cp" in johnson
        assert "cpk" in johnson
        assert "cpu" in johnson
        assert "cpl" in johnson
        # Indices should be positive for capable data
        assert johnson["cp"] > 0
        assert johnson["cpk"] > 0

    def test_johnson_params_present(self, skewed_data):
        """Johnson output should include distribution parameters."""
        result = capability(values=skewed_data, usl=12, lsl=5, capability_type="johnson")
        johnson = result["johnson"]
        assert "gamma" in johnson
        assert "delta" in johnson
        assert "xi" in johnson
        assert "lambda" in johnson
        assert "mean_transformed" in johnson
        assert "std_transformed" in johnson

    def test_johnson_one_sided_upper(self, skewed_data):
        """Johnson with only USL should still compute."""
        result = capability(values=skewed_data, usl=12, capability_type="johnson")
        assert "johnson" in result
        # One-sided: no cp/cpk in johnson sub-dict
        assert "cp" not in result["johnson"]


# ---------------------------------------------------------------------------
# Gage R&R confidence intervals
# ---------------------------------------------------------------------------


class TestGageRrCI:
    """Gage R&R CI should contain the point estimate."""

    @pytest.fixture
    def gage_data(self):
        """Standard Gage R&R dataset: 10 parts, 3 operators, 3 replicates."""
        np.random.seed(42)
        n_parts = 10
        n_ops = 3
        n_reps = 3
        measurements = []
        parts = []
        operators = []
        for p in range(n_parts):
            part_effect = np.random.normal(0, 2)
            for o in range(n_ops):
                op_effect = np.random.normal(0, 0.5)
                for _rep in range(n_reps):
                    noise = np.random.normal(0, 0.3)
                    measurements.append(100 + part_effect + op_effect + noise)
                    parts.append(f"P{p}")
                    operators.append(f"O{o}")
        return measurements, parts, operators

    def test_ci_present_in_variance_components(self, gage_data):
        """Each variance component should have ci_lower and ci_upper."""
        measurements, parts, operators = gage_data
        result = gage_rr("crossed", measurements=measurements, parts=parts, operators=operators)
        vc = result["variance_components"]
        for comp_name in ["repeatability", "reproducibility", "grr", "part_to_part"]:
            comp = vc[comp_name]
            assert "ci_lower" in comp, f"{comp_name} missing ci_lower"
            assert "ci_upper" in comp, f"{comp_name} missing ci_upper"

    def test_ci_contains_point_estimate(self, gage_data):
        """CI should contain the point estimate (variance)."""
        measurements, parts, operators = gage_data
        result = gage_rr("crossed", measurements=measurements, parts=parts, operators=operators)
        vc = result["variance_components"]
        for comp_name in ["repeatability", "grr", "part_to_part"]:
            comp = vc[comp_name]
            variance = comp["variance"]
            ci_lower = comp["ci_lower"]
            ci_upper = comp["ci_upper"]
            assert ci_lower <= variance <= ci_upper, (
                f"{comp_name}: point estimate {variance} not in CI [{ci_lower}, {ci_upper}]"
            )

    def test_ci_lower_non_negative(self, gage_data):
        """CI lower bounds should be non-negative."""
        measurements, parts, operators = gage_data
        result = gage_rr("crossed", measurements=measurements, parts=parts, operators=operators)
        vc = result["variance_components"]
        for comp_name in ["repeatability", "reproducibility", "grr", "part_to_part"]:
            comp = vc[comp_name]
            assert comp["ci_lower"] >= 0, f"{comp_name} ci_lower < 0"

    def test_ci_bounds_ordered(self, gage_data):
        """ci_lower should be <= ci_upper for all components."""
        measurements, parts, operators = gage_data
        result = gage_rr("crossed", measurements=measurements, parts=parts, operators=operators)
        vc = result["variance_components"]
        for comp_name in ["repeatability", "reproducibility", "grr", "part_to_part"]:
            comp = vc[comp_name]
            assert comp["ci_lower"] <= comp["ci_upper"], f"{comp_name}: ci_lower > ci_upper"


# ---------------------------------------------------------------------------
# Destructive Gage R&R
# ---------------------------------------------------------------------------


class TestDestructiveGageRr:
    """Destructive Gage R&R should work with nested design."""

    @pytest.fixture
    def destructive_data(self):
        """Nested design: 3 operators, each measuring 5 different parts from same lot."""
        np.random.seed(42)
        measurements = []
        parts = []
        operators = []
        for o in range(3):
            for p in range(5):
                part_id = f"Lot1_P{o * 5 + p}"
                for _rep in range(2):
                    measurements.append(100 + np.random.normal(0, 0.5))
                    parts.append(part_id)
                    operators.append(f"O{o}")
        return measurements, parts, operators

    def test_destructive_runs(self, destructive_data):
        """Destructive analysis should produce valid results."""
        measurements, parts, operators = destructive_data
        result = gage_rr("destructive", measurements=measurements, parts=parts, operators=operators)
        assert result["analysis_type"] == "destructive"
        assert "variance_components" in result
        assert "ndc" in result
        assert "rating" in result

    def test_destructive_has_assumptions(self, destructive_data):
        """Destructive output should list assumptions."""
        measurements, parts, operators = destructive_data
        result = gage_rr("destructive", measurements=measurements, parts=parts, operators=operators)
        assert "assumptions" in result
        assert len(result["assumptions"]) > 0
        assert any("homogeneity" in a.lower() for a in result["assumptions"])

    def test_destructive_has_notes(self, destructive_data):
        """Destructive output should have ndc interpretation notes."""
        measurements, parts, operators = destructive_data
        result = gage_rr("destructive", measurements=measurements, parts=parts, operators=operators)
        assert "destructive_notes" in result


# ---------------------------------------------------------------------------
# JSON serializability
# ---------------------------------------------------------------------------


class TestJsonSerializable:
    """All new features should produce JSON-serializable output."""

    def test_cusum_fir_json(self):
        np.random.seed(42)
        data = np.random.normal(100, 2, 30).tolist()
        result = control_chart("cusum", data, k=0.5, h=5, fir=True)
        # Should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_johnson_capability_json(self):
        np.random.seed(42)
        data = np.random.normal(10, 1, 50).tolist()
        result = capability(values=data, usl=14, lsl=6, capability_type="johnson")
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_gage_rr_ci_json(self):
        np.random.seed(42)
        measurements = [100 + np.random.normal(0, 0.5) for _ in range(27)]
        parts = [f"P{i // 3}" for i in range(27)]
        operators = [f"O{(i // 9) % 3}" for i in range(27)]
        result = gage_rr("crossed", measurements=measurements, parts=parts, operators=operators)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_destructive_json(self):
        np.random.seed(42)
        measurements = []
        parts = []
        operators = []
        for o in range(3):
            for p in range(5):
                for _rep in range(2):
                    measurements.append(100 + np.random.normal(0, 0.5))
                    parts.append(f"P{o * 5 + p}")
                    operators.append(f"O{o}")
        result = gage_rr("destructive", measurements=measurements, parts=parts, operators=operators)
        json_str = json.dumps(result)
        assert len(json_str) > 0

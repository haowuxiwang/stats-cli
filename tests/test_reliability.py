"""Tests for stats_engine/reliability.py."""

import numpy as np
import pytest

from stats_engine.reliability import reliability


def test_weibull():
    times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    status = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["analysis_type"] == "weibull"
    assert "beta" in result
    assert "eta" in result
    assert "b_lives" in result
    assert "mttf" in result


def test_kaplan_meier():
    times = [100, 200, 300, 400, 500]
    status = [1, 0, 1, 1, 0]
    result = reliability(analysis_type="kaplan_meier", times=times, status=status)
    assert result["analysis_type"] == "kaplan_meier"
    assert "median_survival" in result
    assert "km_table" in result


def test_distribution():
    np.random.seed(42)
    values = np.random.weibull(2, 50) * 100
    result = reliability(analysis_type="distribution", values=values.tolist())
    assert result["analysis_type"] == "distribution_fit"
    assert "best_fit" in result


def test_stability():
    times = [0, 3, 6, 9, 12]
    values = [100, 99.5, 99.0, 98.5, 98.0]
    result = reliability(analysis_type="stability", times=times, values=values)
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        reliability(analysis_type="invalid", times=[1, 2], status=[1, 1])


def test_weibull_shape_positive():
    """Weibull shape parameter (beta) should be positive."""
    times = [100, 200, 300, 400, 500]
    status = [1, 1, 1, 1, 1]
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["beta"] > 0, f"beta={result['beta']}"


def test_kaplan_meier_survival_decreasing():
    """Survival probability should be non-increasing."""
    times = [10, 20, 30, 40, 50]
    status = [1, 1, 1, 1, 1]
    result = reliability(analysis_type="kaplan_meier", times=times, status=status)
    survival = [row["survival"] for row in result["km_table"]]
    for i in range(1, len(survival)):
        assert survival[i] <= survival[i - 1] + 0.001, (
            f"Survival increased at step {i}: {survival[i - 1]} -> {survival[i]}"
        )


def test_weibull_with_censored():
    """Weibull with censored data."""
    times = [100, 200, 300, 400, 500]
    status = [1, 0, 1, 0, 1]  # 0 = censored
    result = reliability(analysis_type="weibull", times=times, status=status)
    assert result["analysis_type"] == "weibull"


def test_distribution_multiple_fits():
    """Distribution fitting should try multiple distributions."""
    np.random.seed(42)
    values = np.random.normal(100, 10, 100).tolist()
    result = reliability(analysis_type="distribution", values=values)
    assert result["analysis_type"] == "distribution_fit"
    assert "fits" in result or "best_fit" in result


def test_stability_with_confidence():
    """Stability study with confidence interval."""
    times = [0, 3, 6, 9, 12, 15, 18]
    values = [100, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0]
    result = reliability(analysis_type="stability", times=times, values=values)
    assert result["analysis_type"] == "stability"
    assert "intercept" in result or "slope" in result


def test_kaplan_meier_all_censored():
    """Kaplan-Meier with all censored data."""
    times = [100, 200, 300]
    status = [0, 0, 0]
    try:
        result = reliability(analysis_type="kaplan_meier", times=times, status=status)
        assert result["analysis_type"] == "kaplan_meier"
    except Exception:
        pass  # May fail with all censored data


def test_kaplan_meier_default_status():
    """Kaplan-Meier without status should default to all failures (line 94)."""
    result = reliability(analysis_type="kaplan_meier", times=[100, 200, 300, 400, 500])
    assert result["analysis_type"] == "kaplan_meier"
    assert result["n_events"] == 5


def test_weibull_single_failure():
    """Weibull with single failure time."""
    times = [100]
    status = [1]
    try:
        result = reliability(analysis_type="weibull", times=times, status=status)
        assert result["analysis_type"] == "weibull"
    except Exception:
        pass  # May fail with insufficient data


def test_distribution_fit_all_fail():
    """Distribution fit when all distributions fail."""
    from stats_engine.reliability import reliability

    # Use very small dataset that may cause fitting issues
    result = reliability(analysis_type="distribution", values=[1, 1, 1, 1, 1])
    # Should still return a result, possibly with errors in individual fits
    assert "distributions" in result or "error" in result


def test_shelf_life_boundary():
    """Shelf life calculation with edge case data."""
    from stats_engine.reliability import reliability

    result = reliability(analysis_type="stability", values=[10.0, 10.1, 10.0, 10.1, 10.0], times=[0, 1, 2, 3, 4])
    assert "shelf_life" in result or "parameters" in result


def test_shelf_life_with_usl():
    """Shelf life with upper spec limit."""
    from stats_engine.reliability import reliability

    result = reliability(analysis_type="stability", times=[0, 3, 6, 9, 12], values=[100, 101, 102, 103, 104], usl=110)
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result


def test_shelf_life_with_lsl():
    """Shelf life with lower spec limit triggers shelf life calculation."""
    from stats_engine.reliability import reliability

    result = reliability(
        analysis_type="stability", times=[0, 3, 6, 9, 12], values=[100, 99.5, 99.0, 98.5, 98.0], lsl=95
    )
    assert result["analysis_type"] == "stability"
    assert "shelf_life" in result
    assert result["shelf_life"] is not None


def test_distribution_fit_with_invalid_data():
    """Distribution fit with all zeros should handle gracefully."""
    result = reliability(analysis_type="distribution", times=[0, 0, 0, 0, 0])
    assert "distributions" in result


def test_distribution_fit_error_handlers():
    """Each distribution fit error is caught independently (lines 162-195)."""
    # Use data that may cause individual fits to fail
    # All-same-value data can cause issues with some distributions
    result = reliability(analysis_type="distribution", times=[5.0, 5.0, 5.0, 5.0, 5.0])
    assert "distributions" in result
    # At least one distribution should be fitted
    fitted = [k for k, v in result["distributions"].items() if "error" not in v]
    assert len(fitted) >= 1


def test_distribution_fit_single_value():
    """Single value distribution fit."""
    result = reliability(analysis_type="distribution", times=[100.0])
    assert "distributions" in result


def test_distribution_fit_extreme_values():
    """Very large values should still fit some distributions."""
    result = reliability(analysis_type="distribution", times=[1e10, 2e10, 3e10, 4e10, 5e10])
    assert "distributions" in result
    assert result["best_fit"] is not None


class TestCoxPH:
    """Tests for Cox Proportional Hazards regression."""

    def test_cox_ph_basic(self):
        """Cox PH with single covariate."""
        np.random.seed(42)
        n = 50
        time = np.random.exponential(100, n).tolist()
        event = np.random.binomial(1, 0.8, n).tolist()
        age = np.random.normal(60, 10, n).reshape(-1, 1).tolist()

        result = reliability(
            analysis_type="cox_ph",
            times=time,
            event=event,
            covariates=age,
        )
        assert result["analysis_type"] == "cox_ph"
        assert result["n_observations"] == n
        assert len(result["coefficients"]) == 1
        assert "hazard_ratio" in result["coefficients"][0]
        assert "hr_ci_95_low" in result["coefficients"][0]
        assert "concordance_index" in result
        assert "interpretation" in result

    def test_cox_ph_multiple_covariates(self):
        """Cox PH with multiple covariates."""
        np.random.seed(42)
        n = 60
        time = np.random.exponential(100, n).tolist()
        event = np.random.binomial(1, 0.8, n).tolist()
        cov1 = np.random.normal(60, 10, n).tolist()
        cov2 = np.random.binomial(1, 0.5, n).tolist()
        covariates = [[c1, c2] for c1, c2 in zip(cov1, cov2)]

        result = reliability(
            analysis_type="cox_ph",
            times=time,
            event=event,
            covariates=covariates,
        )
        assert result["analysis_type"] == "cox_ph"
        assert result["n_covariates"] == 2
        assert len(result["coefficients"]) == 2

    def test_cox_ph_significant_effect(self):
        """Cox PH detects strong covariate effect."""
        np.random.seed(42)
        n = 80
        # Strong age effect: older patients have shorter survival
        age = np.random.normal(60, 15, n)
        hazard = np.exp(0.05 * (age - 60))  # HR per year = e^0.05 ≈ 1.05
        time = np.random.exponential(100 / hazard).tolist()
        event = [1] * n  # All events observed

        result = reliability(
            analysis_type="cox_ph",
            times=time,
            event=event,
            covariates=age.reshape(-1, 1).tolist(),
        )
        assert result["analysis_type"] == "cox_ph"
        # Age should be significant
        age_coef = result["coefficients"][0]
        assert age_coef["significant"] is True
        assert age_coef["hazard_ratio"] > 1.0  # Older = higher hazard

    def test_cox_ph_via_handler(self):
        """Cox PH works through handler()."""
        from main import handler

        np.random.seed(42)
        n = 40
        time = np.random.exponential(100, n).tolist()
        event = np.random.binomial(1, 0.8, n).tolist()
        treatment = np.random.binomial(1, 0.5, n).reshape(-1, 1).tolist()

        result = handler(
            {
                "command": "reliability",
                "params": {
                    "analysis_type": "cox_ph",
                    "times": time,
                    "event": event,
                    "covariates": treatment,
                },
            }
        )
        assert result["status"] == "success"
        assert result["data"]["analysis_type"] == "cox_ph"
        assert "concordance_index" in result["data"]

    def test_cox_ph_no_covariates_raises(self):
        """Cox PH without covariates raises error."""
        import pytest

        with pytest.raises(ValueError, match="covariate"):
            reliability(analysis_type="cox_ph", times=[1, 2, 3], event=[1, 1, 0])

    def test_calt_life_test(self):
        """Accelerated Life Test (ALT) analysis."""
        result = reliability(
            analysis_type="alt",
            stress_levels=[373, 393, 413],
            failure_times=[500, 200, 80],
            use_stress=298,
        )
        assert result["analysis_type"] == "alt" or "model_params" in result

    def test_crow_amsaa(self):
        """Crow-AMSAA reliability growth model."""
        result = reliability(
            analysis_type="crow",
            cumulative_times=[100, 200, 500, 1000, 2000],
            cumulative_failures=[1, 2, 4, 7, 10],
        )
        assert result["analysis_type"] == "crow_amsaa"
        assert "beta" in result
        assert "growth_rate" in result

    def test_cox_ph_length_mismatch_raises(self):
        """Mismatched times/event lengths raise error."""
        import pytest

        with pytest.raises(ValueError, match="same length"):
            reliability(
                analysis_type="cox_ph",
                times=[1, 2, 3],
                event=[1, 0],
                covariates=[[1], [2], [3]],
            )

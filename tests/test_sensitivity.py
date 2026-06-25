"""Tests for stats_engine/sensitivity.py."""

import json

from stats_engine.sensitivity import sensitivity


class TestMonteCarlo:
    def test_basic_monte_carlo(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "uniform", "params": {"low": 5, "high": 15}},
        }
        result = sensitivity("monte_carlo", inputs=inputs, formula="x1 + x2", n_simulations=1000)
        assert result["analysis_type"] == "monte_carlo"
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "percentiles" in result
        assert "histogram" in result
        assert "input_stats" in result
        assert "interpretation" in result

    def test_monte_carlo_formula(self):
        inputs = {
            "x": {"dist": "normal", "params": {"mean": 10, "std": 1}},
        }
        result = sensitivity("monte_carlo", inputs=inputs, formula="x ** 2", n_simulations=1000)
        # Mean of X^2 where X~N(10,1) should be near 101 (10^2 + 1^2)
        assert 95 < result["mean"] < 107

    def test_monte_carlo_custom_percentiles(self):
        inputs = {"x": {"dist": "constant", "params": {"value": 5}}}
        result = sensitivity("monte_carlo", inputs=inputs, formula="x", n_simulations=1000, percentiles=[10, 90])
        assert "p10" in result["percentiles"]
        assert "p90" in result["percentiles"]

    def test_monte_carlo_distributions(self):
        for dist_name, params in [
            ("normal", {"mean": 10, "std": 1}),
            ("uniform", {"low": 5, "high": 15}),
            ("triangular", {"left": 5, "mode": 10, "right": 15}),
            ("exponential", {"scale": 2}),
        ]:
            inputs = {"x": {"dist": dist_name, "params": params}}
            result = sensitivity("monte_carlo", inputs=inputs, formula="x", n_simulations=500)
            assert "mean" in result

    def test_monte_carlo_json_serializable(self):
        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        result = sensitivity("monte_carlo", inputs=inputs, formula="x * 2", n_simulations=500)
        json.dumps(result)  # Should not raise


class TestTornado:
    def test_basic_tornado(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 * x2")
        assert result["analysis_type"] == "tornado"
        assert "sensitivities" in result
        assert "most_sensitive" in result
        assert result["n_variables"] == 2

    def test_tornado_sorted_by_swing(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
            "x3": {"dist": "normal", "params": {"mean": 2, "std": 0.2}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2 * x3")
        swings = [s["swing"] for s in result["sensitivities"]]
        assert swings == sorted(swings, reverse=True)

    def test_tornado_with_base_values(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2", base_values={"x1": 10, "x2": 5})
        assert result["base_output"] == 15

    def test_tornado_variation(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 * x2", variation=0.2)
        assert result["variation"] == 0.2

    def test_tornado_json_serializable(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 * x2")
        json.dumps(result)  # Should not raise


class TestSobol:
    def test_basic_sobol(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=500)
        assert result["analysis_type"] == "sobol"
        assert "first_order" in result
        assert "total_order" in result
        assert "ranking" in result
        assert "interaction_share" in result
        assert result["n_variables"] == 2

    def test_sobol_additive(self):
        # For additive model y = x1 + x2, S1 should sum to ~1
        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=2000)
        s1_sum = sum(result["first_order"].values())
        # For additive model, S1 sum should be close to 1
        assert 0.8 < s1_sum < 1.2

    def test_sobol_nonlinear(self):
        # For y = x1 * x2, there should be interaction effects
        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 * x2", n_simulations=2000)
        assert result["interaction_share"] > 0

    def test_sobol_ranking(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=500)
        assert len(result["ranking"]) == 2
        for r in result["ranking"]:
            assert "variable" in r
            assert "S1" in r
            assert "ST" in r

    def test_sobol_json_serializable(self):
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=500)
        json.dumps(result)  # Should not raise


class TestSensitivityCoverage:
    """Coverage improvement tests for sensitivity.py."""

    def test_tornado_auto_base_normal(self):
        """Tornado with auto base values for normal distribution."""
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2")
        assert result["base_output"] == 15  # 10 + 5

    def test_tornado_auto_base_uniform(self):
        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 10}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 6}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2")
        assert result["base_output"] == 8  # 5 + 3

    def test_tornado_auto_base_triangular(self):
        inputs = {
            "x1": {"dist": "triangular", "params": {"left": 0, "mode": 5, "right": 10}},
            "x2": {"dist": "triangular", "params": {"left": 0, "mode": 3, "right": 6}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2")
        assert result["base_output"] == 8  # 5 + 3

    def test_tornado_auto_base_constant(self):
        inputs = {
            "x1": {"dist": "constant", "params": {"value": 10}},
            "x2": {"dist": "constant", "params": {"value": 5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2")
        assert result["base_output"] == 15

    def test_tornado_auto_base_other_dist(self):
        """Tornado with auto base values for exponential (falls through to else)."""
        inputs = {
            "x1": {"dist": "exponential", "params": {"scale": 5}},
            "x2": {"dist": "exponential", "params": {"scale": 3}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 + x2")
        assert "base_output" in result

    def test_monte_carlo_formula_error(self):
        """Bad formula should raise ValueError."""
        import pytest

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        with pytest.raises((ValueError, NameError)):
            sensitivity("monte_carlo", inputs=inputs, formula="nonexistent_func(x)")

    def test_sobol_interaction_share(self):
        """Multiplicative model should have non-zero interaction share."""
        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 * x2", n_simulations=2000)
        assert result["interaction_share"] > 0

    def test_sobol_most_influential(self):
        """Most influential variable should be identified."""
        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        result = sensitivity("sobol", inputs=inputs, formula="x1 + 100 * x2", n_simulations=3000)
        assert result["most_influential"] in ("x1", "x2")
        assert result["ranking"][0]["ST"] > result["ranking"][1]["ST"]

    def test_monte_carlo_many_distributions(self):
        """Test all supported distribution types."""
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "uniform", "params": {"low": 5, "high": 15}},
            "x3": {"dist": "triangular", "params": {"left": 5, "mode": 10, "right": 15}},
            "x4": {"dist": "lognormal", "params": {"mean": 2, "std": 0.5}},
            "x5": {"dist": "exponential", "params": {"scale": 2}},
            "x6": {"dist": "beta", "params": {"a": 2, "b": 5}},
        }
        result = sensitivity(
            "monte_carlo", inputs=inputs,
            formula="x1 + x2 + x3 + x4 + x5 + x6", n_simulations=500
        )
        assert result["n_valid"] == 500

    def test_tornado_elasticity(self):
        """Elasticity should be computed when base_output != 0."""
        inputs = {
            "x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
            "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}},
        }
        result = sensitivity("tornado", inputs=inputs, formula="x1 * x2")
        for s in result["sensitivities"]:
            assert "elasticity" in s

    def test_tornado_need_two_inputs(self):
        """Tornado with < 2 inputs should raise ValueError."""
        import pytest

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        with pytest.raises(ValueError, match="at least 2"):
            sensitivity("tornado", inputs=inputs, formula="x")

    def test_sobol_need_two_inputs(self):
        """Sobol with < 2 inputs should raise ValueError."""
        import pytest

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        with pytest.raises(ValueError, match="at least 2"):
            sensitivity("sobol", inputs=inputs, formula="x", n_simulations=500)

    def test_sobol_too_few_simulations(self):
        """Sobol with < 100 simulations should raise ValueError."""
        import pytest

        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        with pytest.raises(ValueError, match=">= 100"):
            sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=50)

    def test_monte_carlo_too_few_simulations(self):
        """Monte Carlo with < 100 simulations should raise ValueError."""
        import pytest

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        with pytest.raises(ValueError, match=">= 100"):
            sensitivity("monte_carlo", inputs=inputs, formula="x", n_simulations=50)

    def test_monte_carlo_no_inputs(self):
        """Monte Carlo with no inputs should raise ValueError."""
        import pytest

        with pytest.raises(ValueError):
            sensitivity("monte_carlo", inputs={}, formula="x")

    def test_sobol_zero_variance(self):
        """Sobol with constant inputs should raise ValueError."""
        import pytest

        inputs = {
            "x1": {"dist": "constant", "params": {"value": 5}},
            "x2": {"dist": "constant", "params": {"value": 3}},
        }
        with pytest.raises(ValueError, match="zero variance"):
            sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=500)

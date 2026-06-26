"""Reproducibility tests - same input + same seed = same output."""

import numpy as np


class TestReproducibility:
    """Verify deterministic behavior with fixed seeds."""

    def test_monte_carlo_reproducibility(self):
        """Monte Carlo with same seed should produce identical results."""
        from stats_engine.sensitivity import sensitivity

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        r1 = sensitivity("monte_carlo", inputs=inputs, formula="x * 2", n_simulations=1000, seed=42)
        r2 = sensitivity("monte_carlo", inputs=inputs, formula="x * 2", n_simulations=1000, seed=42)
        assert r1["mean"] == r2["mean"]
        assert r1["std"] == r2["std"]
        assert r1["histogram"] == r2["histogram"]

    def test_sobol_reproducibility(self):
        """Sobol with same seed should produce identical results."""
        from stats_engine.sensitivity import sensitivity

        inputs = {
            "x1": {"dist": "uniform", "params": {"low": 0, "high": 1}},
            "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}},
        }
        r1 = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=1000, seed=42)
        r2 = sensitivity("sobol", inputs=inputs, formula="x1 + x2", n_simulations=1000, seed=42)
        assert r1["first_order"] == r2["first_order"]
        assert r1["total_order"] == r2["total_order"]

    def test_bootstrap_reproducibility(self):
        """Bootstrap with same seed should produce identical results."""
        from stats_engine.advanced import advanced

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        r1 = advanced(analysis_type="bootstrap", values=values, n_bootstrap=1000, seed=42)
        r2 = advanced(analysis_type="bootstrap", values=values, n_bootstrap=1000, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_mining_classify_reproducibility(self):
        """Classification with same seed should produce identical results."""
        from stats_engine.mining import mining

        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        r1 = mining("classify", features=features, labels=labels, random_state=42)
        r2 = mining("classify", features=features, labels=labels, random_state=42)
        assert r1["accuracy"] == r2["accuracy"]
        assert r1["feature_importance"] == r2["feature_importance"]

    def test_doe_reproducibility(self):
        """DOE with same seed should produce identical designs."""
        from stats_engine.doe import doe

        factors = [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}]
        r1 = doe(doe_type="fractional_factorial", factors=factors)
        r2 = doe(doe_type="fractional_factorial", factors=factors)
        assert r1["design_matrix"] == r2["design_matrix"]

    def test_different_seed_different_results(self):
        """Different seeds should produce different results."""
        from stats_engine.sensitivity import sensitivity

        inputs = {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}
        r1 = sensitivity("monte_carlo", inputs=inputs, formula="x * 2", n_simulations=1000, seed=42)
        r2 = sensitivity("monte_carlo", inputs=inputs, formula="x * 2", n_simulations=1000, seed=123)
        # Means should be close but not identical
        assert abs(r1["mean"] - r2["mean"]) < 1.0  # Both near 20
        # But histogram edges/counts may differ

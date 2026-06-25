"""Tests for stats_engine/functional.py."""

import json

import numpy as np
import pytest

from stats_engine.functional import functional

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def quadratic_data():
    """y = x^2 on [0, 4], 21 points."""
    t = np.linspace(0, 4, 21)
    y = t**2
    return t.tolist(), y.tolist()


@pytest.fixture
def sinusoidal_data():
    """y = sin(t) on [0, 2*pi], 50 points."""
    t = np.linspace(0, 2 * np.pi, 50)
    y = np.sin(t)
    return t.tolist(), y.tolist()


@pytest.fixture
def noisy_sinusoidal():
    """y = sin(t) + noise on [0, 2*pi], 100 points."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(t) + rng.normal(0, 0.1, 100)
    return t.tolist(), y.tolist()


@pytest.fixture
def multi_curves():
    """5 curves on a 30-point grid, for FPCA testing."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 30)
    base = np.sin(t)
    curves = []
    for i in range(5):
        curves.append((base + rng.normal(0, 0.2, 30) + i * 0.1 * t).tolist())
    return t.tolist(), curves


# ---------------------------------------------------------------------------
# Basis tests
# ---------------------------------------------------------------------------


class TestBasis:
    def test_bspline(self, quadratic_data):
        t, y = quadratic_data
        result = functional("basis", t=t, values=y, basis_type="bspline", n_basis=6)
        assert result["analysis_type"] == "basis"
        assert result["basis_type"] == "bspline"
        assert result["n_basis"] == 6
        assert len(result["coefficients"]) == 6
        assert len(result["reconstructed_values"]) == len(t)
        # Quadratic is easy to fit; residual should be very low
        assert result["residual_rms"] < 0.5

    def test_fourier(self, sinusoidal_data):
        t, y = sinusoidal_data
        result = functional("basis", t=t, values=y, basis_type="fourier", n_basis=3)
        assert result["analysis_type"] == "basis"
        assert result["basis_type"] == "fourier"
        assert result["n_basis"] == 3
        assert len(result["coefficients"]) == 3
        # Sinusoidal with 3 harmonics should reconstruct well
        assert result["residual_rms"] < 0.5

    def test_polynomial(self, quadratic_data):
        t, y = quadratic_data
        result = functional("basis", t=t, values=y, basis_type="polynomial", n_basis=3)
        assert result["analysis_type"] == "basis"
        assert result["basis_type"] == "polynomial"
        assert result["n_basis"] == 3
        assert len(result["coefficients"]) == 3
        # Exact quadratic -> near-zero residual
        assert result["residual_rms"] < 1e-6

    def test_basis_low_residual(self, quadratic_data):
        """Polynomial basis with degree 2 should fit quadratic data almost perfectly."""
        t, y = quadratic_data
        result = functional("basis", t=t, values=y, basis_type="polynomial", n_basis=3)
        assert result["residual_rms"] < 1e-6

    def test_basis_default_n_basis(self, quadratic_data):
        """Default n_basis should work without error."""
        t, y = quadratic_data
        result = functional("basis", t=t, values=y, basis_type="bspline")
        assert "n_basis" in result
        assert result["n_basis"] >= 2

    def test_basis_unknown_type(self, quadratic_data):
        t, y = quadratic_data
        with pytest.raises(ValueError, match="Unknown basis_type"):
            functional("basis", t=t, values=y, basis_type="invalid")

    def test_basis_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            functional("basis", t=[1, 2, 3], values=[1, 2], basis_type="polynomial")


# ---------------------------------------------------------------------------
# Smooth tests
# ---------------------------------------------------------------------------


class TestSmooth:
    def test_spline(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="spline")
        assert result["analysis_type"] == "smooth"
        assert result["method"] == "spline"
        assert len(result["smoothed_values"]) == len(t)
        assert "smoothing_param" in result
        assert "effective_df" in result
        assert "residual_variance" in result

    def test_kernel(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="kernel")
        assert result["analysis_type"] == "smooth"
        assert result["method"] == "kernel"
        assert len(result["smoothed_values"]) == len(t)
        assert result["effective_df"] > 0

    def test_lowess(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="lowess", smoothing_param=0.3)
        assert result["analysis_type"] == "smooth"
        assert result["method"] == "lowess"
        assert len(result["smoothed_values"]) == len(t)

    def test_smoothing_reduces_noise(self, noisy_sinusoidal):
        """Smoothed values should have lower variance than raw noisy data."""
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="spline")
        raw_var = np.var(y)
        smooth_var = np.var(result["smoothed_values"])
        assert smooth_var < raw_var

    def test_spline_with_param(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="spline", smoothing_param=1.0)
        assert result["analysis_type"] == "smooth"

    def test_kernel_with_bandwidth(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="kernel", smoothing_param=0.5)
        assert result["smoothing_param"] == 0.5

    def test_smooth_unknown_method(self, quadratic_data):
        t, y = quadratic_data
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            functional("smooth", t=t, values=y, method="invalid")

    def test_smooth_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            functional("smooth", t=[1, 2, 3], values=[1, 2], method="spline")


# ---------------------------------------------------------------------------
# Derivative tests
# ---------------------------------------------------------------------------


class TestDerivative:
    def test_order1_finite_diff(self, quadratic_data):
        t, y = quadratic_data
        result = functional("derivative", t=t, values=y, order=1, method="finite_diff")
        assert result["analysis_type"] == "derivative"
        assert result["order"] == 1
        assert result["method"] == "finite_diff"
        assert len(result["derivative_values"]) == len(t)
        assert "notable_points" in result

    def test_order2_finite_diff(self, quadratic_data):
        t, y = quadratic_data
        result = functional("derivative", t=t, values=y, order=2, method="finite_diff")
        assert result["order"] == 2
        # d^2(x^2)/dx^2 = 2, so all values should be close to 2
        deriv = result["derivative_values"]
        interior = deriv[2:-2]  # skip boundary effects
        for v in interior:
            assert abs(v - 2.0) < 0.5

    def test_order1_spline(self, sinusoidal_data):
        t, y = sinusoidal_data
        result = functional("derivative", t=t, values=y, order=1, method="spline")
        assert result["method"] == "spline"
        assert len(result["derivative_values"]) == len(t)
        # d(sin)/dt = cos; check a few interior points
        t_arr = np.array(t)
        deriv = np.array(result["derivative_values"])
        mid = len(t) // 2
        assert abs(deriv[mid] - np.cos(t_arr[mid])) < 0.2

    def test_order2_spline(self, sinusoidal_data):
        t, y = sinusoidal_data
        result = functional("derivative", t=t, values=y, order=2, method="spline")
        assert result["order"] == 2
        assert result["method"] == "spline"

    def test_extrema_detection(self, sinusoidal_data):
        """First derivative of sin(t) should find extrema near pi/2 and 3*pi/2."""
        t, y = sinusoidal_data
        result = functional("derivative", t=t, values=y, order=1, method="spline")
        extrema = result["notable_points"]["extrema"]
        assert len(extrema) >= 1  # at least one extremum detected
        # Check that extrema are near pi/2 (~1.57) or 3*pi/2 (~4.71)
        for ext in extrema:
            t_val = ext["t"]
            assert (abs(t_val - np.pi / 2) < 0.5) or (abs(t_val - 3 * np.pi / 2) < 0.5)

    def test_derivative_unknown_method(self, quadratic_data):
        t, y = quadratic_data
        with pytest.raises(ValueError, match="Unknown derivative method"):
            functional("derivative", t=t, values=y, method="invalid")

    def test_derivative_invalid_order(self, quadratic_data):
        t, y = quadratic_data
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            functional("derivative", t=t, values=y, order=3)


# ---------------------------------------------------------------------------
# FPCA tests
# ---------------------------------------------------------------------------


class TestFPCA:
    def test_basic_fpca(self, multi_curves):
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t, n_components=3)
        assert result["analysis_type"] == "fpca"
        assert result["n_curves"] == 5
        assert result["n_points"] == 30
        assert result["n_components"] == 3
        assert len(result["mean_function"]) == 30
        assert len(result["eigenfunctions"]) == 3
        assert len(result["scores"]) == 5
        assert len(result["variance_explained"]) == 3

    def test_variance_explained_sums(self, multi_curves):
        """Variance explained should sum to <= 1."""
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t, n_components=3)
        total = sum(result["variance_explained"])
        assert total <= 1.0 + 1e-6  # allow tiny numerical overshoot
        assert total > 0.0

    def test_cumulative_variance(self, multi_curves):
        """Cumulative variance should be monotonically increasing."""
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t, n_components=3)
        cum = result["cumulative_variance"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1]

    def test_fpca_default_components(self, multi_curves):
        """Default n_components should extract all."""
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t)
        assert result["n_components"] == min(5, 30)  # min(n_curves, n_points)

    def test_fpca_scores_shape(self, multi_curves):
        """Scores matrix shape should be (n_curves, n_components)."""
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t, n_components=2)
        assert len(result["scores"]) == 5
        for row in result["scores"]:
            assert len(row) == 2

    def test_fpca_t_length_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            functional("fpca", curves=[[1, 2, 3], [4, 5, 6]], t=[0, 1])

    def test_fpca_too_few_curves(self):
        with pytest.raises(ValueError, match="at least 2"):
            functional("fpca", curves=[[1, 2, 3]], t=[0, 1, 2])


# ---------------------------------------------------------------------------
# JSON serializability
# ---------------------------------------------------------------------------


class TestJSON:
    def _assert_json_serializable(self, result):
        """Verify result can be serialized to JSON without errors."""
        json_str = json.dumps(result, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["analysis_type"] == result["analysis_type"]

    def test_basis_json(self, quadratic_data):
        t, y = quadratic_data
        result = functional("basis", t=t, values=y, basis_type="polynomial", n_basis=3)
        self._assert_json_serializable(result)

    def test_smooth_json(self, noisy_sinusoidal):
        t, y = noisy_sinusoidal
        result = functional("smooth", t=t, values=y, method="kernel")
        self._assert_json_serializable(result)

    def test_derivative_json(self, quadratic_data):
        t, y = quadratic_data
        result = functional("derivative", t=t, values=y, order=1)
        self._assert_json_serializable(result)

    def test_fpca_json(self, multi_curves):
        t, curves = multi_curves
        result = functional("fpca", curves=curves, t=t, n_components=2)
        self._assert_json_serializable(result)


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------


class TestGuards:
    def test_unknown_analysis_type(self):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            functional("nonexistent")

    def test_basis_nan_values(self):
        with pytest.raises(ValueError, match="NaN"):
            functional("basis", t=[0, 1, 2], values=[0, float("nan"), 4], basis_type="polynomial", n_basis=2)

    def test_basis_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            functional("basis", t=[1], values=[1], basis_type="polynomial")

    def test_basis_n_basis_too_large(self):
        with pytest.raises(ValueError, match="must be less than"):
            functional("basis", t=[0, 1, 2, 3], values=[0, 1, 4, 9], basis_type="polynomial", n_basis=5)

    def test_basis_n_basis_too_large_fourier(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            functional("basis", t=[0, 1, 2, 3, 4, 5], values=[0, 1, 4, 9, 16, 25], basis_type="fourier", n_basis=4)

    def test_smooth_too_few_spline(self):
        with pytest.raises(ValueError, match="at least 4"):
            functional("smooth", t=[0, 1, 2], values=[0, 1, 4], method="spline")

    def test_smooth_too_few_lowess(self):
        with pytest.raises(ValueError, match="at least 4"):
            functional("smooth", t=[0, 1, 2], values=[0, 1, 4], method="lowess")

    def test_lowess_bad_frac(self):
        with pytest.raises(ValueError, match="smoothing_param"):
            functional("smooth", t=[0, 1, 2, 3, 4], values=[0, 1, 4, 9, 16], method="lowess", smoothing_param=1.5)

    def test_fpca_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            functional("fpca", curves=[1, 2, 3], t=[0, 1, 2])

    def test_derivative_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            functional("derivative", t=[0, 1, 2], values=[0, 1])

    def test_basis_length_mismatch_array(self):
        with pytest.raises(ValueError, match="same length"):
            functional("basis", t=[0, 1, 2, 3, 4], values=[0, 1, 4], basis_type="polynomial")


# ---------------------------------------------------------------------------
# Functional regression tests
# ---------------------------------------------------------------------------


class TestRegression:
    """Tests for functional regression (scalar_on_function, function_on_scalar)."""

    def test_scalar_on_function_basic(self):
        """Predict scalar from curves with a known linear relationship."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        n = 20
        # y_i = integral of curve_i + noise
        curves = np.zeros((n, len(t)))
        y = np.zeros(n)
        for i in range(n):
            slope = rng.uniform(0.5, 3.0)
            curves[i] = slope * t + rng.normal(0, 0.05, len(t))
            y[i] = slope * 0.5 + rng.normal(0, 0.1)  # y ~ integral of curve
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="scalar_on_function",
            y=y.tolist(),
            n_basis=5,
        )
        assert result["analysis_type"] == "regression"
        assert result["mode"] == "scalar_on_function"
        assert result["n_basis"] == 5
        assert "r_squared" in result
        assert 0.0 <= result["r_squared"] <= 1.0
        assert len(result["predictions"]) == n
        assert len(result["residuals"]) == n
        assert "interpretation" in result

    def test_scalar_on_function_good_fit(self):
        """Curves that strongly predict y should give high R-squared."""
        rng = np.random.default_rng(123)
        t = np.linspace(0, 1, 50)
        n = 30
        heights = rng.uniform(1, 5, n)
        curves = np.zeros((n, len(t)))
        for i in range(n):
            curves[i] = heights[i] * np.sin(np.pi * t) + rng.normal(0, 0.02, len(t))
        y = heights.tolist()  # y is just the height
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="scalar_on_function",
            y=y,
            n_basis=6,
        )
        assert result["r_squared"] > 0.8

    def test_scalar_on_function_missing_y(self):
        """Should raise error when y is not provided."""
        curves = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="'y'"):
            functional("regression", curves=curves, t=[0, 1, 2], mode="scalar_on_function")

    def test_scalar_on_function_y_length_mismatch(self):
        curves = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="'y' length"):
            functional(
                "regression",
                curves=curves,
                t=[0, 1, 2],
                mode="scalar_on_function",
                y=[1, 2, 3],
            )

    def test_function_on_scalar_basic(self):
        """Predict curves from scalar predictors."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        n = 15
        X = rng.uniform(0, 10, n)
        curves = np.zeros((n, len(t)))
        for i in range(n):
            curves[i] = X[i] * t + rng.normal(0, 0.1, len(t))
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="function_on_scalar",
            X=X.tolist(),
        )
        assert result["analysis_type"] == "regression"
        assert result["mode"] == "function_on_scalar"
        assert result["n_predictors"] == 1
        assert len(result["r_squared_per_point"]) == len(t)
        assert "mean_r_squared" in result
        assert "coefficient_functions" in result
        assert "intercept" in result["coefficient_functions"]
        assert "predictor_1" in result["coefficient_functions"]

    def test_function_on_scalar_multiple_predictors(self):
        """Predict curves from multiple scalar predictors."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        n = 20
        X1 = rng.uniform(0, 5, n)
        X2 = rng.uniform(0, 3, n)
        X = np.column_stack([X1, X2])
        curves = np.zeros((n, len(t)))
        for i in range(n):
            curves[i] = X1[i] * t + X2[i] * t**2 + rng.normal(0, 0.1, len(t))
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="function_on_scalar",
            X=X.tolist(),
        )
        assert result["n_predictors"] == 2
        assert "predictor_1" in result["coefficient_functions"]
        assert "predictor_2" in result["coefficient_functions"]

    def test_function_on_scalar_missing_X(self):
        curves = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="'X'"):
            functional("regression", curves=curves, t=[0, 1, 2], mode="function_on_scalar")

    def test_regression_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            functional("regression", curves=[[1, 2], [3, 4]], t=[0, 1], mode="invalid")


# ---------------------------------------------------------------------------
# FANOVA tests
# ---------------------------------------------------------------------------


class TestFANOVA:
    """Tests for functional ANOVA."""

    def test_fanova_different_groups(self):
        """Groups with different means should produce significant results."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        n_per_group = 15
        # Group 1: curves around y=0
        g1 = np.zeros((n_per_group, len(t))) + rng.normal(0, 0.1, (n_per_group, len(t)))
        # Group 2: curves around y=3
        g2 = 3.0 + np.zeros((n_per_group, len(t))) + rng.normal(0, 0.1, (n_per_group, len(t)))
        # Group 3: curves around y=6
        g3 = 6.0 + np.zeros((n_per_group, len(t))) + rng.normal(0, 0.1, (n_per_group, len(t)))

        result = functional(
            "fanova",
            groups=[g1.tolist(), g2.tolist(), g3.tolist()],
            t=t.tolist(),
            alpha=0.05,
        )
        assert result["analysis_type"] == "fanova"
        assert result["n_groups"] == 3
        assert result["f_statistic"] > 10  # should be very large
        assert result["n_significant_points"] == len(t)  # all points significant
        assert len(result["significant_regions"]) >= 1
        assert result["alpha"] == 0.05

    def test_fanova_same_groups(self):
        """Groups drawn from same distribution should not be significant."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        n_per_group = 10
        g1 = rng.normal(0, 1, (n_per_group, len(t)))
        g2 = rng.normal(0, 1, (n_per_group, len(t)))

        result = functional(
            "fanova",
            groups=[g1.tolist(), g2.tolist()],
            t=t.tolist(),
            alpha=0.05,
        )
        assert result["analysis_type"] == "fanova"
        assert result["n_groups"] == 2
        # With same distribution, most points should not be significant
        assert result["n_significant_points"] < len(t) * 0.5

    def test_fanova_two_groups(self):
        t = np.linspace(0, 1, 20)
        rng = np.random.default_rng(99)
        g1 = 2.0 + rng.normal(0, 0.05, (10, len(t)))
        g2 = -2.0 + rng.normal(0, 0.05, (10, len(t)))
        result = functional("fanova", groups=[g1.tolist(), g2.tolist()], t=t.tolist())
        assert result["f_statistic"] > 5
        assert len(result["pointwise_p_values"]) == len(t)

    def test_fanova_pointwise_stats_length(self):
        t = np.linspace(0, 1, 15)
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, (8, 15))
        g2 = rng.normal(0, 1, (8, 15))
        result = functional("fanova", groups=[g1.tolist(), g2.tolist()], t=t.tolist())
        assert len(result["pointwise_f_stats"]) == 15
        assert len(result["pointwise_p_values"]) == 15
        assert len(result["adjusted_p_values"]) == 15

    def test_fanova_fewer_than_two_groups(self):
        with pytest.raises(ValueError, match="at least 2"):
            functional("fanova", groups=[[[1, 2], [3, 4]]], t=[0, 1])

    def test_fanova_significant_regions_format(self):
        t = np.linspace(0, 1, 20)
        rng = np.random.default_rng(42)
        g1 = 5.0 + rng.normal(0, 0.01, (12, 20))
        g2 = -5.0 + rng.normal(0, 0.01, (12, 20))
        result = functional("fanova", groups=[g1.tolist(), g2.tolist()], t=t.tolist())
        for region in result["significant_regions"]:
            assert len(region) == 2
            assert region[0] <= region[1]


# ---------------------------------------------------------------------------
# Functional clustering tests
# ---------------------------------------------------------------------------


class TestCluster:
    """Tests for functional clustering."""

    def test_cluster_kmeans_basic(self):
        """Curves from 2 distinct patterns should cluster well."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        # Cluster 1: curves centered at y=0
        c1 = np.zeros((10, 30)) + rng.normal(0, 0.05, (10, 30))
        # Cluster 2: curves centered at y=5
        c2 = 5.0 + np.zeros((10, 30)) + rng.normal(0, 0.05, (10, 30))
        curves = np.vstack([c1, c2])

        result = functional(
            "cluster",
            curves=curves.tolist(),
            t=t.tolist(),
            n_clusters=2,
            method="kmeans",
        )
        assert result["analysis_type"] == "cluster"
        assert result["n_clusters"] == 2
        assert result["n_curves"] == 20
        assert len(result["cluster_labels"]) == 20
        assert len(result["cluster_centers"]) == 2
        assert result["silhouette_score"] > 0.5
        assert sum(result["cluster_sizes"]) == 20

    def test_cluster_hierarchical(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        c1 = np.zeros((8, 30)) + rng.normal(0, 0.05, (8, 30))
        c2 = 5.0 + np.zeros((8, 30)) + rng.normal(0, 0.05, (8, 30))
        curves = np.vstack([c1, c2])

        result = functional(
            "cluster",
            curves=curves.tolist(),
            t=t.tolist(),
            n_clusters=2,
            method="hierarchical",
        )
        assert result["method"] == "hierarchical"
        assert result["silhouette_score"] > 0.3

    def test_cluster_three_groups(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 30)
        c1 = rng.normal(0, 0.1, (8, 30))
        c2 = 5.0 + rng.normal(0, 0.1, (8, 30))
        c3 = 10.0 + rng.normal(0, 0.1, (8, 30))
        curves = np.vstack([c1, c2, c3])

        result = functional(
            "cluster",
            curves=curves.tolist(),
            t=t.tolist(),
            n_clusters=3,
            method="kmeans",
        )
        assert result["n_clusters"] == 3
        assert sum(result["cluster_sizes"]) == 24
        assert len(result["fpca_variance_explained"]) > 0

    def test_cluster_labels_are_integers(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        curves = rng.normal(0, 1, (10, 20))
        result = functional("cluster", curves=curves.tolist(), t=t.tolist(), n_clusters=2)
        for label in result["cluster_labels"]:
            assert isinstance(label, int)
            assert 0 <= label < 2

    def test_cluster_n_clusters_too_low(self):
        with pytest.raises(ValueError, match="at least 2"):
            functional("cluster", curves=[[1, 2], [3, 4]], t=[0, 1], n_clusters=1)

    def test_cluster_n_clusters_too_high(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            functional("cluster", curves=[[1, 2], [3, 4]], t=[0, 1], n_clusters=5)

    def test_cluster_unknown_method(self):
        rng = np.random.default_rng(42)
        curves = rng.normal(0, 1, (10, 20))
        t = np.linspace(0, 1, 20)
        with pytest.raises(ValueError, match="Unknown method"):
            functional("cluster", curves=curves.tolist(), t=t.tolist(), method="invalid")


# ---------------------------------------------------------------------------
# JSON serializability for new types
# ---------------------------------------------------------------------------


class TestJSONNewTypes:
    """JSON serializability tests for regression, fanova, cluster."""

    def _assert_json_serializable(self, result):
        json_str = json.dumps(result, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["analysis_type"] == result["analysis_type"]

    def test_scalar_on_function_json(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        curves = rng.normal(0, 1, (10, 20))
        y = rng.normal(0, 1, 10).tolist()
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="scalar_on_function",
            y=y,
            n_basis=4,
        )
        self._assert_json_serializable(result)

    def test_function_on_scalar_json(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        curves = rng.normal(0, 1, (10, 20))
        X = rng.normal(0, 1, 10).tolist()
        result = functional(
            "regression",
            curves=curves.tolist(),
            t=t.tolist(),
            mode="function_on_scalar",
            X=X,
        )
        self._assert_json_serializable(result)

    def test_fanova_json(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 15)
        g1 = rng.normal(0, 1, (8, 15))
        g2 = rng.normal(0, 1, (8, 15))
        result = functional("fanova", groups=[g1.tolist(), g2.tolist()], t=t.tolist())
        self._assert_json_serializable(result)

    def test_cluster_json(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 20)
        curves = rng.normal(0, 1, (10, 20))
        result = functional("cluster", curves=curves.tolist(), t=t.tolist(), n_clusters=2)
        self._assert_json_serializable(result)

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

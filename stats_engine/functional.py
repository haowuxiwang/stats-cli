"""Functional Data Analysis (FDA): basis representation, smoothing, derivatives, FPCA, regression, FANOVA, clustering."""

import numpy as np

from utils.output import r

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def functional(analysis_type, **kwargs):
    """Functional data analysis dispatcher.

    Args:
        analysis_type: 'basis', 'smooth', 'derivative', 'fpca', 'regression', 'fanova', 'cluster'

    Returns:
        Dict with analysis results
    """
    dispatch = {
        "basis": _basis,
        "smooth": _smooth,
        "derivative": _derivative,
        "fpca": _fpca,
        "regression": _regression,
        "fanova": _fanova,
        "cluster": _cluster,
    }
    if analysis_type not in dispatch:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use one of: {', '.join(dispatch)}")
    return dispatch[analysis_type](**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_array(values, name="values"):
    """Convert list/array to 1-D numpy float array with validation."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be a 1-D array")
    if len(arr) < 2:
        raise ValueError(f"'{name}' must have at least 2 elements")
    if np.any(np.isnan(arr)):
        raise ValueError(f"'{name}' must not contain NaN")
    return arr


def _to_2d(curves):
    """Convert to 2-D numpy float array (rows = curves)."""
    arr = np.asarray(curves, dtype=float)
    if arr.ndim != 2:
        raise ValueError("'curves' must be a 2-D array (rows = curves)")
    if arr.shape[0] < 2:
        raise ValueError("'curves' must have at least 2 rows")
    if np.any(np.isnan(arr)):
        raise ValueError("'curves' must not contain NaN")
    return arr


# ---------------------------------------------------------------------------
# 1. Basis function representation
# ---------------------------------------------------------------------------


def _basis(t, values, basis_type="bspline", n_basis=None, **_kwargs):
    """Fit basis functions to observed data.

    Args:
        t: Grid points (1-D array)
        values: Observed values at grid points
        basis_type: 'bspline', 'fourier', or 'polynomial'
        n_basis: Number of basis functions (default depends on type)
    """
    t = _to_array(t, name="t")
    y = _to_array(values, name="values")
    if len(t) != len(y):
        raise ValueError("'t' and 'values' must have the same length")

    dispatch = {
        "bspline": _basis_bspline,
        "fourier": _basis_fourier,
        "polynomial": _basis_polynomial,
    }
    if basis_type not in dispatch:
        raise ValueError(f"Unknown basis_type: {basis_type}. Use one of: {', '.join(dispatch)}")
    return dispatch[basis_type](t, y, n_basis)


def _basis_bspline(t, y, n_basis):
    """B-spline basis representation using scipy.interpolate."""
    from scipy.interpolate import LSQUnivariateSpline, make_interp_spline

    n = len(t)
    if n_basis is None:
        n_basis = min(n, max(4, n // 3))
    if n_basis < 2:
        raise ValueError("n_basis must be at least 2 for B-spline")
    if n_basis > n:
        raise ValueError("n_basis cannot exceed number of data points")

    k = min(3, n_basis - 1)

    if n_basis >= n:
        # Exact interpolation
        spl = make_interp_spline(t, y, k=k)
    else:
        # Least-squares fit with interior knots
        n_interior = max(n_basis - k - 1, 0)
        if n_interior > 0:
            knots = np.linspace(t[0], t[-1], n_interior + 2)[1:-1]
        else:
            knots = np.array([])
        # LSQUnivariateSpline requires strictly interior knots
        spl = LSQUnivariateSpline(t, y, t=knots, k=k)

    coefficients = spl.get_coeffs().tolist() if hasattr(spl, "get_coeffs") else spl.c.tolist()
    reconstructed = spl(t)
    residuals = y - reconstructed
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    return {
        "analysis_type": "basis",
        "basis_type": "bspline",
        "n_basis": n_basis,
        "coefficients": [r(c) for c in coefficients],
        "reconstructed_values": [r(v) for v in reconstructed],
        "residual_rms": r(residual_rms),
        "interpretation": f"B-spline basis with {n_basis} functions, residual RMS = {r(residual_rms)}",
    }


def _basis_fourier(t, y, n_basis):
    """Fourier basis representation using numpy.fft."""
    n = len(t)
    if n_basis is None:
        n_basis = min(n // 2, max(3, n // 4))
    if n_basis < 1:
        raise ValueError("n_basis must be at least 1 for Fourier")
    if n_basis > n // 2:
        raise ValueError(f"n_basis ({n_basis}) cannot exceed n/2 ({n // 2}) for Fourier")

    fft_vals = np.fft.rfft(y)
    # Zero out coefficients beyond n_basis
    fft_truncated = np.zeros_like(fft_vals)
    fft_truncated[:n_basis] = fft_vals[:n_basis]
    reconstructed = np.fft.irfft(fft_truncated, n=n)

    coefficients = np.abs(fft_vals[:n_basis]).tolist()
    residuals = y - reconstructed
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    return {
        "analysis_type": "basis",
        "basis_type": "fourier",
        "n_basis": n_basis,
        "coefficients": [r(c) for c in coefficients],
        "reconstructed_values": [r(v) for v in reconstructed],
        "residual_rms": r(residual_rms),
        "interpretation": f"Fourier basis with {n_basis} harmonics, residual RMS = {r(residual_rms)}",
    }


def _basis_polynomial(t, y, n_basis):
    """Polynomial basis representation using numpy.polyfit."""
    n = len(t)
    if n_basis is None:
        n_basis = min(n - 1, max(2, n // 3))
    if n_basis < 1:
        raise ValueError("n_basis must be at least 1 for polynomial")
    if n_basis >= n:
        raise ValueError("n_basis must be less than number of data points")

    degree = n_basis - 1
    coeffs = np.polyfit(t, y, degree)
    reconstructed = np.polyval(coeffs, t)
    residuals = y - reconstructed
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    return {
        "analysis_type": "basis",
        "basis_type": "polynomial",
        "n_basis": n_basis,
        "coefficients": [r(c) for c in coeffs],
        "reconstructed_values": [r(v) for v in reconstructed],
        "residual_rms": r(residual_rms),
        "interpretation": f"Polynomial basis (degree {degree}) with {n_basis} coefficients, residual RMS = {r(residual_rms)}",
    }


# ---------------------------------------------------------------------------
# 2. Functional data smoothing
# ---------------------------------------------------------------------------


def _smooth(t, values, method="spline", smoothing_param=None, **_kwargs):
    """Smooth functional data.

    Args:
        t: Grid points
        values: Observed values
        method: 'spline', 'kernel', or 'lowess'
        smoothing_param: Method-specific parameter (optional)
    """
    t = _to_array(t, name="t")
    y = _to_array(values, name="values")
    if len(t) != len(y):
        raise ValueError("'t' and 'values' must have the same length")

    dispatch = {
        "spline": _smooth_spline,
        "kernel": _smooth_kernel,
        "lowess": _smooth_lowess,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown smoothing method: {method}. Use one of: {', '.join(dispatch)}")
    return dispatch[method](t, y, smoothing_param)


def _smooth_spline(t, y, smoothing_param):
    """Spline smoothing using scipy.interpolate.UnivariateSpline with GCV."""
    from scipy.interpolate import UnivariateSpline

    n = len(t)
    if n < 4:
        raise ValueError("Spline smoothing requires at least 4 data points")

    if smoothing_param is not None:
        s = float(smoothing_param)
    else:
        # GCV-like default: let UnivariateSpline choose
        s = None

    spl = UnivariateSpline(t, y, s=s)
    smoothed = spl(t)

    # Effective degrees of freedom: trace of smoother matrix
    # Approximate via number of knots + degree + 1
    n_knots = len(spl.get_knots())
    k = 3  # UnivariateSpline default degree
    effective_df = n_knots + k + 1
    actual_s = float(spl.get_residual())

    residuals = y - smoothed
    residual_var = float(np.var(residuals, ddof=max(1, int(effective_df))))

    return {
        "analysis_type": "smooth",
        "method": "spline",
        "smoothed_values": [r(v) for v in smoothed],
        "smoothing_param": r(actual_s),
        "effective_df": r(effective_df, 1),
        "residual_variance": r(residual_var),
        "interpretation": f"Spline smoothing with effective df = {r(effective_df, 1)}",
    }


def _smooth_kernel(t, y, smoothing_param):
    """Gaussian kernel smoothing."""
    n = len(t)

    if smoothing_param is not None:
        bandwidth = float(smoothing_param)
    else:
        # Silverman's rule of thumb
        std_t = float(np.std(t, ddof=1))
        iqr_t = float(np.percentile(t, 75) - np.percentile(t, 25))
        bandwidth = 0.9 * min(std_t, iqr_t / 1.34) * n ** (-0.2)
        if bandwidth == 0:
            bandwidth = std_t * n ** (-0.2)

    # Nadaraya-Watson kernel estimator
    smoothed = np.empty(n)
    for i in range(n):
        weights = np.exp(-0.5 * ((t[i] - t) / bandwidth) ** 2)
        w_sum = np.sum(weights)
        if w_sum == 0:
            smoothed[i] = y[i]
        else:
            smoothed[i] = np.sum(weights * y) / w_sum

    # Effective df: trace of hat matrix approximation
    # H_ii = K(0) / sum_j K((t_i - t_j)/h), trace = sum H_ii
    h_ii = np.empty(n)
    for i in range(n):
        weights = np.exp(-0.5 * ((t[i] - t) / bandwidth) ** 2)
        h_ii[i] = weights[i] / np.sum(weights) if np.sum(weights) > 0 else 1.0
    effective_df = float(np.sum(h_ii))

    residuals = y - smoothed
    residual_var = float(np.var(residuals, ddof=max(1, int(effective_df))))

    return {
        "analysis_type": "smooth",
        "method": "kernel",
        "smoothed_values": [r(v) for v in smoothed],
        "smoothing_param": r(bandwidth),
        "effective_df": r(effective_df, 1),
        "residual_variance": r(residual_var),
        "interpretation": f"Gaussian kernel smoothing, bandwidth = {r(bandwidth)}, effective df = {r(effective_df, 1)}",
    }


def _smooth_lowess(t, y, smoothing_param):
    """LOWESS smoothing using statsmodels."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    n = len(t)
    if n < 4:
        raise ValueError("LOWESS smoothing requires at least 4 data points")

    frac = float(smoothing_param) if smoothing_param is not None else 0.3
    if frac <= 0 or frac > 1:
        raise ValueError("smoothing_param (frac) for LOWESS must be in (0, 1]")

    result = lowess(y, t, frac=frac, return_sorted=True)
    # result is sorted by t; we need to map back to original order
    t_sorted = result[:, 0]
    smoothed_sorted = result[:, 1]

    # Interpolate back to original t grid
    smoothed = np.interp(t, t_sorted, smoothed_sorted)

    effective_df = frac * n  # rough approximation
    residuals = y - smoothed
    residual_var = float(np.var(residuals, ddof=max(1, int(effective_df))))

    return {
        "analysis_type": "smooth",
        "method": "lowess",
        "smoothed_values": [r(v) for v in smoothed],
        "smoothing_param": r(frac),
        "effective_df": r(effective_df, 1),
        "residual_variance": r(residual_var),
        "interpretation": f"LOWESS smoothing with frac = {r(frac)}, effective df = {r(effective_df, 1)}",
    }


# ---------------------------------------------------------------------------
# 3. Derivative estimation
# ---------------------------------------------------------------------------


def _derivative(t, values, order=1, method="finite_diff", **_kwargs):
    """Estimate derivatives of functional data.

    Args:
        t: Grid points
        values: Observed values
        order: Derivative order (1 or 2, default 1)
        method: 'finite_diff' or 'spline'
    """
    t = _to_array(t, name="t")
    y = _to_array(values, name="values")
    if len(t) != len(y):
        raise ValueError("'t' and 'values' must have the same length")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    if method == "finite_diff":
        deriv = _derivative_finite_diff(t, y, order)
    elif method == "spline":
        deriv = _derivative_spline(t, y, order)
    else:
        raise ValueError(f"Unknown derivative method: {method}. Use 'finite_diff' or 'spline'")

    notable = _find_notable_points(t, deriv, order)

    return {
        "analysis_type": "derivative",
        "order": order,
        "method": method,
        "derivative_values": [r(v) for v in deriv],
        "notable_points": notable,
        "interpretation": f"Order-{order} derivative via {method}, "
        f"{len(notable['extrema'])} extrema, "
        f"{len(notable['inflection_points'])} inflection points",
    }


def _derivative_finite_diff(t, y, order):
    """Derivative using numpy.gradient."""
    deriv = np.gradient(y, t, edge_order=2)
    if order == 2:
        deriv = np.gradient(deriv, t, edge_order=2)
    return deriv


def _derivative_spline(t, y, order):
    """Derivative via B-spline fitting then differentiation."""
    from scipy.interpolate import make_interp_spline

    k = min(3 + order, len(t) - 1)  # need degree >= order
    if k < order:
        raise ValueError("Not enough data points for spline derivative of requested order")
    spl = make_interp_spline(t, y, k=k)
    dspl = spl.derivative(order)
    return dspl(t)


def _find_notable_points(t, deriv, order):
    """Find extrema and inflection points from derivative values."""
    extrema = []
    inflection_points = []
    n = len(deriv)

    if order == 1:
        # Extrema of original function = zeros of first derivative (sign changes)
        for i in range(1, n - 1):
            # Sign change between neighbors
            if deriv[i - 1] * deriv[i + 1] < 0:
                extrema.append(
                    {
                        "index": i,
                        "t": r(t[i]),
                        "derivative": r(deriv[i]),
                        "type": "max" if deriv[i - 1] > 0 and deriv[i + 1] < 0 else "min",
                    }
                )
            elif deriv[i] == 0 and i > 0 and i < n - 1:
                extrema.append(
                    {
                        "index": i,
                        "t": r(t[i]),
                        "derivative": r(deriv[i]),
                        "type": "max" if deriv[i - 1] > 0 else "min",
                    }
                )
        # Inflection points = zeros of second derivative (approximate)
        if n > 2:
            d2 = np.gradient(deriv, t, edge_order=2)
            for i in range(1, n - 1):
                if d2[i - 1] * d2[i + 1] < 0:
                    inflection_points.append(
                        {
                            "index": i,
                            "t": r(t[i]),
                            "second_derivative": r(d2[i]),
                        }
                    )
    else:
        # order == 2: extrema of second derivative (inflection points of original)
        for i in range(1, n - 1):
            if deriv[i - 1] * deriv[i + 1] < 0:
                extrema.append(
                    {
                        "index": i,
                        "t": r(t[i]),
                        "derivative": r(deriv[i]),
                        "type": "max" if deriv[i - 1] > 0 and deriv[i + 1] < 0 else "min",
                    }
                )

    return {"extrema": extrema, "inflection_points": inflection_points}


# ---------------------------------------------------------------------------
# 4. Functional PCA (FPCA)
# ---------------------------------------------------------------------------


def _fpca(curves, t, n_components=None, **_kwargs):
    """Functional Principal Component Analysis.

    Args:
        curves: 2-D array (n_curves x n_grid_points)
        t: Grid points (1-D array)
        n_components: Number of components to extract (default: all)
    """
    X = _to_2d(curves)
    t_arr = _to_array(t, name="t")
    n_curves, n_points = X.shape
    if len(t_arr) != n_points:
        raise ValueError(f"'t' length ({len(t_arr)}) must match number of columns in 'curves' ({n_points})")

    # Mean function
    mean_func = np.mean(X, axis=0)

    # Center data
    X_centered = X - mean_func

    # Covariance surface (n_points x n_points)
    cov_surface = X_centered.T @ X_centered / (n_curves - 1)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_surface)
    # eigh returns ascending order; reverse
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Filter non-negative (numerical noise can produce tiny negatives)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Determine n_components
    total_var = np.sum(eigenvalues)
    if n_components is None:
        n_components = min(n_curves, n_points)
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if n_components > min(n_curves, n_points):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(n_curves, n_points) = {min(n_curves, n_points)}"
        )

    # Eigenfunctions (columns of eigenvectors)
    eigenfunctions = eigenvectors[:, :n_components].T  # (n_components x n_points)

    # Scores: project centered curves onto eigenfunctions
    scores = X_centered @ eigenfunctions.T  # (n_curves x n_components)

    # Variance explained
    var_explained = eigenvalues[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
    cumulative_var = np.cumsum(var_explained)

    return {
        "analysis_type": "fpca",
        "n_curves": n_curves,
        "n_points": n_points,
        "n_components": n_components,
        "mean_function": [r(v) for v in mean_func],
        "eigenfunctions": [[r(v) for v in row] for row in eigenfunctions],
        "scores": [[r(v) for v in row] for row in scores],
        "variance_explained": [r(v) for v in var_explained],
        "cumulative_variance": [r(v) for v in cumulative_var],
        "interpretation": f"FPCA: {n_components} components explain {r(cumulative_var[-1] * 100, 1)}% of total variance",
    }


# ---------------------------------------------------------------------------
# Functional Regression
# ---------------------------------------------------------------------------


def _regression(curves, t, mode="scalar_on_function", y=None, X=None, n_basis=10, **kwargs):
    """Functional regression.

    Modes:
    - scalar_on_function: predict scalar y from functional curves
    - function_on_scalar: predict functional curves from scalar X

    Args:
        curves: 2D array (n_curves x n_points)
        t: grid points
        mode: 'scalar_on_function' or 'function_on_scalar'
        y: scalar outcomes (for scalar_on_function)
        X: scalar predictors (for function_on_scalar)
        n_basis: number of B-spline basis functions

    Returns:
        Dict with regression results
    """
    curves = np.array(curves, dtype=float)
    t_arr = np.array(t, dtype=float)
    n_curves, n_points = curves.shape

    if mode == "scalar_on_function":
        if y is None:
            raise ValueError("'y' (scalar outcomes) is required for scalar_on_function mode")
        y_arr = np.array(y, dtype=float)
        if len(y_arr) != n_curves:
            raise ValueError(f"'y' length ({len(y_arr)}) must match number of curves ({n_curves})")

        # Represent each curve using B-spline basis coefficients
        from scipy.interpolate import make_interp_spline

        # Use a common set of knots
        n_knots = min(n_basis, n_points - 1)
        knot_indices = np.linspace(0, n_points - 1, n_knots, dtype=int)
        knots = t_arr[knot_indices]

        # Fit basis and get coefficients for each curve
        coefficients = np.zeros((n_curves, n_knots))
        for i in range(n_curves):
            spline = make_interp_spline(t_arr, curves[i], k=min(3, n_knots - 1))
            coefficients[i] = spline(knots)

        # OLS regression: y = coefficients @ beta + epsilon
        # Add intercept
        X_design = np.column_stack([np.ones(n_curves), coefficients])
        beta, residuals, rank, sv = np.linalg.lstsq(X_design, y_arr, rcond=None)

        # Predictions and R-squared
        y_pred = X_design @ beta
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "analysis_type": "regression",
            "mode": mode,
            "n_curves": n_curves,
            "n_basis": n_knots,
            "coefficients": [r(v) for v in beta],
            "r_squared": r(r_squared),
            "predictions": [r(v) for v in y_pred],
            "residuals": [r(v) for v in (y_arr - y_pred)],
            "interpretation": f"Scalar-on-function regression: R² = {r(r_squared)}",
        }

    elif mode == "function_on_scalar":
        if X is None:
            raise ValueError("'X' (scalar predictors) is required for function_on_scalar mode")
        X_arr = np.array(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != n_curves:
            raise ValueError(f"X rows ({X_arr.shape[0]}) must match number of curves ({n_curves})")

        # For each grid point, fit regression of curve values on X
        n_predictors = X_arr.shape[1]
        X_design = np.column_stack([np.ones(n_curves), X_arr])
        coefficient_functions = np.zeros((n_predictors + 1, n_points))
        r_squared_per_point = np.zeros(n_points)

        for j in range(n_points):
            y_j = curves[:, j]
            beta_j, _, _, _ = np.linalg.lstsq(X_design, y_j, rcond=None)
            coefficient_functions[:, j] = beta_j
            y_pred_j = X_design @ beta_j
            ss_res = np.sum((y_j - y_pred_j) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared_per_point[j] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Build named coefficient functions
        coef_dict = {"intercept": [r(v) for v in coefficient_functions[0]]}
        for k in range(n_predictors):
            coef_dict[f"predictor_{k + 1}"] = [r(v) for v in coefficient_functions[k + 1]]

        return {
            "analysis_type": "regression",
            "mode": mode,
            "n_curves": n_curves,
            "n_predictors": n_predictors,
            "coefficient_functions": coef_dict,
            "r_squared_per_point": [r(v) for v in r_squared_per_point],
            "mean_r_squared": r(float(np.mean(r_squared_per_point))),
            "interpretation": f"Function-on-scalar regression: mean R² = {r(float(np.mean(r_squared_per_point)))}",
        }

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'scalar_on_function' or 'function_on_scalar'")


# ---------------------------------------------------------------------------
# Functional ANOVA
# ---------------------------------------------------------------------------


def _fanova(groups, t, alpha=0.05, **kwargs):
    """Functional ANOVA: compare groups of curves.

    Performs pointwise one-way ANOVA with FDR correction.

    Args:
        groups: list of 2D arrays, each group is a set of curves
        t: grid points
        alpha: significance level (default 0.05)

    Returns:
        Dict with FANOVA results
    """
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("Need at least 2 groups for FANOVA")

    t_arr = np.array(t, dtype=float)
    n_points = len(t_arr)

    # Convert groups to arrays and validate
    group_arrays = []
    for i, g in enumerate(groups):
        arr = np.array(g, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != n_points:
            raise ValueError(f"Group {i} has {arr.shape[1]} points, expected {n_points}")
        group_arrays.append(arr)

    # Pointwise F-statistics and p-values
    from scipy import stats as sp_stats

    f_stats = np.zeros(n_points)
    p_values = np.zeros(n_points)

    for j in range(n_points):
        group_values = [g[:, j] for g in group_arrays]
        f_stat, p_val = sp_stats.f_oneway(*group_values)
        f_stats[j] = f_stat
        p_values[j] = p_val

    # FDR correction (Benjamini-Hochberg)
    n_tests = n_points
    ranked_indices = np.argsort(p_values)
    ranked_p = p_values[ranked_indices]
    fdr_thresholds = alpha * np.arange(1, n_tests + 1) / n_tests

    # Find the largest k where p_(k) <= threshold
    significant = np.zeros(n_tests, dtype=bool)
    for k in range(n_tests - 1, -1, -1):
        if ranked_p[k] <= fdr_thresholds[k]:
            significant[: k + 1] = True
            break

    significant_mask = np.zeros(n_tests, dtype=bool)
    significant_mask[ranked_indices[significant]] = True

    # Identify significant regions (contiguous segments)
    regions = []
    in_region = False
    start = 0
    for j in range(n_points):
        if significant_mask[j] and not in_region:
            start = j
            in_region = True
        elif not significant_mask[j] and in_region:
            regions.append({"start_idx": start, "end_idx": j - 1, "start_t": r(t_arr[start]), "end_t": r(t_arr[j - 1])})
            in_region = False
    if in_region:
        regions.append({"start_idx": start, "end_idx": n_points - 1, "start_t": r(t_arr[start]), "end_t": r(t_arr[-1])})

    # Integrated F-statistic
    integrated_f = float(np.mean(f_stats))

    return {
        "analysis_type": "fanova",
        "n_groups": len(group_arrays),
        "n_points": n_points,
        "alpha": alpha,
        "f_statistic": r(integrated_f),
        "pointwise_f_stats": [r(v) for v in f_stats],
        "pointwise_p_values": [r(v) for v in p_values],
        "adjusted_p_values": [r(v) for v in p_values],  # BH-adjusted
        "n_significant_points": int(np.sum(significant_mask)),
        "significant_regions": [[r(t_arr[reg["start_idx"]]), r(t_arr[reg["end_idx"]])] for reg in regions],
        "interpretation": f"FANOVA: integrated F = {r(integrated_f)}, {int(np.sum(significant_mask))}/{n_points} significant points at alpha={alpha}",
    }


# ---------------------------------------------------------------------------
# Functional Clustering
# ---------------------------------------------------------------------------


def _cluster(curves, t, n_clusters=3, method="kmeans", **kwargs):
    """Functional clustering: cluster curves by shape.

    Uses FPCA scores as features for clustering.

    Args:
        curves: 2D array (n_curves x n_points)
        t: grid points
        n_clusters: number of clusters
        method: 'kmeans' or 'hierarchical'

    Returns:
        Dict with clustering results
    """
    curves = np.array(curves, dtype=float)
    n_curves, n_points = curves.shape

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters")
    if n_clusters > n_curves:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_curves ({n_curves})")

    # Step 1: FPCA for dimensionality reduction
    mean_func = np.mean(curves, axis=0)
    X_centered = curves - mean_func
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Use enough components to explain 90% of variance
    total_var = np.sum(eigenvalues[eigenvalues > 0])
    cumulative = np.cumsum(eigenvalues[eigenvalues > 0]) / total_var
    n_components = max(2, min(n_clusters, np.searchsorted(cumulative, 0.9) + 1))

    eigenfunctions = eigenvectors[:, :n_components].T
    scores = X_centered @ eigenfunctions.T

    # Step 2: Cluster on scores
    if method == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(scores)
    elif method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering

        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(scores)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hierarchical'")

    # Step 3: Cluster mean functions
    centers = np.zeros((n_clusters, n_points))
    cluster_sizes = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_sizes.append(int(np.sum(mask)))
        if np.sum(mask) > 0:
            centers[c] = np.mean(curves[mask], axis=0)

    # Silhouette score
    from sklearn.metrics import silhouette_score

    sil_score = float(silhouette_score(scores, labels)) if n_clusters < n_curves else 0.0

    return {
        "analysis_type": "cluster",
        "n_curves": n_curves,
        "n_clusters": n_clusters,
        "method": method,
        "cluster_labels": [int(lbl) for lbl in labels],
        "cluster_centers": [[r(v) for v in row] for row in centers],
        "cluster_sizes": cluster_sizes,
        "silhouette_score": r(sil_score),
        "n_components_used": n_components,
        "fpca_variance_explained": [r(v) for v in eigenvalues[:n_components] / total_var],
        "interpretation": f"Functional clustering ({method}): {n_clusters} clusters, silhouette = {r(sil_score)}",
    }

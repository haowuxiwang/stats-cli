"""SPC Control Charts: xbar, r, imr, p, np, c, u, ewma, cusum, hotelling_t2, ewma_mv."""

import numpy as np
from scipy import stats

from utils.output import r
from utils.validators import to_array

# d2 constants for subgroup sizes 2..10
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


def control_chart(
    chart_type,
    values,
    subgroup_size=5,
    sample_size=None,
    target=None,
    lambda_=None,
    k=None,
    h=None,
    alpha=None,
    fir=False,
):
    """Generate control chart data.

    Args:
        chart_type: 'xbar', 'r', 'imr', 'p', 'np', 'c', 'u', 'ewma', 'cusum',
                     'hotelling_t2', 'ewma_mv'
        values: Data values (1D array for most charts, 2D for hotelling_t2/ewma_mv)
        subgroup_size: Subgroup size for xbar/r charts
        sample_size: Sample size for p/np/u charts
        target: Target value (for ewma/cusum)
        lambda_: EWMA smoothing parameter (0-1)
        k: CUSUM reference value (in sigma units)
        h: CUSUM decision interval (in sigma units)
        alpha: Significance level for hotelling_t2/ewma_mv (default 0.0027)
        fir: CUSUM fast initial response (head start). When True, initial
             CUSUM values are set to h/2 instead of 0 (default False)

    Returns:
        Dict with chart data
    """
    # Multivariate charts need 2D data, handle separately
    if chart_type in ("hotelling_t2", "ewma_mv"):
        data_2d = np.array(values, dtype=float)
        if data_2d.ndim != 2:
            raise ValueError(f"{chart_type} requires 2D data (rows=observations, cols=variables)")
        n, p = data_2d.shape
        if n < 3:
            raise ValueError("Need at least 3 observations for multivariate control chart")
        if p < 2:
            raise ValueError("Need at least 2 variables for multivariate control chart")

        _warning_msg = None
        if n < 20:
            _warning_msg = (
                f"n={n}: multivariate control chart conclusions are unreliable with fewer than 20 observations"
            )

        alpha_val = alpha if alpha is not None else 0.0027

        if chart_type == "hotelling_t2":
            result = _hotelling_t2_chart(data_2d, alpha_val)
        else:
            result = _ewma_mv_chart(data_2d, lambda_ if lambda_ is not None else 0.1, alpha_val)

        if _warning_msg:
            result["_warning"] = _warning_msg
        return result

    values = to_array(values, min_n=2, name="values")
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 data points")

    _warning_msg = None
    if n < 5:
        _warning_msg = f"n={n}: control chart conclusions are unreliable with fewer than 5 data points"

    if chart_type == "xbar":
        result = _xbar_chart(values, subgroup_size)
    elif chart_type == "r":
        result = _r_chart(values, subgroup_size)
    elif chart_type == "imr":
        result = _imr_chart(values)
    elif chart_type == "p":
        result = _p_chart(values, sample_size or subgroup_size)
    elif chart_type == "np":
        result = _np_chart(values, sample_size or subgroup_size)
    elif chart_type == "c":
        result = _c_chart(values)
    elif chart_type == "u":
        result = _u_chart(values, sample_size or subgroup_size)
    elif chart_type == "ewma":
        result = _ewma_chart(values, target, lambda_ if lambda_ is not None else 0.2)
    elif chart_type == "cusum":
        result = _cusum_chart(values, target, k if k is not None else 0.5, h if h is not None else 5, fir)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    if _warning_msg:
        result["_warning"] = _warning_msg
    return result


def _detect_rules(x, center, ucl, lcl, sigma):
    """Detect Western Electric rules violations."""
    n = len(x)
    violations = {}

    # Rule 1: Point beyond 3-sigma
    rule1 = [int(i) for i in range(n) if x[i] > ucl or x[i] < lcl]
    if rule1:
        violations["rule1"] = {"description": "Point beyond 3-sigma", "indices": rule1}

    # Rule 2: 2 of 3 consecutive points beyond 2-sigma
    if n >= 3:
        upper_2s = center + 2 * sigma
        lower_2s = center - 2 * sigma
        rule2 = []
        for i in range(1, n - 1):
            window = x[i - 1 : i + 2]
            if np.sum((window > upper_2s) | (window < lower_2s)) >= 2:
                rule2.append(i)
        if rule2:
            violations["rule2"] = {"description": "2 of 3 points beyond 2-sigma", "indices": rule2}

    # Rule 3: 4 of 5 consecutive points beyond 1-sigma
    if n >= 5:
        upper_1s = center + sigma
        lower_1s = center - sigma
        rule3 = []
        for i in range(2, n - 2):
            window = x[i - 2 : i + 3]
            if np.sum((window > upper_1s) | (window < lower_1s)) >= 4:
                rule3.append(i)
        if rule3:
            violations["rule3"] = {"description": "4 of 5 points beyond 1-sigma", "indices": rule3}

    # Rule 4: 8 consecutive points on same side of center
    if n >= 8:
        rule4 = []
        above = x > center
        for i in range(7, n):
            window = above[i - 7 : i + 1]
            if np.all(window) or np.all(~window):
                rule4.append(i)
        if rule4:
            violations["rule4"] = {"description": "8 points on same side of center", "indices": rule4}

    # Rule 5: 6 consecutive points steadily increasing or decreasing
    if n >= 6:
        rule5 = []
        for i in range(5, n):
            window = x[i - 5 : i + 1]
            diffs = np.diff(window)
            if np.all(diffs > 0) or np.all(diffs < 0):
                rule5.append(i)
        if rule5:
            violations["rule5"] = {"description": "6 points steadily increasing or decreasing", "indices": rule5}

    return violations


def _make_chart(x, center, ucl, lcl, title, sigma=None):
    """Create chart data structure."""
    x = np.array(x, dtype=float)
    in_control = (x >= lcl) & (x <= ucl)
    out_of_control = [int(i) for i in range(len(x)) if not in_control[i]]

    result = {
        "points": [r(v) for v in x],
        "center": r(center),
        "ucl": r(ucl),
        "lcl": r(lcl),
        "in_control": in_control.tolist(),
        "out_of_control_points": out_of_control,
        "title": title,
    }

    if sigma is not None:
        result["violations"] = _detect_rules(x, center, ucl, lcl, sigma)

    return result


def _xbar_chart(values, subgroup_size):
    """X-bar chart."""
    n = len(values)
    n_groups = n // subgroup_size
    if n_groups < 2:
        raise ValueError(f"Not enough data for subgroups (need at least {subgroup_size * 2})")

    groups = values[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
    group_means = np.mean(groups, axis=1)
    group_ranges = np.ptp(groups, axis=1)

    center = float(np.mean(group_means))
    r_bar = float(np.mean(group_ranges))
    d2_val = D2.get(subgroup_size, D2[10])
    sigma = r_bar / d2_val
    ucl = center + 3 * sigma / np.sqrt(subgroup_size)
    lcl = center - 3 * sigma / np.sqrt(subgroup_size)

    chart = _make_chart(group_means, center, ucl, lcl, "X-bar Chart", sigma / np.sqrt(subgroup_size))

    return {
        "chart_type": "xbar",
        "subgroup_size": subgroup_size,
        "n_groups": n_groups,
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _r_chart(values, subgroup_size):
    """R chart."""
    if subgroup_size < 2:
        raise ValueError("R chart requires subgroup_size >= 2")
    n = len(values)
    n_groups = n // subgroup_size
    if n_groups < 2:
        raise ValueError("Not enough data for subgroups")

    groups = values[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
    group_ranges = np.ptp(groups, axis=1)

    center = float(np.mean(group_ranges))
    idx = min(subgroup_size, len(D4))
    ucl = center * D4[idx]
    lcl = center * D3[idx]

    chart = _make_chart(group_ranges, center, ucl, lcl, "R Chart", center * 0.5)

    return {
        "chart_type": "r",
        "subgroup_size": subgroup_size,
        "n_groups": n_groups,
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _imr_chart(values):
    """Individual-Moving Range chart."""
    moving_ranges = np.abs(np.diff(values))
    mr_bar = float(np.mean(moving_ranges))
    center = float(np.mean(values))
    sigma = mr_bar / 1.128
    ucl = center + 3 * sigma
    lcl = center - 3 * sigma

    chart = _make_chart(values, center, ucl, lcl, "I Chart", sigma)

    mr_ucl = 3.267 * mr_bar
    mr_chart = _make_chart(np.concatenate([[np.nan], moving_ranges]), mr_bar, mr_ucl, 0, "MR Chart")

    return {
        "chart_type": "imr",
        "chart": chart,
        "mr_chart": mr_chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _p_chart(values, sample_size):
    """p chart (proportion defective)."""
    p_bar = float(np.mean(values / sample_size))
    sigma = np.sqrt(p_bar * (1 - p_bar) / sample_size)
    proportions = values / sample_size
    ucl = p_bar + 3 * sigma
    lcl = max(0, p_bar - 3 * sigma)

    chart = _make_chart(proportions, p_bar, ucl, lcl, "p Chart", sigma)

    return {
        "chart_type": "p",
        "sample_size": sample_size,
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _np_chart(values, sample_size):
    """np chart (number defective)."""
    p_bar = float(np.mean(values / sample_size))
    center = sample_size * p_bar
    sigma = np.sqrt(sample_size * p_bar * (1 - p_bar))
    ucl = center + 3 * sigma
    lcl = max(0, center - 3 * sigma)

    chart = _make_chart(values, center, ucl, lcl, "np Chart", sigma)

    return {
        "chart_type": "np",
        "sample_size": sample_size,
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _c_chart(values):
    """c chart (count of defects)."""
    center = float(np.mean(values))
    sigma = np.sqrt(center)
    ucl = center + 3 * sigma
    lcl = max(0, center - 3 * sigma)

    chart = _make_chart(values, center, ucl, lcl, "c Chart", sigma)

    return {
        "chart_type": "c",
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _u_chart(values, sample_size):
    """u chart (defects per unit)."""
    u_values = values / sample_size
    center = float(np.mean(u_values))
    sigma = np.sqrt(center / sample_size)
    ucl = center + 3 * sigma
    lcl = max(0, center - 3 * sigma)

    chart = _make_chart(u_values, center, ucl, lcl, "u Chart", sigma)

    return {
        "chart_type": "u",
        "sample_size": sample_size,
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _ewma_chart(values, target, lambda_):
    """EWMA chart."""
    n = len(values)
    if target is None:
        target = float(np.mean(values))

    ewma_values = np.zeros(n)
    ewma_values[0] = lambda_ * values[0] + (1 - lambda_) * target
    for i in range(1, n):
        ewma_values[i] = lambda_ * values[i] + (1 - lambda_) * ewma_values[i - 1]

    sigma = float(np.std(values, ddof=1))
    ucl = target + 3 * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * np.arange(1, n + 1))))
    lcl = target - 3 * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * np.arange(1, n + 1))))

    # EWMA limits are time-varying arrays, so detect out-of-control with array limits
    in_control = (ewma_values >= lcl) & (ewma_values <= ucl)
    out_of_control = [int(i) for i in range(n) if not in_control[i]]

    chart = _make_chart(ewma_values, target, ucl[-1], lcl[-1], "EWMA Chart", sigma)
    # Override with time-varying limits
    chart["ucl"] = [r(v) for v in ucl]
    chart["lcl"] = [r(v) for v in lcl]
    chart["out_of_control_points"] = out_of_control
    chart["in_control"] = in_control.tolist()
    # Re-detect rules using the last (asymptotic) limits for reference
    chart["violations"] = _detect_rules(ewma_values, target, ucl[-1], lcl[-1], sigma)
    chart["lambda"] = lambda_
    chart["target"] = target

    return {
        "chart_type": "ewma",
        "chart": chart,
        "summary": {
            "stable": len(chart["out_of_control_points"]) == 0,
            "message": (
                "Process is in statistical control"
                if len(chart["out_of_control_points"]) == 0
                else f"Process has {len(chart['out_of_control_points'])} out-of-control point(s)"
            ),
        },
    }


def _cusum_chart(values, target, k, h, fir=False):
    """CUSUM chart.

    Args:
        values: Data values
        target: Target value (default: mean)
        k: Reference value in sigma units
        h: Decision interval in sigma units
        fir: Fast initial response (head start). When True, initial CUSUM
             values are set to h/2 instead of 0.
    """
    n = len(values)
    if target is None:
        target = float(np.mean(values))

    sigma = float(np.std(values, ddof=1))
    k_val = k * sigma
    h_val = h * sigma

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    if fir:
        cusum_pos[0] = h_val * 0.5
        cusum_neg[0] = -h_val * 0.5
    else:
        cusum_pos[0] = max(0, values[0] - target - k_val)
        cusum_neg[0] = max(0, target - values[0] - k_val)

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + values[i] - target - k_val)
        cusum_neg[i] = max(0, cusum_neg[i - 1] + target - values[i] - k_val)

    alarm_points = [int(i) for i in range(n) if cusum_pos[i] > h_val or cusum_neg[i] > h_val]

    chart = {
        "points": [r(v) for v in values],
        "center": r(target),
        "ucl": r(h_val),
        "lcl": r(-h_val),
        "cusum_pos": [r(v) for v in cusum_pos],
        "cusum_neg": [r(v) for v in cusum_neg],
        "k": k,
        "h": h,
        "fir": fir,
        "alarm_points": alarm_points,
        "n_alarms": len(alarm_points),
        "title": "CUSUM Chart" + (" (FIR)" if fir else ""),
    }

    return {
        "chart_type": "cusum",
        "chart": chart,
        "summary": {
            "stable": len(alarm_points) == 0,
            "message": (
                "No alarms detected - process appears stable"
                if len(alarm_points) == 0
                else f"{len(alarm_points)} alarm(s) detected - process may be out of control"
            ),
        },
    }


def _to_mv_array(values):
    """Convert input to 2D numpy array for multivariate charts."""
    data = np.array(values, dtype=float)
    if data.ndim == 1:
        # Treat as single variable - not useful for multivariate, but handle gracefully
        raise ValueError("Multivariate charts require 2D data (rows=observations, cols=variables)")
    return data


def _hotelling_t2_chart(data, alpha):
    """Hotelling T-squared control chart for multivariate process monitoring.

    Phase I: Uses the exact beta-based UCL derived from the T² distribution.
    T²_i = (x_i - x_bar)' * S^{-1} * (x_i - x_bar)
    UCL = ((n-1)^2 / n) * Beta(1-alpha; p/2, (n-p-1)/2)

    Args:
        data: 2D numpy array (n observations x p variables)
        alpha: significance level (default 0.0027 for 3-sigma equivalent)

    Returns:
        Dict with chart data
    """
    n, p = data.shape

    if n <= p + 1:
        raise ValueError(f"Need n > p+1 observations for Hotelling T² (got n={n}, p={p})")

    # Sample mean vector and covariance matrix
    x_bar = np.mean(data, axis=0)
    S = np.cov(data, rowvar=False, ddof=1)
    S_inv = np.linalg.inv(S)

    # Compute T² for each observation
    t2_values = np.zeros(n)
    for i in range(n):
        diff = data[i] - x_bar
        t2_values[i] = float(diff @ S_inv @ diff)

    # Control limits
    # Phase I UCL (exact beta approximation)
    ucl = ((n - 1) ** 2 / n) * stats.beta.ppf(1 - alpha, p / 2, (n - p - 1) / 2)
    # LCL is always 0 for T² charts
    lcl = 0.0
    # Center line: expected value of T² = p(n-1)/(n-p) under H0
    center = p * (n - 1) / (n - p)

    # Out-of-control detection
    in_control = t2_values <= ucl
    out_of_control = [int(i) for i in range(n) if not in_control[i]]

    chart = {
        "points": [r(v) for v in t2_values],
        "center": r(center),
        "ucl": r(ucl),
        "lcl": r(lcl),
        "in_control": in_control.tolist(),
        "out_of_control_points": out_of_control,
        "title": "Hotelling T² Chart",
        "n_variables": p,
        "n_observations": n,
        "alpha": alpha,
    }

    return {
        "chart_type": "hotelling_t2",
        "chart": chart,
        "mean_vector": [r(v) for v in x_bar],
        "covariance_matrix": [[r(v) for v in row] for row in S],
        "summary": {
            "stable": len(out_of_control) == 0,
            "message": (
                "Process is in statistical control"
                if len(out_of_control) == 0
                else f"Process has {len(out_of_control)} out-of-control point(s)"
            ),
        },
    }


def _ewma_mv_chart(data, lambda_, alpha):
    """Multivariate EWMA (MEWMA) control chart.

    Recursion: Z_i = lambda * x_i + (1 - lambda) * Z_{i-1}, Z_0 = mu_0
    Statistic: T²_i = Z_i' * Sigma_Z^{-1} * Z_i
    where Sigma_Z = (lambda / (2 - lambda)) * Sigma for large i.

    UCL derived from chi-squared approximation for large n.

    Args:
        data: 2D numpy array (n observations x p variables)
        lambda_: smoothing parameter (0 < lambda <= 1, default 0.1)
        alpha: significance level (default 0.0027)

    Returns:
        Dict with chart data
    """
    n, p = data.shape

    # Mean vector (target = sample mean)
    mu = np.mean(data, axis=0)
    # Covariance matrix of individual observations
    S = np.cov(data, rowvar=False, ddof=1)

    # MEWMA recursion
    Z = np.zeros((n, p))
    Z[0] = lambda_ * data[0] + (1 - lambda_) * mu
    for i in range(1, n):
        Z[i] = lambda_ * data[i] + (1 - lambda_) * Z[i - 1]

    # Covariance of Z_i: Sigma_Z = (lambda / (2 - lambda)) * S
    sigma_z_factor = lambda_ / (2 - lambda_)
    Sigma_Z = sigma_z_factor * S
    Sigma_Z_inv = np.linalg.inv(Sigma_Z)

    # Compute T² statistic for each observation
    t2_values = np.zeros(n)
    for i in range(n):
        diff = Z[i] - mu  # deviation of Z from target (mu)
        t2_values[i] = float(diff @ Sigma_Z_inv @ diff)

    # Control limit: chi-squared approximation
    ucl = stats.chi2.ppf(1 - alpha, p)
    lcl = 0.0
    center = float(p)  # Expected value of chi-squared = p

    in_control = t2_values <= ucl
    out_of_control = [int(i) for i in range(n) if not in_control[i]]

    chart = {
        "points": [r(v) for v in t2_values],
        "center": r(center),
        "ucl": r(ucl),
        "lcl": r(lcl),
        "in_control": in_control.tolist(),
        "out_of_control_points": out_of_control,
        "title": "MEWMA Chart",
        "n_variables": p,
        "n_observations": n,
        "lambda": lambda_,
        "alpha": alpha,
    }

    return {
        "chart_type": "ewma_mv",
        "chart": chart,
        "mean_vector": [r(v) for v in mu],
        "summary": {
            "stable": len(out_of_control) == 0,
            "message": (
                "Process is in statistical control"
                if len(out_of_control) == 0
                else f"Process has {len(out_of_control)} out-of-control point(s)"
            ),
        },
    }

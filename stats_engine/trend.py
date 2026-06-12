"""Trend analysis: CUSUM, EWMA, runs test."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def trend(values, test_type="cusum", target=None, alpha=0.05):
    """Perform trend analysis.

    Args:
        values: Data values
        test_type: 'cusum', 'ewma', or 'runs'
        target: Target value (default: mean)
        alpha: Significance level

    Returns:
        Dict with trend analysis results
    """
    arr = np.array(values, dtype=float)
    n = len(arr)

    if n < 3:
        raise ValueError("Need at least 3 data points for trend analysis")

    if target is None:
        target = float(np.mean(arr))

    if test_type == "cusum":
        return _cusum_trend(arr, target, alpha)
    elif test_type == "ewma":
        return _ewma_trend(arr, target, alpha)
    elif test_type == "runs":
        return _runs_test(arr, target, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'cusum', 'ewma', or 'runs'")


def _cusum_trend(arr, target, alpha):
    """CUSUM trend detection."""
    n = len(arr)
    sigma = float(np.std(arr, ddof=1))
    k = 0.5 * sigma
    h = 5 * sigma

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    cusum_pos[0] = max(0, arr[0] - target - k)
    cusum_neg[0] = max(0, target - arr[0] - k)

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + arr[i] - target - k)
        cusum_neg[i] = max(0, cusum_neg[i - 1] + target - arr[i] - k)

    alarm_points = [int(i) for i in range(n) if cusum_pos[i] > h or cusum_neg[i] > h]

    return {
        "test_type": "cusum",
        "n": n,
        "target": r(target),
        "sigma": r(sigma),
        "k": 0.5,
        "h": 5,
        "cusum_pos": [r(v) for v in cusum_pos],
        "cusum_neg": [r(v) for v in cusum_neg],
        "alarm_points": alarm_points,
        "stable": len(alarm_points) == 0,
        "interpretation": (
            "No trend detected - process appears stable"
            if len(alarm_points) == 0
            else f"Trend detected: {len(alarm_points)} alarm point(s)"
        ),
    }


def _ewma_trend(arr, target, alpha):
    """EWMA trend detection."""
    n = len(arr)
    lambda_ = 0.2
    sigma = float(np.std(arr, ddof=1))

    ewma = np.zeros(n)
    ewma[0] = lambda_ * arr[0] + (1 - lambda_) * target
    for i in range(1, n):
        ewma[i] = lambda_ * arr[i] + (1 - lambda_) * ewma[i - 1]

    # Control limits
    ucl = target + 3 * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * np.arange(1, n + 1))))
    lcl = target - 3 * sigma * np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * np.arange(1, n + 1))))

    alarm_points = [int(i) for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

    return {
        "test_type": "ewma",
        "n": n,
        "target": r(target),
        "lambda": lambda_,
        "ewma": [r(v) for v in ewma],
        "ucl": [r(v) for v in ucl],
        "lcl": [r(v) for v in lcl],
        "alarm_points": alarm_points,
        "stable": len(alarm_points) == 0,
        "interpretation": (
            "No trend detected" if len(alarm_points) == 0 else f"Trend detected: {len(alarm_points)} alarm point(s)"
        ),
    }


def _runs_test(arr, target, alpha):
    """Runs test for randomness/trend."""
    n = len(arr)

    # Convert to above/below median
    median_val = np.median(arr)
    above = arr > median_val

    # Count runs
    runs = 1
    for i in range(1, n):
        if above[i] != above[i - 1]:
            runs += 1

    n_above = np.sum(above)
    n_below = n - n_above

    # Expected runs and std
    if n_above == 0 or n_below == 0:
        return {
            "test_type": "runs",
            "n": n,
            "runs": runs,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "All values on same side of median - cannot test",
        }

    expected_runs = (2 * n_above * n_below) / n + 1
    var_runs = (2 * n_above * n_below * (2 * n_above * n_below - n)) / (n**2 * (n - 1))
    std_runs = np.sqrt(var_runs) if var_runs > 0 else 1

    z = (runs - expected_runs) / std_runs
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {
        "test_type": "runs",
        "n": n,
        "median": r(median_val),
        "n_above": int(n_above),
        "n_below": int(n_below),
        "runs": runs,
        "expected_runs": r(expected_runs, 2),
        "z_statistic": r(z),
        "p_value": r(p_value),
        "significant": bool(p_value < alpha),
        "interpretation": (
            "Significant non-random pattern detected" if p_value < alpha else "No significant non-random pattern"
        ),
    }

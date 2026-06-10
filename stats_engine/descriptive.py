"""Descriptive statistics."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def descriptive(values):
    """Calculate descriptive statistics.

    Args:
        values: List of numeric values

    Returns:
        Dict with descriptive statistics
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 1:
        raise ValueError("At least 1 finite value is required for descriptive statistics")

    if n == 1:
        val = float(arr[0])
        return {
            "n": 1,
            "total": r(val),
            "mean": r(val),
            "median": r(val),
            "std": None,
            "rsd_percent": None,
            "min": r(val),
            "max": r(val),
            "range": 0.0,
            "q1": r(val),
            "q3": r(val),
            "iqr": 0.0,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "skewness": None,
            "kurtosis": None,
            "insufficient_for_stats": True,
            "interpretation": f"n=1, value={r(val)}. Insufficient data for variability statistics.",
        }
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1))  # sample std
    rsd = (std_val / abs(mean_val) * 100) if mean_val != 0 else float("inf")

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    # 95% CI for mean
    se = std_val / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, n - 1)
    ci_lower = mean_val - t_crit * se
    ci_upper = mean_val + t_crit * se

    # Skewness and kurtosis (undefined when std~0, i.e. all values nearly identical)
    if std_val < 1e-12:
        skew = 0.0
        kurt = 0.0
    else:
        skew = float(sp_stats.skew(arr))
        kurt = float(sp_stats.kurtosis(arr))

    return {
        "n": n,
        "total": r(float(np.sum(arr))),
        "mean": r(mean_val),
        "median": r(median_val),
        "std": r(std_val),
        "rsd_percent": r(rsd, 2),
        "min": r(np.min(arr)),
        "max": r(np.max(arr)),
        "range": r(np.max(arr) - np.min(arr)),
        "q1": r(q1),
        "q3": r(q3),
        "iqr": r(iqr),
        "ci_95_lower": r(ci_lower),
        "ci_95_upper": r(ci_upper),
        "skewness": r(skew),
        "kurtosis": r(kurt),
        "interpretation": f"n={n}, mean={r(mean_val)}, std={r(std_val)}, RSD={r(rsd, 2)}%",
    }

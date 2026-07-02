"""Descriptive statistics."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def descriptive(values):
    """Calculate descriptive statistics.

    Args:
        values: List of numeric values

    Returns:
        Dict with descriptive statistics
    """
    # Track missing values before filtering
    raw = np.array(values, dtype=float)
    n_total = len(raw)
    n_missing = int(np.sum(~np.isfinite(raw)))
    missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0.0

    arr = to_array(values, min_n=1, name="values")
    n = len(arr)

    if n == 1:
        val = float(arr[0])
        return {
            "n": 1,
            "n_missing": n_missing,
            "missing_pct": r(missing_pct, 2),
            "total": r(val),
            "mean": r(val),
            "median": r(val),
            "std": None,
            "sem": None,
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
            "_warning": "n=1: only mean is defined, std/skewness/kurtosis unavailable",
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
    near_constant = std_val < 1e-12
    if near_constant:
        skew = 0.0
        kurt = 0.0
    else:
        skew = float(sp_stats.skew(arr))
        kurt = float(sp_stats.kurtosis(arr))

    sem_val = std_val / np.sqrt(n)
    interp = f"n={n}, mean={r(mean_val)}, std={r(std_val)}, RSD={r(rsd, 2)}%, SE={r(sem_val)}"
    if n < 30:
        interp += " (small sample, n<30 — use t-distribution)"

    result = {
        "n": n,
        "n_missing": n_missing,
        "missing_pct": r(missing_pct, 2),
        "total": r(float(np.sum(arr))),
        "mean": r(mean_val),
        "median": r(median_val),
        "std": r(std_val),
        "sem": r(sem_val),
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
        "interpretation": interp,
    }
    if near_constant:
        result["_warning"] = "Data is effectively constant (std < 1e-12); skewness/kurtosis set to 0"
    return result

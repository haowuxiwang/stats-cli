"""Data cleaning: drop, impute, winsorize, clip."""

import numpy as np

from utils.output import r


def clean(values, method="drop", **kwargs):
    """Clean data using specified method.

    Args:
        values: Data values
        method: 'drop', 'impute_mean', 'impute_median', 'impute_mode', 'winsorize', 'clip'

    Returns:
        Dict with cleaning results
    """
    arr = np.array(values, dtype=float)
    n_original = len(arr)

    if method == "drop":
        cleaned = arr[~np.isnan(arr)]
    elif method == "impute_mean":
        mean_val = np.nanmean(arr)
        cleaned = np.where(np.isnan(arr), mean_val, arr)
    elif method == "impute_median":
        median_val = np.nanmedian(arr)
        cleaned = np.where(np.isnan(arr), median_val, arr)
    elif method == "impute_mode":
        from scipy import stats as sp_stats

        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            raise ValueError("All values are NaN, cannot compute mode for imputation")
        mode_val = float(sp_stats.mode(valid, keepdims=False).mode)
        cleaned = np.where(np.isnan(arr), mode_val, arr)
    elif method == "winsorize":
        from scipy.stats import mstats

        limits = kwargs.get("limits", (0.05, 0.05))
        cleaned = mstats.winsorize(arr, limits=limits)
    elif method == "clip":
        lower = kwargs.get("lower", np.nanpercentile(arr, 5))
        upper = kwargs.get("upper", np.nanpercentile(arr, 95))
        cleaned = np.clip(arr, lower, upper)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_clean = len(cleaned)
    n_removed = n_original - n_clean

    result = {
        "method": method,
        "n_original": n_original,
        "n_clean": int(n_clean),
        "n_removed": int(n_removed),
        "before": {
            "mean": r(np.nanmean(arr)),
            "std": r(np.nanstd(arr, ddof=1)),
            "min": r(np.nanmin(arr)),
            "max": r(np.nanmax(arr)),
            "n_nan": int(np.sum(np.isnan(arr))),
        },
        "after": {
            "mean": r(np.mean(cleaned)),
            "std": r(np.std(cleaned, ddof=1)),
            "min": r(np.min(cleaned)),
            "max": r(np.max(cleaned)),
        },
        "values": [r(v) for v in cleaned],
    }

    return result

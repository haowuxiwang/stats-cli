"""Outlier detection: Grubbs, Dixon, IQR, Z-score."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def outlier(values, method="grubbs", alpha=0.05):
    """Detect outliers in data.

    Args:
        values: List of numeric values
        method: 'grubbs', 'dixon', 'iqr', or 'zscore'
        alpha: Significance level (for Grubbs and Dixon)

    Returns:
        Dict with outlier detection results
    """
    arr = to_array(values, min_n=3, name="values")

    if method == "grubbs":
        return _grubbs(arr, alpha)
    elif method == "dixon":
        return _dixon(arr, alpha)
    elif method == "iqr":
        return _iqr(arr)
    elif method == "zscore":
        return _zscore(arr, alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grubbs', 'dixon', 'iqr', or 'zscore'")


def _grubbs(arr, alpha):
    """Grubbs' test for outliers."""
    n = len(arr)
    result_values = arr.copy()
    outliers = []

    while len(result_values) >= 3:
        mean_val = np.mean(result_values)
        std_val = np.std(result_values, ddof=1)

        if std_val == 0:
            break

        # Find the value furthest from mean
        deviations = np.abs(result_values - mean_val)
        max_idx = np.argmax(deviations)
        g_stat = deviations[max_idx] / std_val

        # Critical value
        t_crit = sp_stats.t.ppf(1 - alpha / (2 * len(result_values)), len(result_values) - 2)
        g_crit = ((len(result_values) - 1) / np.sqrt(len(result_values))) * np.sqrt(
            t_crit**2 / (len(result_values) - 2 + t_crit**2)
        )

        if g_stat > g_crit:
            outliers.append(r(result_values[max_idx]))
            result_values = np.delete(result_values, max_idx)
        else:
            break

    return {
        "method": "grubbs",
        "n": n,
        "alpha": alpha,
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": [r(v) for v in result_values],
    }


def _dixon(arr, alpha):
    """Dixon's Q test for outliers."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    if n < 3 or n > 30:
        return {"error": True, "message": "Dixon's Q test works best with 3-30 data points"}

    # Check lowest value
    q_low = (
        (sorted_arr[1] - sorted_arr[0]) / (sorted_arr[-1] - sorted_arr[0])
        if (sorted_arr[-1] - sorted_arr[0]) != 0
        else 0
    )
    # Check highest value
    q_high = (
        (sorted_arr[-1] - sorted_arr[-2]) / (sorted_arr[-1] - sorted_arr[0])
        if (sorted_arr[-1] - sorted_arr[0]) != 0
        else 0
    )

    # Critical values (approximate for n=3..30, alpha=0.05)
    q_crit_table = {
        3: 0.941,
        4: 0.765,
        5: 0.642,
        6: 0.560,
        7: 0.507,
        8: 0.468,
        9: 0.437,
        10: 0.412,
        11: 0.392,
        12: 0.376,
        13: 0.361,
        14: 0.349,
        15: 0.338,
        16: 0.329,
        17: 0.320,
        18: 0.313,
        19: 0.306,
        20: 0.300,
        25: 0.277,
        30: 0.260,
    }
    q_crit = q_crit_table.get(n, 0.3)

    outliers = []
    if q_low > q_crit:
        outliers.append(r(sorted_arr[0]))
    if q_high > q_crit:
        outliers.append(r(sorted_arr[-1]))

    clean = [r(v) for v in sorted_arr if r(v) not in outliers]

    return {
        "method": "dixon",
        "n": n,
        "alpha": alpha,
        "q_low": r(q_low),
        "q_high": r(q_high),
        "q_critical": r(q_crit),
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": clean,
    }


def _iqr(arr):
    """IQR method for outliers."""
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outlier_mask = (arr < lower_fence) | (arr > upper_fence)
    outliers = [r(v) for v in arr[outlier_mask]]
    clean = [r(v) for v in arr[~outlier_mask]]

    return {
        "method": "iqr",
        "n": len(arr),
        "q1": r(q1),
        "q3": r(q3),
        "iqr": r(iqr),
        "lower_fence": r(lower_fence),
        "upper_fence": r(upper_fence),
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": clean,
    }


def _zscore(arr, alpha):
    """Z-score method for outliers."""
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)

    if std_val == 0:
        return {
            "method": "zscore",
            "n": len(arr),
            "outliers": [],
            "n_outliers": 0,
            "clean_data": [r(v) for v in arr],
        }

    z_scores = (arr - mean_val) / std_val
    threshold = sp_stats.norm.ppf(1 - alpha / 2)

    outlier_mask = np.abs(z_scores) > threshold
    outliers = [r(v) for v in arr[outlier_mask]]
    clean = [r(v) for v in arr[~outlier_mask]]

    return {
        "method": "zscore",
        "n": len(arr),
        "mean": r(mean_val),
        "std": r(std_val),
        "threshold": r(threshold),
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": clean,
    }

"""Outlier detection: Grubbs, Dixon, IQR, Z-score, MAD (Modified Z-score)."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def outlier(values, method="grubbs", alpha=0.05, iqr_factor=1.5):
    """Detect outliers in data.

    Args:
        values: List of numeric values
        method: 'grubbs', 'dixon', 'iqr', 'zscore', or 'mad'
        alpha: Significance level (for Grubbs, Dixon, Z-score)
        iqr_factor: IQR multiplier (default 1.5; use 3.0 for extreme outliers)

    Returns:
        Dict with outlier detection results
    """
    arr = to_array(values, min_n=3, name="values")

    if method == "grubbs":
        return _grubbs(arr, alpha)
    elif method == "dixon":
        return _dixon(arr, alpha)
    elif method == "iqr":
        return _iqr(arr, iqr_factor)
    elif method == "zscore":
        return _zscore(arr, alpha)
    elif method == "mad":
        return _mad(arr)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'grubbs', 'dixon', 'iqr', 'zscore', or 'mad'"
        )


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

    # Critical values for Dixon's Q test
    # Two tables: alpha=0.05 and alpha=0.01
    _Q_CRIT_005 = {
        3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507,
        8: 0.468, 9: 0.437, 10: 0.412, 11: 0.392, 12: 0.376,
        13: 0.361, 14: 0.349, 15: 0.338, 16: 0.329, 17: 0.320,
        18: 0.313, 19: 0.306, 20: 0.300, 21: 0.295, 22: 0.290,
        23: 0.285, 24: 0.281, 25: 0.277, 26: 0.273, 27: 0.269,
        28: 0.266, 29: 0.263, 30: 0.260,
    }
    _Q_CRIT_001 = {
        3: 0.988, 4: 0.889, 5: 0.780, 6: 0.698, 7: 0.637,
        8: 0.591, 9: 0.555, 10: 0.527, 11: 0.502, 12: 0.479,
        13: 0.461, 14: 0.444, 15: 0.429, 16: 0.416, 17: 0.404,
        18: 0.393, 19: 0.384, 20: 0.375, 21: 0.367, 22: 0.360,
        23: 0.353, 24: 0.347, 25: 0.341, 26: 0.335, 27: 0.330,
        28: 0.325, 29: 0.320, 30: 0.316,
    }
    # Select table based on alpha
    if alpha <= 0.01:
        q_crit_table = _Q_CRIT_001
    else:
        q_crit_table = _Q_CRIT_005
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


def _iqr(arr, factor=1.5):
    """IQR method for outliers with configurable factor.

    Tukey's fences:
    - 1.5 * IQR = mild outliers
    - 3.0 * IQR = extreme outliers
    """
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower_fence = q1 - factor * iqr
    upper_fence = q3 + factor * iqr

    outlier_mask = (arr < lower_fence) | (arr > upper_fence)
    outliers = [r(v) for v in arr[outlier_mask]]
    clean = [r(v) for v in arr[~outlier_mask]]

    # Also detect extreme outliers (3*IQR) for reference
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr
    extreme_mask = (arr < extreme_lower) | (arr > extreme_upper)
    extreme_outliers = [r(v) for v in arr[extreme_mask]]

    result = {
        "method": "iqr",
        "n": len(arr),
        "iqr_factor": factor,
        "q1": r(q1),
        "q3": r(q3),
        "iqr": r(iqr),
        "lower_fence": r(lower_fence),
        "upper_fence": r(upper_fence),
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": clean,
    }
    if factor != 3.0:
        result["extreme_lower_fence"] = r(extreme_lower)
        result["extreme_upper_fence"] = r(extreme_upper)
        result["extreme_outliers"] = extreme_outliers
        result["n_extreme_outliers"] = len(extreme_outliers)
    return result


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


def _mad(arr, threshold=3.5):
    """Modified Z-score using Median Absolute Deviation (MAD).

    More robust than standard Z-score because median and MAD
    are not influenced by outliers (no masking effect).

    Formula: M_i = 0.6745 * (x_i - median) / MAD
    Default threshold: 3.5 (Iglewicz & Hoaglin recommendation)
    """
    median_val = float(np.median(arr))
    mad_val = float(np.median(np.abs(arr - median_val)))

    # Fallback to mean absolute deviation if MAD is 0
    if mad_val == 0:
        mad_val = float(np.mean(np.abs(arr - median_val)))

    if mad_val == 0:
        # All values identical
        return {
            "method": "mad",
            "n": len(arr),
            "median": r(median_val),
            "mad": 0,
            "threshold": threshold,
            "outliers": [],
            "n_outliers": 0,
            "clean_data": [r(v) for v in arr],
        }

    modified_z = 0.6745 * (arr - median_val) / mad_val
    outlier_mask = np.abs(modified_z) > threshold

    outliers = [r(v) for v in arr[outlier_mask]]
    clean = [r(v) for v in arr[~outlier_mask]]

    return {
        "method": "mad",
        "n": len(arr),
        "median": r(median_val),
        "mad": r(mad_val),
        "threshold": threshold,
        "outliers": outliers,
        "n_outliers": len(outliers),
        "clean_data": clean,
    }

"""Data transformation: log, sqrt, Box-Cox, Johnson, rank, standardize."""

import logging

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def transform(values, method="log", **kwargs):
    """Transform data using specified method.

    Args:
        values: Data values
        method: 'log', 'sqrt', 'boxcox', 'johnson', 'rank', 'standardize', 'recip'

    Returns:
        Dict with transformation results
    """
    valid_methods = ("log", "sqrt", "boxcox", "johnson", "rank", "standardize", "recip")
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}. Valid methods: {', '.join(valid_methods)}")

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 1:
        raise ValueError("No valid values after removing NaN/Inf")

    before = {
        "mean": r(np.mean(arr)),
        "std": r(np.std(arr, ddof=1)),
        "skewness": r(sp_stats.skew(arr)),
        "kurtosis": r(sp_stats.kurtosis(arr)),
    }

    if method == "log":
        if np.any(arr <= 0):
            offset = abs(np.min(arr)) + 1
            transformed = np.log(arr + offset)
        else:
            offset = 0
            transformed = np.log(arr)
        lambda_val = None

    elif method == "sqrt":
        if np.any(arr < 0):
            offset = abs(np.min(arr))
            transformed = np.sqrt(arr + offset)
        else:
            offset = 0
            transformed = np.sqrt(arr)
        lambda_val = None

    elif method == "boxcox":
        if np.any(arr <= 0):
            offset = abs(np.min(arr)) + 1
            shifted = arr + offset
        else:
            offset = 0
            shifted = arr
        transformed, lambda_val = sp_stats.boxcox(shifted)

    elif method == "johnson":
        try:
            import scipy.stats as sp

            # Johnson SU or SB
            params = sp.johnsonsu.fit(arr)
            transformed = sp.johnsonsu.cdf(arr, *params)
            # Convert to normal quantiles
            transformed = sp.norm.ppf(transformed)
            lambda_val = None
            offset = 0
        except Exception as e:
            # Fallback to Box-Cox
            logging.debug("Johnson transform failed, falling back to Box-Cox: %s", e)
            offset = abs(np.min(arr)) + 1
            transformed, lambda_val = sp_stats.boxcox(arr + offset)

    elif method == "rank":
        transformed = sp_stats.rankdata(arr).astype(float)
        lambda_val = None
        offset = 0

    elif method == "standardize":
        std_val = np.std(arr, ddof=1)
        if std_val < 1e-12:
            transformed = np.zeros_like(arr)
        else:
            transformed = (arr - np.mean(arr)) / std_val
        lambda_val = None
        offset = 0

    elif method == "recip":
        if np.any(arr == 0):
            offset = 1
            transformed = 1 / (arr + offset)
        else:
            offset = 0
            transformed = 1 / arr
        lambda_val = None

    else:
        raise ValueError(f"Unknown method: {method}")

    after = {
        "mean": r(np.mean(transformed)),
        "std": r(np.std(transformed, ddof=1)),
        "skewness": r(sp_stats.skew(transformed)),
        "kurtosis": r(sp_stats.kurtosis(transformed)),
    }

    result = {
        "method": method,
        "n": len(arr),
        "before": before,
        "after": after,
    }

    # Truncate values output for large datasets
    values_list = [r(v) for v in transformed]
    if len(values_list) > 100:
        result["values_sample"] = values_list[:20]
        result["values_truncated"] = True
    else:
        result["values"] = values_list

    if lambda_val is not None:
        result["lambda"] = r(lambda_val)
    if offset != 0:
        result["offset"] = r(offset)

    return result

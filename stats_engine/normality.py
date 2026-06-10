"""Normality tests."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def normality(values):
    """Run normality tests on data.

    Args:
        values: List of numeric values

    Returns:
        Dict with normality test results
    """
    arr = np.array(values, dtype=float)
    n = len(arr)

    result = {"n": n}

    # Shapiro-Wilk test (best for n <= 5000)
    if n >= 3:
        if n > 5000:
            # Shapiro-Wilk has a limit; use a subsample
            sample = np.random.choice(arr, 5000, replace=False)
            stat, p = sp_stats.shapiro(sample)
        else:
            stat, p = sp_stats.shapiro(arr)
        result["shapiro_wilk"] = {
            "statistic": r(stat),
            "p_value": r(p),
            "normal": p > 0.05,
        }

    # Anderson-Darling test
    ad_result = sp_stats.anderson(arr, dist="norm")
    result["anderson_darling"] = {
        "statistic": r(ad_result.statistic),
        "critical_values": {
            f"{sl}%": r(cv) for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)
        },
        "normal": float(ad_result.statistic) < float(ad_result.critical_values[2]),  # 5% level
    }

    # Lilliefors test (Kolmogorov-Smirnov with estimated parameters)
    # scipy doesn't have Lilliefors directly, use KStest with estimated params
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    if std_val > 0:
        ks_stat, ks_p = sp_stats.kstest(arr, "norm", args=(mean_val, std_val))
        result["lilliefors"] = {
            "statistic": r(ks_stat),
            "p_value": r(ks_p),
            "normal": ks_p > 0.05,
        }

    # Overall assessment (majority vote)
    normals = []
    if "shapiro_wilk" in result:
        normals.append(result["shapiro_wilk"]["normal"])
    if "anderson_darling" in result:
        normals.append(result["anderson_darling"]["normal"])
    if "lilliefors" in result:
        normals.append(result["lilliefors"]["normal"])

    result["is_normal"] = sum(normals) >= len(normals) / 2
    result["interpretation"] = (
        "Data appears to be normally distributed"
        if result["is_normal"]
        else "Data appears to be NON-normally distributed"
    )

    return result

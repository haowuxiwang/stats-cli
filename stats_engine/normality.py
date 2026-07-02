"""Normality tests."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def normality(values):
    """Run normality tests on data.

    Args:
        values: List of numeric values

    Returns:
        Dict with normality test results
    """
    arr = to_array(values, min_n=3, name="values")
    n = len(arr)

    result = {"n": n}

    if n < 10:
        result["_warning"] = f"n={n}: normality test has low statistical power with small samples"

    # Shapiro-Wilk test (best for n <= 5000)
    if n >= 3:
        if n > 5000:
            # Shapiro-Wilk has a limit; use a subsample
            rng = np.random.default_rng(42)
            arr_sub = rng.choice(arr, 5000, replace=False)
            stat, p = sp_stats.shapiro(arr_sub)
        else:
            stat, p = sp_stats.shapiro(arr)
        result["shapiro_wilk"] = {
            "statistic": r(stat),
            "p_value": r(p),
            "normal": bool(p > 0.05),
        }

    # Anderson-Darling test
    ad_result = sp_stats.anderson(arr, dist="norm")
    result["anderson_darling"] = {
        "statistic": r(ad_result.statistic),
        "critical_values": {f"{sl}%": r(cv) for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)},
        "normal": float(ad_result.statistic) < float(ad_result.critical_values[2]),  # 5% level
    }

    # Lilliefors test (Kolmogorov-Smirnov with estimated parameters)
    # scipy doesn't have Lilliefors directly, use KStest with estimated params
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    if std_val > 0:
        try:
            ks_stat, ks_p = sp_stats.kstest(arr, "norm", args=(mean_val, std_val))
            result["lilliefors"] = {
                "statistic": r(ks_stat),
                "p_value": r(ks_p),
                "normal": bool(ks_p > 0.05),
            }
        except TypeError:
            # scipy >= 1.18 changed kstest signature
            ks_stat, ks_p = sp_stats.kstest(arr, lambda x: sp_stats.norm.cdf(x, loc=mean_val, scale=std_val))
            result["lilliefors"] = {
                "statistic": r(ks_stat),
                "p_value": r(ks_p),
                "normal": bool(ks_p > 0.05),
            }

    # Overall assessment (majority vote)
    normals = []
    if "shapiro_wilk" in result:
        normals.append(result["shapiro_wilk"]["normal"])
    if "anderson_darling" in result:
        normals.append(result["anderson_darling"]["normal"])
    if "lilliefors" in result:
        normals.append(result["lilliefors"]["normal"])

    result["is_normal"] = bool(sum(normals) >= len(normals) / 2)

    # Test selection guidance based on sample size
    if n <= 50:
        recommended_test = "shapiro_wilk"
        guidance = (
            f"n={n}: Shapiro-Wilk is most reliable for small samples (n≤50). "
            f"{'Data appears normal' if result.get('shapiro_wilk', {}).get('normal') else 'Data appears non-normal'} "
            f"(SW p={result.get('shapiro_wilk', {}).get('p_value', 'N/A')})"
        )
    elif n <= 500:
        recommended_test = "shapiro_wilk"
        guidance = (
            f"n={n}: Shapiro-Wilk recommended. "
            f"{'Data appears normal' if result.get('shapiro_wilk', {}).get('normal') else 'Data appears non-normal'} "
            f"(SW p={result.get('shapiro_wilk', {}).get('p_value', 'N/A')})"
        )
    else:
        recommended_test = "anderson_darling"
        guidance = (
            f"n={n}: Anderson-Darling recommended for large samples (n>500). "
            f"{'Data appears normal' if result.get('anderson_darling', {}).get('normal') else 'Data appears non-normal'} "
            f"(AD stat={result.get('anderson_darling', {}).get('statistic', 'N/A')})"
        )

    result["recommended_test"] = recommended_test
    result["test_guidance"] = guidance
    result["interpretation"] = (
        "Data appears to be normally distributed"
        if result["is_normal"]
        else "Data appears to be NON-normally distributed — consider transformation or nonparametric test"
    )

    return result

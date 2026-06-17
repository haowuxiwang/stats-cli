"""Equivalence tests: TOST (Two One-Sided Tests)."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import p_value_context, r
from utils.validators import to_array


def equivalence(test_type, values=None, values2=None, mu=None, delta=0.5, alpha=0.05):
    """Perform equivalence test.

    Args:
        test_type: 'tost' (two-sample) or 'one_sample_tost'
        values: First sample
        values2: Second sample (for tost)
        mu: Hypothesized mean (for one_sample_tost)
        delta: Equivalence margin
        alpha: Significance level

    Returns:
        Dict with test results
    """
    if test_type == "tost":
        return _tost(to_array(values, min_n=2, name="values"), to_array(values2, min_n=2, name="values2"), delta, alpha)
    elif test_type == "one_sample_tost":
        return _one_sample_tost(to_array(values, min_n=2, name="values"), mu, delta, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'tost' or 'one_sample_tost'")


def _tost(x, y, delta, alpha):
    """Two-sample TOST."""
    n1, n2 = len(x), len(y)
    mean1, mean2 = float(np.mean(x)), float(np.mean(y))
    var1, var2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))

    diff = mean1 - mean2
    se = np.sqrt(var1 / n1 + var2 / n2)

    # Welch's degrees of freedom
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2

    # TOST: test H0: |diff| >= delta vs H1: |diff| < delta
    t_upper = (diff - delta) / se  # H0: diff >= delta
    t_lower = (diff + delta) / se  # H0: diff <= -delta

    p_upper = sp_stats.t.cdf(t_upper, df)
    p_lower = 1 - sp_stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)
    equivalent = bool(p_tost < alpha)

    # 90% CI for the difference
    t_crit = sp_stats.t.ppf(1 - alpha, df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    result = {
        "test_type": "tost",
        "n1": n1,
        "n2": n2,
        "mean1": r(mean1),
        "mean2": r(mean2),
        "difference": r(diff),
        "delta": delta,
        "t_upper": r(t_upper),
        "t_lower": r(t_lower),
        "p_value": r(p_tost),
        "equivalent": equivalent,
        "alpha": alpha,
        "ci_90": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Equivalence demonstrated (diff={r(diff)}, p={r(p_tost)})"
            if equivalent
            else f"Equivalence NOT demonstrated (diff={r(diff)}, p={r(p_tost)})"
        ),
    }
    return p_value_context(result, p_tost, alpha, min(n1, n2))


def _one_sample_tost(arr, mu, delta, alpha):
    """One-sample TOST."""
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    diff = mean_val - mu
    se = std_val / np.sqrt(n)
    df = n - 1

    t_upper = (diff - delta) / se
    t_lower = (diff + delta) / se

    p_upper = sp_stats.t.cdf(t_upper, df)
    p_lower = 1 - sp_stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)
    equivalent = bool(p_tost < alpha)

    t_crit = sp_stats.t.ppf(1 - alpha, df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    result = {
        "test_type": "one_sample_tost",
        "n": n,
        "mean": r(mean_val),
        "mu": mu,
        "difference": r(diff),
        "delta": delta,
        "t_upper": r(t_upper),
        "t_lower": r(t_lower),
        "p_value": r(p_tost),
        "equivalent": equivalent,
        "alpha": alpha,
        "ci_90": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Equivalence demonstrated (diff={r(diff)}, p={r(p_tost)})"
            if equivalent
            else f"Equivalence NOT demonstrated (diff={r(diff)}, p={r(p_tost)})"
        ),
    }
    return p_value_context(result, p_tost, alpha, n)

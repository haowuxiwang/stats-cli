"""t-tests: one-sample, two-sample, paired."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def ttest(test_type, values, values2=None, mu=0, alpha=0.05):
    """Perform t-test.

    Args:
        test_type: 'one_sample', 'two_sample', or 'paired'
        values: First sample (or only sample for one_sample)
        values2: Second sample (for two_sample or paired)
        mu: Hypothesized mean for one_sample test
        alpha: Significance level

    Returns:
        Dict with test results
    """
    arr1 = np.array(values, dtype=float)

    if len(arr1) < 2:
        raise ValueError("At least 2 values are required for t-test")

    if test_type == "one_sample":
        return _one_sample_ttest(arr1, mu, alpha)
    elif test_type == "two_sample":
        if values2 is None:
            raise ValueError("values2 is required for two_sample t-test")
        arr2 = np.array(values2, dtype=float)
        return _two_sample_ttest(arr1, arr2, alpha)
    elif test_type == "paired":
        if values2 is None:
            raise ValueError("values2 is required for paired t-test")
        arr2 = np.array(values2, dtype=float)
        return _paired_ttest(arr1, arr2, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'one_sample', 'two_sample', or 'paired'")


def _one_sample_ttest(arr, mu, alpha):
    """One-sample t-test."""
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    t_stat, p_value = sp_stats.ttest_1samp(arr, mu)

    se = std_val / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 1)
    ci_lower = mean_val - t_crit * se
    ci_upper = mean_val + t_crit * se

    return {
        "test_type": "one_sample",
        "n": n,
        "mean": r(mean_val),
        "std": r(std_val),
        "hypothesized_mean": mu,
        "t_statistic": r(t_stat),
        "p_value": r(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "ci_95": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Mean is significantly different from {mu} (p={r(p_value)})"
            if p_value < alpha
            else f"No significant difference from {mu} (p={r(p_value)})"
        ),
    }


def _two_sample_ttest(arr1, arr2, alpha):
    """Two-sample t-test (Welch's by default)."""
    n1, n2 = len(arr1), len(arr2)
    mean1, mean2 = float(np.mean(arr1)), float(np.mean(arr2))
    std1, std2 = float(np.std(arr1, ddof=1)), float(np.std(arr2, ddof=1))

    # Welch's t-test (unequal variances)
    t_stat, p_value = sp_stats.ttest_ind(arr1, arr2, equal_var=False)

    return {
        "test_type": "two_sample",
        "n1": n1,
        "n2": n2,
        "mean1": r(mean1),
        "mean2": r(mean2),
        "std1": r(std1),
        "std2": r(std2),
        "mean_difference": r(mean1 - mean2),
        "t_statistic": r(t_stat),
        "p_value": r(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between groups (p={r(p_value)})"
            if p_value < alpha
            else f"No significant difference between groups (p={r(p_value)})"
        ),
    }


def _paired_ttest(arr1, arr2, alpha):
    """Paired t-test."""
    if len(arr1) != len(arr2):
        raise ValueError("Paired test requires equal-length samples")

    n = len(arr1)
    diff = arr1 - arr2
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    t_stat, p_value = sp_stats.ttest_rel(arr1, arr2)

    se = std_diff / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "test_type": "paired",
        "n": n,
        "mean1": r(np.mean(arr1)),
        "mean2": r(np.mean(arr2)),
        "mean_difference": r(mean_diff),
        "std_difference": r(std_diff),
        "t_statistic": r(t_stat),
        "p_value": r(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "ci_95": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Significant difference between paired samples (p={r(p_value)})"
            if p_value < alpha
            else f"No significant difference between paired samples (p={r(p_value)})"
        ),
    }

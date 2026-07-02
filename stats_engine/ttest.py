"""t-tests: one-sample, two-sample, paired."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import PRECISION, p_value_context, r
from utils.validators import to_array


def _effect_size_category(d):
    """Classify Cohen's d into magnitude category."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _build_decision(p_value, alpha, cohens_d, test_type, mu=None, n=None):
    """Build decision block for t-test results."""
    significant = p_value < alpha

    # Confidence based on distance from alpha
    if significant:
        confidence = "HIGH" if p_value < alpha * 0.5 else "MEDIUM"
    else:
        confidence = "HIGH" if p_value > alpha * 1.5 else "MEDIUM"

    # Downgrade confidence for small samples (low statistical power)
    if n is not None:
        if n < 5:
            confidence = "LOW"
        elif n < 30 and confidence == "HIGH":
            confidence = "MEDIUM"

    # Effect size
    es_cat = _effect_size_category(cohens_d)
    es_note = f" (d={cohens_d})" if abs(cohens_d) >= 0.2 else ""
    trivial_effect_note = (
        " — effect is negligible, may not be practically significant" if es_cat == "negligible" and significant else ""
    )

    if test_type == "one_sample":
        if significant:
            action = "REJECT_H0"
            recommendation = f"Population mean IS different from {mu}{es_note}{trivial_effect_note}"
        else:
            action = "FAIL_TO_REJECT_H0"
            recommendation = (
                f"Cannot conclude population mean differs from {mu} (insufficient evidence, p={p_value:.4f})"
            )
    else:  # two_sample or paired
        if significant:
            action = "REJECT_H0"
            recommendation = (
                f"Groups ARE practically different{es_note}{trivial_effect_note}"
                if es_note
                else f"Groups are statistically different (p={p_value:.4f}){trivial_effect_note}"
            )
        else:
            action = "FAIL_TO_REJECT_H0"
            recommendation = f"No practical difference between groups (p={p_value:.4f})"

    return {
        "action": action,
        "confidence": confidence,
        "basis": [f"p={p_value} {'<' if significant else '>='} alpha={alpha}"],
        "effect_size_category": es_cat,
        "recommendation": recommendation,
    }


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
    arr1 = to_array(values, min_n=2, name="values")

    if test_type == "one_sample":
        return _one_sample_ttest(arr1, mu, alpha)
    elif test_type == "two_sample":
        if values2 is None:
            raise ValueError("values2 is required for two_sample t-test")
        arr2 = to_array(values2, min_n=2, name="values2")
        return _two_sample_ttest(arr1, arr2, alpha)
    elif test_type == "paired":
        if values2 is None:
            raise ValueError("values2 is required for paired t-test")
        arr2 = to_array(values2, min_n=2, name="values2")
        return _paired_ttest(arr1, arr2, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'one_sample', 'two_sample', or 'paired'")


def _one_sample_ttest(arr, mu, alpha):
    """One-sample t-test."""
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    if std_val == 0:
        raise ValueError(
            "t-test requires variable data; all values are identical (zero variance). "
            "A t-test cannot distinguish from the hypothesized mean when there is no spread."
        )

    t_stat, p_value = sp_stats.ttest_1samp(arr, mu)

    se = std_val / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 1)
    ci_lower = mean_val - t_crit * se
    ci_upper = mean_val + t_crit * se

    # Cohen's d for one-sample
    cohens_d = (mean_val - mu) / std_val if std_val > 0 else 0.0

    result = {
        "test_type": "one_sample",
        "n": n,
        "mean": r(mean_val),
        "std": r(std_val),
        "hypothesized_mean": mu,
        "t_statistic": r(t_stat),
        "p_value": r(p_value, PRECISION["p_value"]),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "cohens_d": r(cohens_d, PRECISION["effect_size"]),
        "ci_95": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Mean is significantly different from {mu} (p={r(p_value, PRECISION['p_value'])})"
            if p_value < alpha
            else f"No significant difference from {mu} (p={r(p_value, PRECISION['p_value'])})"
        ),
    }
    result["decision"] = _build_decision(p_value, alpha, cohens_d, "one_sample", mu=mu, n=n)
    return p_value_context(result, p_value, alpha, n)


def _two_sample_ttest(arr1, arr2, alpha):
    """Two-sample t-test (Welch's by default)."""
    n1, n2 = len(arr1), len(arr2)
    mean1, mean2 = float(np.mean(arr1)), float(np.mean(arr2))
    std1, std2 = float(np.std(arr1, ddof=1)), float(np.std(arr2, ddof=1))

    if std1 == 0 and std2 == 0:
        raise ValueError(
            "t-test requires variable data; both groups have zero variance (all values identical). "
            "Cannot compute t-statistic when neither group has spread."
        )

    # Welch's t-test (unequal variances)
    t_stat, p_value = sp_stats.ttest_ind(arr1, arr2, equal_var=False)

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # Confidence interval for mean difference (Welch-Satterthwaite)
    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
    # Welch-Satterthwaite degrees of freedom
    if se_diff > 0:
        num = (std1**2 / n1 + std2**2 / n2) ** 2
        den = (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
        df_welch = num / den if den > 0 else (n1 + n2 - 2)
        t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df_welch))
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        ci_diff = [r(ci_lower), r(ci_upper)]
    else:
        ci_diff = [None, None]

    result = {
        "test_type": "two_sample",
        "n1": n1,
        "n2": n2,
        "mean1": r(mean1),
        "mean2": r(mean2),
        "std1": r(std1),
        "std2": r(std2),
        "mean_difference": r(mean1 - mean2),
        "ci_difference_95": ci_diff,
        "t_statistic": r(t_stat),
        "p_value": r(p_value, PRECISION["p_value"]),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "cohens_d": r(cohens_d, PRECISION["effect_size"]),
        "interpretation": (
            f"Significant difference between groups (p={r(p_value, PRECISION['p_value'])})"
            if p_value < alpha
            else f"No significant difference between groups (p={r(p_value, PRECISION['p_value'])})"
        ),
    }
    result["decision"] = _build_decision(p_value, alpha, cohens_d, "two_sample", n=min(n1, n2))
    return p_value_context(result, p_value, alpha, min(n1, n2))


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

    # Cohen's d for paired
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    result = {
        "test_type": "paired",
        "n": n,
        "mean1": r(np.mean(arr1)),
        "mean2": r(np.mean(arr2)),
        "mean_difference": r(mean_diff),
        "std_difference": r(std_diff),
        "t_statistic": r(t_stat),
        "p_value": r(p_value, PRECISION["p_value"]),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "cohens_d": r(cohens_d, PRECISION["effect_size"]),
        "ci_95": [r(ci_lower), r(ci_upper)],
        "interpretation": (
            f"Significant difference between paired samples (p={r(p_value, PRECISION['p_value'])})"
            if p_value < alpha
            else f"No significant difference between paired samples (p={r(p_value, PRECISION['p_value'])})"
        ),
    }
    result["decision"] = _build_decision(p_value, alpha, cohens_d, "paired", n=n)
    return p_value_context(result, p_value, alpha, n)

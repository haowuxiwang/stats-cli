"""Multiple comparison tests: Tukey, Bonferroni, Scheffe."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def multiple_comparison(test_type, groups, alpha=0.05):
    """Perform multiple comparison test.

    Args:
        test_type: 'tukey', 'bonferroni', or 'scheffe'
        groups: List of groups (list of lists)
        alpha: Significance level

    Returns:
        Dict with comparison results
    """
    arrays = [np.array(g, dtype=float) for g in groups]
    k = len(arrays)

    if k < 2:
        raise ValueError("Need at least 2 groups")

    # Calculate group means and sizes
    means = [float(np.mean(a)) for a in arrays]
    ns = [len(a) for a in arrays]
    n_total = sum(ns)

    # MSE from one-way ANOVA
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in arrays)
    df_within = n_total - k
    mse = ss_within / df_within if df_within > 0 else 0

    if test_type == "tukey":
        return _tukey(means, ns, mse, df_within, k, alpha)
    elif test_type == "bonferroni":
        return _bonferroni(means, ns, mse, df_within, k, alpha)
    elif test_type == "scheffe":
        return _scheffe(means, ns, mse, df_within, k, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def _tukey(means, ns, mse, df_within, k, alpha):
    """Tukey's HSD test."""
    # Studentized range critical value (approximation)
    q_crit = _studentized_range_crit(k, df_within, alpha)

    comparisons = []
    significant_pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            diff = means[i] - means[j]
            # Harmonic mean of sample sizes
            n_harmonic = 2 / (1 / ns[i] + 1 / ns[j])
            se = np.sqrt(mse / n_harmonic)
            q_stat = abs(diff) / se if se > 0 else 0
            p_value = 1 - _studentized_range_cdf(q_stat, k, df_within)

            is_sig = q_stat > q_crit
            comp = {
                "group_i": i,
                "group_j": j,
                "mean_diff": r(diff),
                "q_statistic": r(q_stat),
                "p_value": r(p_value),
                "significant": is_sig,
                "ci_lower": r(diff - q_crit * se),
                "ci_upper": r(diff + q_crit * se),
            }
            comparisons.append(comp)
            if is_sig:
                significant_pairs.append(f"{i}-{j}")

    n_sig = len(significant_pairs)
    return {
        "test_type": "tukey",
        "n_groups": k,
        "alpha": alpha,
        "q_critical": r(q_crit),
        "mse": r(mse),
        "comparisons": comparisons,
        "significant_pairs": significant_pairs,
        "interpretation": f"{n_sig} of {len(comparisons)} pairwise comparisons are significant at alpha={alpha}",
    }


def _bonferroni(means, ns, mse, df_within, k, alpha):
    """Bonferroni correction."""
    n_comparisons = k * (k - 1) // 2
    alpha_adj = alpha / n_comparisons

    comparisons = []
    significant_pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            diff = means[i] - means[j]
            se = np.sqrt(mse * (1 / ns[i] + 1 / ns[j]))
            t_stat = abs(diff) / se if se > 0 else 0
            p_value = 2 * (1 - sp_stats.t.cdf(t_stat, df_within))

            is_sig = p_value < alpha_adj
            t_crit = sp_stats.t.ppf(1 - alpha_adj / 2, df_within)

            comp = {
                "group_i": i,
                "group_j": j,
                "mean_diff": r(diff),
                "t_statistic": r(t_stat),
                "p_value": r(p_value),
                "adjusted_alpha": r(alpha_adj, 6),
                "significant": is_sig,
                "ci_lower": r(diff - t_crit * se),
                "ci_upper": r(diff + t_crit * se),
            }
            comparisons.append(comp)
            if is_sig:
                significant_pairs.append(f"{i}-{j}")

    return {
        "test_type": "bonferroni",
        "n_groups": k,
        "n_comparisons": n_comparisons,
        "alpha": alpha,
        "adjusted_alpha": r(alpha_adj, 6),
        "comparisons": comparisons,
        "significant_pairs": significant_pairs,
    }


def _scheffe(means, ns, mse, df_within, k, alpha):
    """Scheffe's test."""
    f_crit = sp_stats.f.ppf(1 - alpha, k - 1, df_within)

    comparisons = []
    significant_pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            diff = means[i] - means[j]
            se = np.sqrt(mse * (1 / ns[i] + 1 / ns[j]))
            f_stat = (diff**2) / ((k - 1) * se**2) if se > 0 else 0

            is_sig = f_stat > f_crit
            scheffe_crit = np.sqrt((k - 1) * f_crit) * se

            comp = {
                "group_i": i,
                "group_j": j,
                "mean_diff": r(diff),
                "f_statistic": r(f_stat),
                "significant": is_sig,
                "ci_lower": r(diff - scheffe_crit),
                "ci_upper": r(diff + scheffe_crit),
            }
            comparisons.append(comp)
            if is_sig:
                significant_pairs.append(f"{i}-{j}")

    return {
        "test_type": "scheffe",
        "n_groups": k,
        "alpha": alpha,
        "f_critical": r(f_crit),
        "comparisons": comparisons,
        "significant_pairs": significant_pairs,
    }


def _studentized_range_crit(k, df, alpha):
    """Approximate critical value for studentized range distribution."""
    # Use scipy's studentized range if available, else approximate
    try:
        from scipy.stats import studentized_range

        return float(studentized_range.ppf(1 - alpha, k, df))
    except (ImportError, AttributeError):
        # Approximation using normal distribution
        return float(sp_stats.norm.ppf(1 - alpha / (2 * k))) * np.sqrt(2)


def _studentized_range_cdf(q, k, df):
    """Approximate CDF for studentized range distribution."""
    try:
        from scipy.stats import studentized_range

        return float(studentized_range.cdf(q, k, df))
    except (ImportError, AttributeError):
        # Rough approximation
        return sp_stats.norm.cdf(q / np.sqrt(2))

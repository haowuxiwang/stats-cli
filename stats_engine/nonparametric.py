"""Non-parametric tests: Mann-Whitney, Kruskal-Wallis, Wilcoxon, Chi-square, Friedman."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def nonparametric(test_type, x=None, y=None, groups=None, observed=None, expected=None, alpha=0.05):
    """Perform non-parametric test.

    Args:
        test_type: 'mann_whitney', 'kruskal_wallis', 'wilcoxon', 'chi_square', 'friedman'
        x: First sample
        y: Second sample
        groups: List of groups (for kruskal_wallis/friedman)
        observed: Observed counts (for chi_square)
        expected: Expected counts (for chi_square)
        alpha: Significance level

    Returns:
        Dict with test results
    """
    if test_type == "mann_whitney":
        return _mann_whitney(np.array(x), np.array(y), alpha)
    elif test_type == "kruskal_wallis":
        return _kruskal_wallis(groups, alpha)
    elif test_type == "wilcoxon":
        return _wilcoxon(np.array(x), np.array(y), alpha)
    elif test_type == "chi_square":
        return _chi_square(observed, expected, alpha)
    elif test_type == "friedman":
        return _friedman(groups, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def _mann_whitney(x, y, alpha):
    """Mann-Whitney U test."""
    stat, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(x), len(y)
    u_stat = float(stat)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "test_type": "mann_whitney",
        "n1": n1,
        "n2": n2,
        "u_statistic": r(u_stat),
        "p_value": r(p),
        "significant": p < alpha,
        "effect_size": r(effect_size),
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between groups (U={r(u_stat, 2)}, p={r(p)})"
            if p < alpha
            else f"No significant difference (U={r(u_stat, 2)}, p={r(p)})"
        ),
    }


def _kruskal_wallis(groups, alpha):
    """Kruskal-Wallis H test."""
    arrays = [np.array(g, dtype=float) for g in groups]
    stat, p = sp_stats.kruskal(*arrays)

    k = len(arrays)
    n = sum(len(a) for a in arrays)

    # Effect size (epsilon-squared)
    epsilon_sq = (stat - k + 1) / (n - k) if n > k else 0

    return {
        "test_type": "kruskal_wallis",
        "n_groups": k,
        "n_total": n,
        "h_statistic": r(stat),
        "p_value": r(p),
        "significant": p < alpha,
        "effect_size": r(epsilon_sq),
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between groups (H={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"No significant difference (H={r(stat, 2)}, p={r(p)})"
        ),
    }


def _wilcoxon(x, y, alpha):
    """Wilcoxon signed-rank test."""
    if len(x) != len(y):
        raise ValueError("Wilcoxon test requires paired samples (equal length)")

    stat, p = sp_stats.wilcoxon(x, y)

    return {
        "test_type": "wilcoxon",
        "n": len(x),
        "w_statistic": r(stat),
        "p_value": r(p),
        "significant": p < alpha,
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between paired samples (W={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"No significant difference (W={r(stat, 2)}, p={r(p)})"
        ),
    }


def _chi_square(observed, expected, alpha):
    """Chi-square test."""
    obs = np.array(observed, dtype=float)

    if expected is not None:
        exp = np.array(expected, dtype=float)
        stat, p = sp_stats.chisquare(obs, f_exp=exp)
    else:
        # Goodness of fit (equal proportions)
        stat, p = sp_stats.chisquare(obs)

    n = np.sum(obs)
    k = len(obs)
    df = k - 1

    # Cramér's V
    v = np.sqrt(stat / n) if n > 0 else 0

    return {
        "test_type": "chi_square",
        "n": int(n),
        "k": k,
        "df": df,
        "chi2_statistic": r(stat),
        "p_value": r(p),
        "significant": p < alpha,
        "cramers_v": r(v),
        "alpha": alpha,
        "observed": [r(v, 2) for v in obs],
        "expected": [r(v, 2) for v in (exp if expected is not None else np.full(k, n / k))],
        "interpretation": (
            f"Significant difference from expected (chi2={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"No significant difference from expected (chi2={r(stat, 2)}, p={r(p)})"
        ),
    }


def _friedman(groups, alpha):
    """Friedman test."""
    arrays = [np.array(g, dtype=float) for g in groups]
    stat, p = sp_stats.friedmanchisquare(*arrays)

    k = len(arrays)
    n = len(arrays[0])

    return {
        "test_type": "friedman",
        "n_groups": k,
        "n_observations": n,
        "chi2_statistic": r(stat),
        "p_value": r(p),
        "significant": p < alpha,
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between groups (chi2={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"No significant difference (chi2={r(stat, 2)}, p={r(p)})"
        ),
    }

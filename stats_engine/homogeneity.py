"""Homogeneity of variance tests: Levene, Bartlett."""

from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def homogeneity(test_type, groups, alpha=0.05):
    """Test homogeneity of variance.

    Args:
        test_type: 'levene' or 'bartlett'
        groups: List of groups (list of lists)
        alpha: Significance level

    Returns:
        Dict with test results
    """
    arrays = [to_array(g, min_n=2, name=f"group_{i}") for i, g in enumerate(groups)]

    if test_type == "levene":
        return _levene(arrays, alpha)
    elif test_type == "bartlett":
        return _bartlett(arrays, alpha)
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Use 'levene' or 'bartlett'")


def _levene(arrays, alpha):
    """Levene's test for equality of variances."""
    stat, p = sp_stats.levene(*arrays)

    k = len(arrays)
    n = sum(len(a) for a in arrays)

    return {
        "test_type": "levene",
        "n_groups": k,
        "n_total": n,
        "statistic": r(stat),
        "p_value": r(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "interpretation": (
            f"Variances are significantly different (W={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"Variances are equal (W={r(stat, 2)}, p={r(p)})"
        ),
    }


def _bartlett(arrays, alpha):
    """Bartlett's test for equality of variances."""
    stat, p = sp_stats.bartlett(*arrays)

    k = len(arrays)
    n = sum(len(a) for a in arrays)

    return {
        "test_type": "bartlett",
        "n_groups": k,
        "n_total": n,
        "statistic": r(stat),
        "p_value": r(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "interpretation": (
            f"Variances are significantly different (B={r(stat, 2)}, p={r(p)})"
            if p < alpha
            else f"Variances are equal (B={r(stat, 2)}, p={r(p)})"
        ),
    }

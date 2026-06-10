"""Correlation analysis: Pearson, Spearman, Kendall."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def correlation(x, y, method="pearson"):
    """Calculate correlation between two variables.

    Args:
        x: First variable values
        y: Second variable values
        method: 'pearson', 'spearman', or 'kendall'

    Returns:
        Dict with correlation results
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)

    # Check for constant input (correlation undefined)
    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "method": method,
            "n": len(x),
            "correlation": None,
            "r_squared": None,
            "p_value": None,
            "significant": False,
            "strength": "undefined",
            "direction": "none",
            "interpretation": "Correlation is undefined when one or both inputs are constant",
        }

    if method == "pearson":
        corr, p = sp_stats.pearsonr(x, y)
    elif method == "spearman":
        corr, p = sp_stats.spearmanr(x, y)
    elif method == "kendall":
        corr, p = sp_stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'")

    r_sq = corr**2

    # Interpretation
    abs_r = abs(corr)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if corr > 0 else "negative"

    return {
        "method": method,
        "n": n,
        "correlation": r(corr),
        "r_squared": r(r_sq),
        "p_value": r(p),
        "significant": p < 0.05,
        "strength": strength,
        "direction": direction,
        "interpretation": f"{strength.capitalize()} {direction} correlation (r={r(corr)}, p={r(p)})",
    }

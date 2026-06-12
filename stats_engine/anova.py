"""ANOVA: one-way and two-way."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def anova(anova_type, groups, alpha=0.05, data=None):
    """Perform ANOVA.

    Args:
        anova_type: 'one_way' or 'two_way'
        groups: List of lists (one_way) or dict with 'factor_a', 'factor_b', 'values' (two_way)
        alpha: Significance level
        data: DataFrame for two_way (alternative to groups)

    Returns:
        Dict with ANOVA results
    """
    if anova_type == "one_way":
        return _one_way_anova(groups, alpha)
    elif anova_type == "two_way":
        return _two_way_anova(groups, alpha, data)
    else:
        raise ValueError(f"Unknown anova_type: {anova_type}. Use 'one_way' or 'two_way'")


def _one_way_anova(groups, alpha):
    """One-way ANOVA."""
    if not groups or len(groups) < 2:
        raise ValueError("At least 2 groups are required for ANOVA")

    arrays = [np.array(g, dtype=float) for g in groups]

    # Filter out empty groups
    arrays = [a for a in arrays if len(a) > 0]
    if len(arrays) < 2:
        raise ValueError("At least 2 non-empty groups are required for ANOVA")

    k = len(arrays)
    n_total = sum(len(a) for a in arrays)

    # Run scipy one-way ANOVA
    f_stat, p_value = sp_stats.f_oneway(*arrays)

    # Calculate eta-squared (effect size)
    grand_mean = np.mean(np.concatenate(arrays))
    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in arrays)
    ss_total = sum(np.sum((a - grand_mean) ** 2) for a in arrays)
    ss_within = ss_total - ss_between
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    omega_sq = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if (ss_total + ms_within) > 0
        else 0
    )
    omega_sq = max(0, omega_sq)

    # Group statistics
    group_stats = []
    for i, arr in enumerate(arrays):
        group_stats.append(
            {
                "group": i,
                "n": len(arr),
                "mean": r(np.mean(arr)),
                "std": r(np.std(arr, ddof=1)),
            }
        )

    return {
        "anova_type": "one_way",
        "n_groups": k,
        "n_total": n_total,
        "f_statistic": r(f_stat),
        "p_value": r(p_value),
        "significant": bool(p_value < alpha),
        "eta_squared": r(eta_sq),
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": r(ss_between),
        "ss_within": r(ss_within),
        "ms_between": r(ms_between),
        "ms_within": r(ms_within),
        "omega_squared": r(omega_sq),
        "alpha": alpha,
        "group_stats": group_stats,
        "interpretation": (
            f"Significant difference between groups (F={r(f_stat, 2)}, p={r(p_value)})"
            if p_value < alpha
            else f"No significant difference between groups (F={r(f_stat, 2)}, p={r(p_value)})"
        ),
    }


def _two_way_anova(groups, alpha, data=None):
    """Two-way ANOVA using statsmodels."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError("statsmodels is required for two-way ANOVA")

    if data is not None:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    else:
        # groups should be dict with 'factor_a', 'factor_b', 'values'
        df = pd.DataFrame(
            {
                "factor_a": groups["factor_a"],
                "factor_b": groups["factor_b"],
                "values": groups["values"],
            }
        )

    model = ols("values ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract results
    results = {
        "anova_type": "two_way",
        "sources": [],
    }

    ss_residual = anova_table.loc["Residual", "sum_sq"]

    for source in anova_table.index:
        if source == "Residual":
            continue
        row = anova_table.loc[source]
        ss_effect = row["sum_sq"]
        partial_eta_sq = ss_effect / (ss_effect + ss_residual) if (ss_effect + ss_residual) > 0 else 0
        results["sources"].append(
            {
                "source": source,
                "sum_sq": r(row["sum_sq"]),
                "df": int(row["df"]),
                "f_statistic": r(row["F"]) if not np.isnan(row["F"]) else None,
                "p_value": r(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
                "significant": bool(row["PR(>F)"] < alpha) if not np.isnan(row["PR(>F)"]) else None,
                "partial_eta_squared": r(partial_eta_sq),
            }
        )

    results["alpha"] = alpha
    results["r_squared"] = r(model.rsquared)

    return results

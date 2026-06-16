"""Advanced statistical methods: mixed effects, exact test, McNemar, Cochran's Q."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import p_value_context, r


def advanced(analysis_type, **kwargs):
    """Perform advanced statistical analysis.

    Args:
        analysis_type: 'mixed_effects', 'exact_test', 'mcnemar', 'cochran_q'

    Returns:
        Dict with analysis results
    """
    if analysis_type == "mixed_effects":
        return _mixed_effects(**kwargs)
    elif analysis_type == "exact_test":
        return _exact_test(**kwargs)
    elif analysis_type == "mcnemar":
        return _mcnemar(**kwargs)
    elif analysis_type == "cochran_q":
        return _cochran_q(**kwargs)
    else:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type}. Use 'mixed_effects', 'exact_test', 'mcnemar', or 'cochran_q'"
        )


def _mixed_effects(groups, group_ids, alpha=0.05, **kwargs):
    """Mixed effects linear model.

    Args:
        groups: List of observation values
        group_ids: List of group/subject identifiers (random effect)
        alpha: Significance level
    """
    try:
        import pandas as pd
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        raise ImportError("statsmodels required for mixed effects models")

    y = np.array(groups, dtype=float)
    groups_arr = np.array(group_ids)

    df = pd.DataFrame({"y": y, "group": groups_arr})
    df["intercept"] = 1.0

    model = MixedLM.from_formula("y ~ 1", groups="group", data=df)
    result = model.fit()

    # Extract variance components
    re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result, "cov_re") else 0
    residual_var = float(result.scale)

    return {
        "analysis_type": "mixed_effects",
        "n": len(y),
        "n_groups": len(np.unique(groups_arr)),
        "fixed_effects": {
            "intercept": r(result.fe_params["Intercept"]),
            "se": r(result.bse["Intercept"]),
            "p_value": r(result.pvalues["Intercept"]),
            "ci_lower": r(result.conf_int().iloc[0, 0]),
            "ci_upper": r(result.conf_int().iloc[0, 1]),
        },
        "variance_components": {
            "random_effect": r(re_var),
            "residual": r(residual_var),
        },
        "icc": r(re_var / (re_var + residual_var)) if (re_var + residual_var) > 0 else 0,
        "aic": r(result.aic),
        "bic": r(result.bic),
        "significant": result.pvalues["Intercept"] < alpha,
        "interpretation": (
            f"Fixed effect intercept = {r(result.fe_params['Intercept'])} "
            f"(p={r(result.pvalues['Intercept'])}). "
            f"ICC = {r(re_var / (re_var + residual_var)) if (re_var + residual_var) > 0 else 0}"
        ),
    }


def _exact_test(observed, alpha=0.05, **kwargs):
    """Fisher's exact test for 2x2 contingency table.

    Args:
        observed: 2x2 contingency table as list of lists
        alpha: Significance level
    """
    table = np.array(observed, dtype=int)
    if table.shape != (2, 2):
        raise ValueError("Fisher's exact test requires a 2x2 contingency table")

    odds_ratio, p_value = sp_stats.fisher_exact(table)

    # Marginal totals
    row_totals = table.sum(axis=1).tolist()
    col_totals = table.sum(axis=0).tolist()
    n_total = int(table.sum())

    result = {
        "analysis_type": "exact_test",
        "test": "Fisher's exact test",
        "observed": table.tolist(),
        "row_totals": row_totals,
        "col_totals": col_totals,
        "n_total": n_total,
        "odds_ratio": r(odds_ratio),
        "p_value": r(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "interpretation": (
            f"Significant association (OR={r(odds_ratio)}, p={r(p_value)})"
            if p_value < alpha
            else f"No significant association (OR={r(odds_ratio)}, p={r(p_value)})"
        ),
    }
    return p_value_context(result, p_value, alpha, n_total)


def _mcnemar(observed, alpha=0.05, **kwargs):
    """McNemar's test for paired nominal data.

    Args:
        observed: 2x2 contingency table as list of lists
        alpha: Significance level
    """
    table = np.array(observed, dtype=int)
    if table.shape != (2, 2):
        raise ValueError("McNemar's test requires a 2x2 contingency table")

    # McNemar focuses on discordant pairs: b and c
    b = table[0, 1]
    c = table[1, 0]

    n_discordant = b + c

    if n_discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    elif n_discordant < 25:
        # Use exact binomial test for small samples
        p_value = float(sp_stats.binomtest(int(b), n_discordant, 0.5).pvalue)
        chi2 = None
    else:
        # Use chi-squared approximation with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / n_discordant
        p_value = float(1 - sp_stats.chi2.cdf(chi2, 1))

    result = {
        "analysis_type": "mcnemar",
        "observed": table.tolist(),
        "discordant_pairs": n_discordant,
        "b_discordant": int(b),
        "c_discordant": int(c),
        "chi2": r(chi2) if chi2 is not None else None,
        "p_value": r(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "interpretation": (
            f"Significant change in paired proportions (p={r(p_value)})"
            if p_value < alpha
            else f"No significant change in paired proportions (p={r(p_value)})"
        ),
    }
    n_total = int(table.sum())
    return p_value_context(result, p_value, alpha, n_total)


def _cochran_q(data, alpha=0.05, **kwargs):
    """Cochran's Q test for k related binary treatments.

    Args:
        data: List of lists (k groups, each with n binary observations: 0 or 1)
        alpha: Significance level
    """
    matrix = np.array(data, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("Cochran's Q requires a 2D matrix (k groups x n observations)")

    k = matrix.shape[0]  # number of treatments
    n = matrix.shape[1]  # number of subjects

    # Row totals (Ri) - sum per treatment
    row_totals = matrix.sum(axis=1)
    # Column totals (Cj) - sum per subject
    col_totals = matrix.sum(axis=0)
    # Grand total
    T = matrix.sum()

    # Cochran's Q statistic
    sum_Ri_sq = np.sum(row_totals**2)
    sum_Cj_sq = np.sum(col_totals**2)

    numerator = (k - 1) * (k * sum_Ri_sq - T**2)
    denominator = k * T - sum_Cj_sq

    if denominator == 0:
        q_stat = 0.0
        p_value = 1.0
    else:
        q_stat = float(numerator / denominator)
        p_value = float(1 - sp_stats.chi2.cdf(q_stat, k - 1))

    result = {
        "analysis_type": "cochran_q",
        "n_treatments": k,
        "n_subjects": n,
        "treatment_totals": row_totals.tolist(),
        "subject_totals": col_totals.tolist(),
        "grand_total": int(T),
        "q_statistic": r(q_stat),
        "df": k - 1,
        "p_value": r(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "interpretation": (
            f"Significant difference between treatments (Q={r(q_stat)}, p={r(p_value)})"
            if p_value < alpha
            else f"No significant difference between treatments (Q={r(q_stat)}, p={r(p_value)})"
        ),
    }
    return p_value_context(result, p_value, alpha, n * k)

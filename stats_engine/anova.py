"""ANOVA: one-way and two-way."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import p_value_context, r


def anova(
    groups=None,
    anova_type="one_way",
    alpha=0.05,
    data=None,
    factors=None,
    response=None,
    subject=None,
    within=None,
    response_col="value",
):
    """Perform ANOVA.

    Args:
        groups: List of lists (one_way) or dict with 'factor_a', 'factor_b', 'values' (two_way)
        anova_type: 'one_way', 'two_way', or 'n_way' (default: 'one_way')
        alpha: Significance level
        data: DataFrame for two_way/n_way (alternative to groups)
        factors: Dict of {name: [values]} for n_way ANOVA
        response: Response values for n_way ANOVA

    Returns:
        Dict with ANOVA results
    """
    if anova_type == "one_way":
        if not groups:
            raise ValueError("groups parameter is required for one_way ANOVA")
        return _one_way_anova(groups, alpha)
    elif anova_type == "two_way":
        return _two_way_anova(groups, alpha, data)
    elif anova_type == "n_way":
        n_way_groups = {"factors": factors, "response": response} if factors and response else groups
        return _n_way_anova(n_way_groups, alpha, data)
    elif anova_type == "repeated":
        return _repeated_anova(groups, alpha, data, subject, within, response_col)
    else:
        raise ValueError(f"Unknown anova_type: {anova_type}. Use 'one_way', 'two_way', 'n_way', or 'repeated'")


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
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0
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

    result = {
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
    return p_value_context(result, p_value, alpha, n_total)


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

    # Build interpretation from significant sources
    sig_sources = [s["source"] for s in results["sources"] if s.get("significant")]
    if sig_sources:
        results["interpretation"] = (
            f"Two-way ANOVA: significant effects for {', '.join(sig_sources)}, R² = {r(model.rsquared)}"
        )
    else:
        results["interpretation"] = f"Two-way ANOVA: no significant effects, R² = {r(model.rsquared)}"

    return results


def _n_way_anova(groups, alpha, data=None):
    """N-way ANOVA (3+ factors) using statsmodels formula API.

    Args:
        groups: Dict with 'factors' (dict of {name: [values]}) and 'response' (list)
        alpha: Significance level
        data: DataFrame (alternative to groups)

    Returns:
        Dict with ANOVA results for all main effects and interactions
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError("statsmodels and pandas are required for N-way ANOVA")

    # Build DataFrame from inputs
    if data is not None:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    elif isinstance(groups, dict) and "factors" in groups:
        factor_dict = groups["factors"]
        response = groups.get("response", groups.get("values", []))
        factor_dict["__response__"] = response
        df = pd.DataFrame(factor_dict)
    else:
        raise ValueError("n_way ANOVA requires {'factors': {...}, 'response': [...]} format")

    factor_names = [c for c in df.columns if c != "__response__"]

    if len(factor_names) < 2:
        raise ValueError("N-way ANOVA requires at least 2 factors")
    if len(factor_names) > 5:
        raise ValueError(f"Maximum 5 factors supported, got {len(factor_names)}")

    # Build formula using Q() to quote column names (avoids patsy C() ambiguity)
    from itertools import combinations

    quoted = [f"Q('{name}')" for name in factor_names]
    main_effects = " + ".join(quoted)

    # Generate interaction terms (2-way through n-way)
    interaction_terms = []
    for order in range(2, len(factor_names) + 1):
        for combo in combinations(quoted, order):
            interaction_terms.append(":".join(combo))

    formula = f"__response__ ~ {main_effects}"
    if interaction_terms:
        formula += " + " + " + ".join(interaction_terms)

    # Pre-check: ensure enough observations for the model degrees of freedom
    # Model df = number of coefficients (including intercept) - 1 + residuals
    # For a saturated design (n <= n_params), statsmodels produces inf F-values
    n_obs = len(df)
    # Rough upper bound on model parameters: product of factor levels
    n_params = 1  # intercept
    for fname in factor_names:
        n_params *= len(df[fname].unique())
    if n_obs <= n_params + 1:
        raise ValueError(
            f"N-way ANOVA needs more observations ({n_obs}) than model parameters ({n_params + 1}). "
            f"Provide at least {n_params + 2} observations for the given factor design."
        )

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Parse results
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    results = {
        "anova_type": "n_way",
        "n_factors": len(factor_names),
        "factor_names": factor_names,
        "formula": formula.replace("__response__", "response"),
        "sources": [],
    }

    for source in anova_table.index:
        if source == "Residual":
            continue
        row = anova_table.loc[source]
        ss_effect = row["sum_sq"]
        partial_eta_sq = ss_effect / (ss_effect + ss_residual) if (ss_effect + ss_residual) > 0 else 0
        f_val = row["F"]
        p_val = row["PR(>F)"]
        results["sources"].append(
            {
                "source": source.replace("C(", "").replace(")", ""),
                "sum_sq": r(ss_effect),
                "df": int(row["df"]),
                "f_statistic": r(f_val) if not np.isnan(f_val) and not np.isinf(f_val) else None,
                "p_value": r(p_val) if not np.isnan(p_val) and not np.isinf(p_val) else None,
                "significant": bool(p_val < alpha) if not np.isnan(p_val) and not np.isinf(p_val) else None,
                "partial_eta_squared": r(partial_eta_sq),
            }
        )

    results["alpha"] = alpha
    results["r_squared"] = r(model.rsquared)
    results["adj_r_squared"] = r(model.rsquared_adj)
    n_obs = int(model.nobs)
    results["n_observations"] = n_obs

    # Build interpretation
    sig_sources = [s["source"] for s in results["sources"] if s.get("significant")]
    if sig_sources:
        results["interpretation"] = (
            f"{len(factor_names)}-way ANOVA: significant effects for {', '.join(sig_sources)}, R² = {r(model.rsquared)}"
        )
    else:
        results["interpretation"] = f"{len(factor_names)}-way ANOVA: no significant effects, R² = {r(model.rsquared)}"

    p_overall = min(s["p_value"] for s in results["sources"] if s["p_value"] is not None) if results["sources"] else 1.0
    return p_value_context(results, p_overall, alpha, n_obs)


def _repeated_anova(groups, alpha, data=None, subject=None, within=None, response_col="value"):
    """Repeated measures ANOVA using statsmodels AnovaRM.

    Args:
        groups: Dict with 'data' (DataFrame) or 'file' path
        alpha: Significance level
        data: DataFrame with subject, within, and response columns
        subject: Column name identifying subjects
        within: Column name(s) identifying within-subject factors (str or list)
        response_col: Column name for the response variable

    Returns:
        Dict with repeated measures ANOVA results
    """
    try:
        import pandas as pd
        from statsmodels.stats.anova import AnovaRM
    except ImportError:
        raise ImportError("statsmodels and pandas are required for repeated measures ANOVA")

    # Load data
    if data is not None:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    elif isinstance(groups, dict) and "data" in groups:
        df = groups["data"] if isinstance(groups["data"], pd.DataFrame) else pd.DataFrame(groups["data"])
    elif isinstance(groups, dict) and "file" in groups:
        from utils.data_loader import read_dataframe

        df = read_dataframe(groups["file"])
    else:
        raise ValueError("repeated ANOVA requires DataFrame or {'data': [...]} or {'file': path}")

    if subject is None:
        raise ValueError("'subject' parameter is required for repeated measures ANOVA")
    if within is None:
        raise ValueError("'within' parameter is required for repeated measures ANOVA")
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found. Available: {list(df.columns)}")

    # Run AnovaRM
    within_list = [within] if isinstance(within, str) else within

    # AnovaRM expects: depvar, subject, within
    if len(within_list) == 1:
        anova_result = AnovaRM(df, depvar=response_col, subject=subject, within=within_list).fit()
    else:
        # Multiple within factors: combine into single factor for AnovaRM
        combined_name = "_x_".join(within_list)
        df[combined_name] = df[within_list].astype(str).agg("_".join, axis=1)
        anova_result = AnovaRM(df, depvar=response_col, subject=subject, within=[combined_name]).fit()

    # Extract results
    anova_table = anova_result.anova_table

    results = {
        "anova_type": "repeated",
        "subject": subject,
        "within": within_list,
        "response": response_col,
        "n_subjects": int(df[subject].nunique()),
        "sources": [],
    }

    for source in anova_table.index:
        if source == "Residual":
            continue
        row = anova_table.loc[source]
        f_val = row["F Value"]
        p_val = row["Pr > F"]
        results["sources"].append(
            {
                "source": source,
                "f_statistic": r(f_val) if not np.isnan(f_val) and not np.isinf(f_val) else None,
                "p_value": r(p_val) if not np.isnan(p_val) and not np.isinf(p_val) else None,
                "significant": bool(p_val < alpha) if not np.isnan(p_val) and not np.isinf(p_val) else None,
                "num_df": r(row["Num DF"]),
                "den_df": r(row["Den DF"]),
            }
        )

    results["alpha"] = alpha
    results["n_observations"] = len(df)

    # Build interpretation
    sig_sources = [s["source"] for s in results["sources"] if s.get("significant")]
    if sig_sources:
        results["interpretation"] = f"Repeated measures ANOVA: significant effects for {', '.join(sig_sources)}"
    else:
        results["interpretation"] = "Repeated measures ANOVA: no significant within-subject effects"

    # Sphericity warning (AnovaRM assumes sphericity)
    results["sphericity_note"] = (
        "Sphericity assumed (Mauchly test not available via AnovaRM). If violated, use Greenhouse-Geisser correction."
    )

    p_overall = min(s["p_value"] for s in results["sources"] if s["p_value"] is not None) if results["sources"] else 1.0
    return p_value_context(results, p_overall, alpha, len(df))

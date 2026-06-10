"""MSA / Gage R&R analysis: crossed, nested, attribute, bias, linearity, stability."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def gage_rr(analysis_type, **kwargs):
    """Perform Gage R&R / MSA analysis.

    Args:
        analysis_type: 'crossed', 'nested', 'attribute', 'bias', 'linearity', 'stability'

    Returns:
        Dict with MSA results
    """
    if analysis_type == "crossed":
        return _crossed(**kwargs)
    elif analysis_type == "nested":
        return _nested(**kwargs)
    elif analysis_type == "attribute":
        return _attribute(**kwargs)
    elif analysis_type == "bias":
        return _bias(**kwargs)
    elif analysis_type == "linearity":
        return _linearity(**kwargs)
    elif analysis_type == "stability":
        return _stability(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _crossed(measurements, parts, operators, tolerance=None, **kwargs):
    """Crossed Gage R&R (ANOVA method)."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError("statsmodels required for Gage R&R")

    measurements = np.array(measurements, dtype=float)
    parts = np.array(parts)
    operators = np.array(operators)

    n = len(measurements)
    part_levels = np.unique(parts)
    op_levels = np.unique(operators)
    k = len(part_levels)
    m = len(op_levels)
    n_reps = n // (k * m)

    df = pd.DataFrame(
        {
            "measurement": measurements,
            "part": pd.Categorical(parts, categories=part_levels),
            "operator": pd.Categorical(operators, categories=op_levels),
        }
    )

    if n_reps > 1:
        # Full model with interaction
        model = ols("measurement ~ C(part) * C(operator)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        df_part = int(anova_table.loc["C(part)", "df"])
        df_oper = int(anova_table.loc["C(operator)", "df"])
        df_interact = int(anova_table.loc["C(part):C(operator)", "df"])
        df_error = int(anova_table.loc["Residual", "df"])

        ms_part = float(anova_table.loc["C(part)", "sum_sq"]) / df_part
        ms_oper = float(anova_table.loc["C(operator)", "sum_sq"]) / df_oper
        ms_interact = float(anova_table.loc["C(part):C(operator)", "sum_sq"]) / df_interact
        ms_error = float(anova_table.loc["Residual", "sum_sq"]) / df_error

        sigma2_error = ms_error
        sigma2_interaction = max(0, (ms_interact - ms_error) / n_reps)
        sigma2_operator = max(0, (ms_oper - ms_interact) / (k * n_reps))
        sigma2_part = max(0, (ms_part - ms_interact) / (m * n_reps))

        has_interaction = True

        anova_sources = ["Part", "Operator", "Part:Operator", "Residual", "Total"]
        anova_df = [df_part, df_oper, df_interact, df_error, df_part + df_oper + df_interact + df_error]
        anova_ss = [
            float(anova_table.loc["C(part)", "sum_sq"]),
            float(anova_table.loc["C(operator)", "sum_sq"]),
            float(anova_table.loc["C(part):C(operator)", "sum_sq"]),
            float(anova_table.loc["Residual", "sum_sq"]),
            float(anova_table["sum_sq"].sum()),
        ]
        anova_ms = [ms_part, ms_oper, ms_interact, ms_error, None]
    else:
        model = ols("measurement ~ C(part) + C(operator)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        df_part = int(anova_table.loc["C(part)", "df"])
        df_oper = int(anova_table.loc["C(operator)", "df"])
        df_error = int(anova_table.loc["Residual", "df"])

        ms_part = float(anova_table.loc["C(part)", "sum_sq"]) / df_part
        ms_oper = float(anova_table.loc["C(operator)", "sum_sq"]) / df_oper
        ms_error = float(anova_table.loc["Residual", "sum_sq"]) / df_error

        ms_interact = None
        df_interact = 0
        sigma2_interaction = 0
        sigma2_error = ms_error
        sigma2_operator = max(0, (ms_oper - ms_error) / k)
        sigma2_part = max(0, (ms_part - ms_error) / m)

        has_interaction = False

        anova_sources = ["Part", "Operator", "Residual", "Total"]
        anova_df = [df_part, df_oper, df_error, df_part + df_oper + df_error]
        anova_ss = [
            float(anova_table.loc["C(part)", "sum_sq"]),
            float(anova_table.loc["C(operator)", "sum_sq"]),
            float(anova_table.loc["Residual", "sum_sq"]),
            float(anova_table["sum_sq"].sum()),
        ]
        anova_ms = [ms_part, ms_oper, ms_error, None]

    sigma2_grr = sigma2_error + sigma2_interaction + sigma2_operator
    sigma2_total = sigma2_part + sigma2_grr

    sd_repeatability = np.sqrt(sigma2_error)
    sd_reproducibility = np.sqrt(sigma2_operator + sigma2_interaction)
    sd_grr = np.sqrt(sigma2_grr)
    sd_part = np.sqrt(sigma2_part)
    sd_total = np.sqrt(sigma2_total) if sigma2_total > 0 else 1

    pct_contrib_repeatability = (sigma2_error / sigma2_total * 100) if sigma2_total > 0 else 0
    pct_contrib_reproducibility = (
        ((sigma2_operator + sigma2_interaction) / sigma2_total * 100) if sigma2_total > 0 else 0
    )
    pct_contrib_grr = (sigma2_grr / sigma2_total * 100) if sigma2_total > 0 else 0
    pct_contrib_part = (sigma2_part / sigma2_total * 100) if sigma2_total > 0 else 0

    sv_repeatability = 6 * sd_repeatability
    sv_reproducibility = 6 * sd_reproducibility
    sv_grr = 6 * sd_grr
    sv_part = 6 * sd_part

    pct_sv_grr = (sd_grr / sd_total * 100) if sd_total > 0 else 0

    ndc = int(np.floor(1.41 * (sd_part / sd_grr))) if sd_grr > 0 else 99

    if pct_sv_grr < 10:
        rating = "Excellent"
        rating_desc = "Measurement system is acceptable"
    elif pct_sv_grr < 30:
        rating = "Acceptable"
        rating_desc = "Measurement system may be acceptable depending on application"
    else:
        rating = "Unacceptable"
        rating_desc = "Measurement system needs improvement"

    result = {
        "analysis_type": "crossed",
        "n": n,
        "n_parts": k,
        "n_operators": m,
        "n_replicates": int(n_reps),
        "has_interaction": has_interaction,
        "anova_table": {
            "source": anova_sources,
            "df": anova_df,
            "sum_sq": [r(v) for v in anova_ss],
            "mean_sq": [r(v) if v is not None else None for v in anova_ms],
        },
        "mean_sq": {
            "part": r(ms_part),
            "operator": r(ms_oper),
            "interaction": r(ms_interact) if ms_interact is not None else None,
            "error": r(ms_error),
        },
        "variance_components": {
            "repeatability": {
                "variance": r(sigma2_error),
                "sd": r(sd_repeatability),
                "sv": r(sv_repeatability),
            },
            "reproducibility": {
                "variance": r(sigma2_operator + sigma2_interaction),
                "sd": r(sd_reproducibility),
                "sv": r(sv_reproducibility),
                "operator": {"variance": r(sigma2_operator), "sd": r(np.sqrt(sigma2_operator))},
                "interaction": {
                    "variance": r(sigma2_interaction),
                    "sd": r(np.sqrt(sigma2_interaction)),
                },
            },
            "grr": {"variance": r(sigma2_grr), "sd": r(sd_grr), "sv": r(sv_grr)},
            "part_to_part": {
                "variance": r(sigma2_part),
                "sd": r(sd_part),
                "sv": r(sv_part),
            },
            "total": {"variance": r(sigma2_total), "sd": r(sd_total)},
        },
        "contribution": {
            "repeatability": r(pct_contrib_repeatability, 2),
            "reproducibility": r(pct_contrib_reproducibility, 2),
            "grr": r(pct_contrib_grr, 2),
            "part_to_part": r(pct_contrib_part, 2),
        },
        "study_variation": {
            "repeatability": r(sd_repeatability / sd_total * 100, 2) if sd_total > 0 else 0,
            "reproducibility": r(sd_reproducibility / sd_total * 100, 2) if sd_total > 0 else 0,
            "grr": r(pct_sv_grr, 2),
            "part_to_part": r(sd_part / sd_total * 100, 2) if sd_total > 0 else 0,
        },
        "ndc": ndc,
        "rating": rating,
        "rating_desc": rating_desc,
        "interpretation": f"Gage R&R = {r(pct_sv_grr, 1)}% of study variation. {rating_desc} (ndc = {ndc})",
    }

    if tolerance and tolerance > 0:
        result["tolerance"] = {
            "grr": r(sv_grr / tolerance * 100, 2),
            "part_to_part": r(sv_part / tolerance * 100, 2),
        }

    return result


def _nested(measurements, parts, operators, **kwargs):
    """Nested Gage R&R."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        raise ImportError("statsmodels required for Gage R&R")

    measurements = np.array(measurements, dtype=float)
    df = pd.DataFrame(
        {
            "measurement": measurements,
            "part": pd.Categorical(parts),
            "operator": pd.Categorical(operators),
        }
    )

    model = ols("measurement ~ C(operator) / C(part)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    df_operator = float(anova_table.loc["C(operator)", "df"])
    df_part_in_op = float(anova_table.loc["C(operator):C(part)", "df"])
    df_error = float(anova_table.loc["Residual", "df"])

    ms_operator = float(anova_table.loc["C(operator)", "sum_sq"]) / df_operator
    ms_part_in_op = float(anova_table.loc["C(operator):C(part)", "sum_sq"]) / df_part_in_op
    ms_error = float(anova_table.loc["Residual", "sum_sq"]) / df_error

    max_reps = max(pd.Series(operators).value_counts())

    sigma2_error = ms_error
    sigma2_part_in_op = max(0, (ms_part_in_op - ms_error) / max_reps)
    n_parts_per_op = len(np.unique(parts)) / len(np.unique(operators))
    sigma2_operator = max(0, (ms_operator - ms_part_in_op) / n_parts_per_op)

    sigma2_grr = sigma2_error + sigma2_operator
    sigma2_part = sigma2_part_in_op
    sigma2_total = sigma2_part + sigma2_grr

    sd_grr = np.sqrt(sigma2_grr)
    sd_part = np.sqrt(sigma2_part)
    sd_total = np.sqrt(sigma2_total) if sigma2_total > 0 else 1

    pct_sv_grr = (sd_grr / sd_total * 100) if sd_total > 0 else 0
    ndc = int(np.floor(1.41 * (sd_part / sd_grr))) if sd_grr > 0 else 99

    if pct_sv_grr < 10:
        rating = "Excellent"
    elif pct_sv_grr < 30:
        rating = "Acceptable"
    else:
        rating = "Unacceptable"

    return {
        "analysis_type": "nested",
        "n": len(measurements),
        "n_parts": len(np.unique(parts)),
        "n_operators": len(np.unique(operators)),
        "variance_components": {
            "repeatability": {"variance": r(sigma2_error), "sd": r(np.sqrt(sigma2_error))},
            "operator": {"variance": r(sigma2_operator), "sd": r(np.sqrt(sigma2_operator))},
            "part_in_operator": {
                "variance": r(sigma2_part_in_op),
                "sd": r(np.sqrt(sigma2_part_in_op)),
            },
            "grr": {"variance": r(sigma2_grr), "sd": r(sd_grr)},
            "total": {"variance": r(sigma2_total), "sd": r(sd_total)},
        },
        "ndc": ndc,
        "rating": rating,
        "interpretation": f"Gage R&R = {r(pct_sv_grr, 1)}% of study variation. {rating} (ndc = {ndc})",
    }


def _attribute(reference, ratings, **kwargs):
    """Attribute agreement analysis."""
    n_samples = len(ratings)
    n_operators = len(ratings[0])
    n_replicates = len(ratings[0][0])

    operator_agreement = []
    operator_kappa = []

    for op in range(n_operators):
        matches = 0
        total = 0
        for s in range(n_samples):
            ref = reference[s]
            for rep in range(n_replicates):
                if ratings[s][op][rep] is not None:
                    total += 1
                    if ratings[s][op][rep] == ref:
                        matches += 1
        agreement = (matches / total * 100) if total > 0 else 0
        p_o = matches / total if total > 0 else 0
        # Compute expected agreement from marginal frequencies
        n_pass_ref = sum(1 for s2 in range(n_samples) if reference[s2] == 1)
        n_fail_ref = n_samples - n_pass_ref
        op_pass = 0
        op_total = 0
        for s2 in range(n_samples):
            for rep2 in range(n_replicates):
                if ratings[s2][op][rep2] is not None:
                    op_total += 1
                    if ratings[s2][op][rep2] == 1:
                        op_pass += 1
        op_fail = op_total - op_pass
        if total > 0 and n_samples > 0 and op_total > 0:
            p_e = (n_pass_ref / n_samples) * (op_pass / op_total) + (n_fail_ref / n_samples) * (op_fail / op_total)
        else:
            p_e = 0.5
        kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0
        operator_agreement.append(r(agreement, 2))
        operator_kappa.append(r(kappa, 3))

    all_matches = 0
    all_total = 0
    for s in range(n_samples):
        ref = reference[s]
        for op in range(n_operators):
            for rep in range(n_replicates):
                if ratings[s][op][rep] is not None:
                    all_total += 1
                    if ratings[s][op][rep] == ref:
                        all_matches += 1

    overall_agreement = (all_matches / all_total * 100) if all_total > 0 else 0
    # Compute overall kappa with proper expected agreement
    total_pass_ref = sum(1 for s in range(n_samples) if reference[s] == 1)
    total_fail_ref = n_samples - total_pass_ref
    total_pass_ratings = 0
    total_ratings = 0
    for s in range(n_samples):
        for op in range(n_operators):
            for rep in range(n_replicates):
                if ratings[s][op][rep] is not None:
                    total_ratings += 1
                    if ratings[s][op][rep] == 1:
                        total_pass_ratings += 1
    total_fail_ratings = total_ratings - total_pass_ratings
    if n_samples > 0 and total_ratings > 0:
        p_e_overall = (total_pass_ref / n_samples) * (total_pass_ratings / total_ratings) + (total_fail_ref / n_samples) * (total_fail_ratings / total_ratings)
    else:
        p_e_overall = 0.5
    overall_kappa = (overall_agreement / 100 - p_e_overall) / (1 - p_e_overall) if (1 - p_e_overall) != 0 else 0

    return {
        "analysis_type": "attribute_agreement",
        "n_samples": n_samples,
        "n_operators": n_operators,
        "n_replicates": n_replicates,
        "overall": {"agreement_pct": r(overall_agreement, 2), "kappa": r(overall_kappa, 3)},
        "operators": [
            {"name": f"Operator_{op + 1}", "agreement_pct": operator_agreement[op], "kappa": operator_kappa[op]}
            for op in range(n_operators)
        ],
        "interpretation": f"Overall agreement = {r(overall_agreement, 1)}%, Kappa = {r(overall_kappa, 3)}",
    }


def _bias(measurements, reference_value, **kwargs):
    """Bias study."""
    measurements = np.array(measurements, dtype=float)
    n = len(measurements)
    mean_val = float(np.mean(measurements))
    sd_val = float(np.std(measurements, ddof=1))
    bias = mean_val - reference_value
    bias_pct = (bias / reference_value * 100) if reference_value != 0 else 0

    t_stat = bias / (sd_val / np.sqrt(n))
    p_value = 2 * sp_stats.t.cdf(-abs(t_stat), n - 1)

    return {
        "analysis_type": "bias",
        "n": n,
        "reference_value": reference_value,
        "mean": r(mean_val),
        "sd": r(sd_val),
        "bias": r(bias),
        "bias_pct": r(bias_pct, 2),
        "t_statistic": r(t_stat),
        "p_value": r(p_value),
        "significant": p_value < 0.05,
        "interpretation": (
            f"Significant bias detected (bias={r(bias)}, p={r(p_value)})"
            if p_value < 0.05
            else f"No significant bias detected (bias={r(bias)}, p={r(p_value)})"
        ),
    }


def _linearity(reference_values, measurements, **kwargs):
    """Linearity study."""
    ref = np.array(reference_values, dtype=float)
    meas = np.array(measurements, dtype=float)

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(ref, meas)
    bias = meas - ref
    linearity = abs(slope - 1) * (np.max(ref) - np.min(ref))

    return {
        "analysis_type": "linearity",
        "n": len(meas),
        "intercept": r(intercept),
        "slope": r(slope),
        "r_squared": r(r_value**2),
        "linearity": r(linearity),
        "bias_mean": r(np.mean(bias)),
        "bias_sd": r(np.std(bias, ddof=1)),
        "interpretation": f"Slope={r(slope)} (ideal=1.0), Intercept={r(intercept)} (ideal=0.0), Linearity={r(linearity)}",
    }


def _stability(measurements, time_points=None, reference_value=None, tolerance=None, **kwargs):
    """Stability study."""
    measurements = np.array(measurements, dtype=float)
    n = len(measurements)
    mean_val = float(np.mean(measurements))
    sd_val = float(np.std(measurements, ddof=1))

    ucl = mean_val + 3 * sd_val
    lcl = mean_val - 3 * sd_val
    ooc = [int(i) for i in range(n) if measurements[i] > ucl or measurements[i] < lcl]

    slope = None
    p_trend = None
    if time_points is not None and len(time_points) > 1:
        time_numeric = np.arange(n, dtype=float)
        slope, intercept, r_value, p_trend, std_err = sp_stats.linregress(time_numeric, measurements)
        slope = float(slope)
        p_trend = float(p_trend)

    result = {
        "analysis_type": "stability",
        "n": n,
        "mean": r(mean_val),
        "sd": r(sd_val),
        "ucl": r(ucl),
        "lcl": r(lcl),
        "out_of_control": ooc,
        "n_out_of_control": len(ooc),
        "trend_slope": r(slope) if slope is not None else None,
        "trend_p_value": r(p_trend) if p_trend is not None else None,
        "trend_significant": p_trend < 0.05 if p_trend is not None else None,
        "interpretation": (
            "Measurement system appears stable - no out-of-control points"
            if len(ooc) == 0
            else f"Measurement system may be unstable - {len(ooc)} out-of-control point(s) detected"
        ),
    }

    if tolerance and tolerance > 0:
        result["pct_tolerance"] = r(6 * sd_val / tolerance * 100, 2)

    return result

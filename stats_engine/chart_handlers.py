"""Chart generation handlers for each command.

Each handler takes (result, params) and returns a base64 PNG string or None.
"""

from stats_engine.charts import (
    acf_plot,
    bootstrap_plot,
    boxplot,
    capability_plot,
    control_chart_plot,
    functional_plot,
    heatmap,
    histogram,
    means_comparison_plot,
    missing_values_plot,
    outlier_plot,
    pareto_plot,
    posterior_plot,
    power_curve,
    qq_plot,
    residual_plot,
    scatter_with_regression,
    time_series_plot,
    tost_plot,
    ttest_plot,
    variance_component_plot,
    weibull_plot,
)


def _histogram_handler(result, params, title="Distribution"):
    """Simple histogram from _values."""
    vals = result.get("_values", [])
    return histogram(vals, title=title) if vals else None


def _qq_handler(result, params):
    """Q-Q plot from _values."""
    return qq_plot(result.get("_values", []), title="Q-Q Plot")


def _control_chart_handler(result, params):
    """Control chart from result data."""
    return control_chart_plot(
        result.get("_values", []),
        chart_type=result.get("chart_type", "imr"),
        ucl=result.get("ucl"),
        cl=result.get("cl"),
        lcl=result.get("lcl"),
        out_of_control=result.get("out_of_control", []),
    )


def _capability_handler(result, params):
    """Capability histogram with spec limits."""
    return capability_plot(
        result.get("_values", []),
        usl=result.get("usl"),
        lsl=result.get("lsl"),
        target=result.get("target"),
        cp=result.get("cp"),
        cpk=result.get("cpk"),
    )


def _scatter_handler(result, params):
    """Simple scatter plot from x/y params."""
    x, y = params.get("x", []), params.get("y", [])
    return scatter_with_regression(x, y, title="Correlation") if x and y else None


def _regression_handler(result, params):
    """Regression plot with residual diagnostics."""
    coeffs = result.get("coefficients", {})
    slope = intercept = None
    if isinstance(coeffs, dict):
        slope = coeffs.get("slope")
        intercept = coeffs.get("intercept")
    x_vals = params.get("x", [])
    y_vals = params.get("y", [])
    if slope is not None and intercept is not None and x_vals and y_vals:
        return residual_plot(
            x_vals,
            y_vals,
            slope=slope,
            intercept=intercept,
            title=f"Regression Diagnostics ({result.get('regression_type', '')})",
        )
    return scatter_with_regression(
        x_vals,
        y_vals,
        title=f"Regression ({result.get('regression_type', '')})",
        slope=slope,
        intercept=intercept,
        r_squared=result.get("r_squared"),
    )


def _timeseries_handler(result, params):
    """Time series or ACF plot."""
    analysis_type = result.get("analysis_type", "")
    if analysis_type == "acf":
        return acf_plot(
            result.get("_values", []),
            max_lag=params.get("max_lag", 20),
            title="Autocorrelation Function",
        )
    return time_series_plot(
        result.get("_values", []),
        title=f"Time Series ({analysis_type})",
        fitted=result.get("fitted_values"),
        forecast=result.get("forecast"),
    )


def _ttest_handler(result, params):
    """T-test boxplot with p-value."""
    g1 = params.get("values", [])
    g2 = params.get("values2")
    return ttest_plot(
        g1,
        group2=g2,
        p_value=result.get("p_value"),
        ci=result.get("ci_95"),
        title=f"t-Test ({result.get('test_type', '')})",
    )


def _boxplot_handler(result, params, title="Group Comparison"):
    """Generic boxplot from groups param."""
    groups = params.get("groups", [])
    return boxplot(groups, title=title) if groups else None


def _multiple_comparison_handler(result, params):
    """Means comparison plot with CI."""
    comparisons = result.get("comparisons", [])
    if not comparisons or not isinstance(comparisons, list):
        return None
    names = [c.get("group1", "") + " vs " + c.get("group2", "") for c in comparisons if isinstance(c, dict)]
    means = [c.get("mean_diff", 0) for c in comparisons if isinstance(c, dict)]
    ci_lowers = [c.get("ci_lower", 0) for c in comparisons if isinstance(c, dict)]
    ci_uppers = [c.get("ci_upper", 0) for c in comparisons if isinstance(c, dict)]
    sig_pairs = [(i, i) for i, c in enumerate(comparisons) if isinstance(c, dict) and c.get("significant")]
    return (
        means_comparison_plot(
            names, means, ci_lowers, ci_uppers, significant_pairs=sig_pairs, title="Multiple Comparison"
        )
        if names
        else None
    )


def _equivalence_handler(result, params):
    """TOST equivalence diagram."""
    ci = result.get("ci_90", [])
    delta = params.get("delta", 0)
    if ci and len(ci) == 2 and delta > 0:
        obs_diff = result.get("mean_diff", (ci[0] + ci[1]) / 2)
        return tost_plot(obs_diff, ci[0], ci[1], delta, title="TOST Equivalence")
    return None


def _power_handler(result, params):
    """Power curve."""
    effect_size = params.get("effect_size")
    if effect_size:
        n_values = list(range(5, 201, 5))
        return power_curve(effect_size, n_values, target_power=params.get("power", 0.8), title="Power Analysis")
    return None


def _multivariate_handler(result, params):
    """Heatmap for correlation matrix."""
    if result.get("analysis_type") == "correlation_matrix":
        matrix = result.get("correlation_matrix", [])
        labels = result.get("columns", [])
        if matrix:
            return heatmap(matrix, labels=labels, title="Correlation Matrix")
    return None


def _trend_handler(result, params):
    """Trend chart using control chart plot."""
    vals = result.get("_values", [])
    if vals:
        return control_chart_plot(
            vals,
            chart_type=result.get("test_type", "cusum"),
            ucl=result.get("ucl"),
            cl=result.get("cl"),
            lcl=result.get("lcl"),
        )
    return None


def _outlier_handler(result, params):
    """Outlier highlight plot."""
    vals = result.get("_values", [])
    outliers = result.get("outliers", [])
    outlier_indices = []
    if isinstance(outliers, list):
        for o in outliers:
            if isinstance(o, dict) and "index" in o:
                outlier_indices.append(o["index"])
            elif isinstance(o, (int, float)):
                outlier_indices.append(int(o))
    return outlier_plot(vals, outlier_indices=outlier_indices, title="Outlier Detection") if vals else None


def _reliability_handler(result, params):
    """Weibull probability plot."""
    times = params.get("times", [])
    if not times:
        return None
    params_dict = result.get("parameters", {})
    shape = params_dict.get("shape") if isinstance(params_dict, dict) else None
    scale = params_dict.get("scale") if isinstance(params_dict, dict) else None
    return weibull_plot(times, shape=shape, scale=scale, title=f"Weibull ({result.get('analysis_type', '')})")


def _gage_rr_handler(result, params):
    """Variance component bar chart."""
    contribution = result.get("contribution", {})
    if contribution and isinstance(contribution, dict):
        return variance_component_plot(contribution, title="Gage R&R - Variance Components")
    return None


def _explore_handler(result, params):
    """Missing values bar chart."""
    columns = result.get("columns", [])
    missing = result.get("missing_values", {})
    if columns and missing:
        counts = [missing.get(col, 0) for col in columns]
        return missing_values_plot(columns, counts, title="Data Overview - Missing Values")
    return None


def _doe_handler(result, params):
    """Pareto chart of factor effects."""
    effects = result.get("effects", {})
    if effects and isinstance(effects, dict):
        names = list(effects.keys())
        values = list(effects.values())
        return pareto_plot(values, names, title="DOE - Pareto Chart of Effects")
    return None


def _distribution_handler(result, params):
    """Chart for distribution fitting."""
    from stats_engine.charts import histogram

    values = result.get("_values") or params.get("values")
    if values is None:
        return None
    return histogram(values, title=f"Distribution: {result.get('distribution', 'fitted')}", normal_curve=True)


def _acceptance_sampling_handler(result, params):
    """Chart for acceptance sampling OC curve."""
    from stats_engine.charts import oc_curve_plot

    if "oc_curve" in result:
        curve = result["oc_curve"]
        return oc_curve_plot(curve.get("defect_rates", []), curve.get("accept_probs", []))
    return None


def _sensitivity_handler(result, params):
    """Chart for sensitivity analysis (tornado)."""
    from stats_engine.charts import tornado_plot

    if result.get("analysis_type") == "tornado" and "sensitivities" in result:
        sens = result["sensitivities"]
        variables = [s["variable"] for s in sens]
        swings = [s["swing"] for s in sens]
        return tornado_plot(variables, swings)
    return None


def _advanced_handler(result, params):
    """Chart for advanced analysis (bootstrap CI)."""
    if result.get("analysis_type") == "bootstrap":
        return bootstrap_plot(
            result.get("original_statistic", 0),
            result.get("ci_lower", 0),
            result.get("ci_upper", 0),
            result.get("bootstrap_mean", 0),
        )
    return None


def _bayesian_handler(result, params):
    """Chart for bayesian estimation."""
    if result.get("analysis_type") == "estimate":
        return posterior_plot(
            result.get("prior_mean", 0),
            result.get("prior_std", 1),
            result.get("posterior_mean", 0),
            result.get("posterior_std", 1),
            result.get("credible_interval", [0, 0]),
        )
    return None


def _functional_handler(result, params):
    """Chart for functional data analysis."""
    t = params.get("t")
    curves = params.get("curves")
    if t and curves:
        mean_curve = result.get("mean_function")
        return functional_plot(t, curves, mean_curve)
    return None


def _mining_handler(result, params):
    """Chart for mining results. Mining (classify/anomaly/associate) has no chart output."""
    return None


# Chart handler registry: command_name -> handler_function
CHART_HANDLERS = {
    "descriptive": lambda r, p: _histogram_handler(r, p, "Distribution"),
    "normality": _qq_handler,
    "control_chart": _control_chart_handler,
    "capability": _capability_handler,
    "correlation": _scatter_handler,
    "regression": _regression_handler,
    "timeseries": _timeseries_handler,
    "report": lambda r, p: _histogram_handler(r, p, "Report - Distribution"),
    "ttest": _ttest_handler,
    "anova": lambda r, p: _boxplot_handler(r, p, "ANOVA - Group Comparison"),
    "homogeneity": lambda r, p: _boxplot_handler(r, p, "Homogeneity of Variance"),
    "multiple_comparison": _multiple_comparison_handler,
    "equivalence": _equivalence_handler,
    "power": _power_handler,
    "multivariate": _multivariate_handler,
    "trend": _trend_handler,
    "outlier": _outlier_handler,
    "reliability": _reliability_handler,
    "gage_rr": _gage_rr_handler,
    "nonparametric": lambda r, p: _boxplot_handler(r, p, "Non-parametric - Group Comparison"),
    "explore": _explore_handler,
    "doe": _doe_handler,
    "distribution": _distribution_handler,
    "acceptance_sampling": _acceptance_sampling_handler,
    "sensitivity": _sensitivity_handler,
    "advanced": _advanced_handler,
    "bayesian": _bayesian_handler,
    "functional": _functional_handler,
    "mining": _mining_handler,
}

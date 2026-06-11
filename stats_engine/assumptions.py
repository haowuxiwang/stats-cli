"""Statistical assumption checking and method recommendation.

Automatically checks assumptions for statistical tests and recommends
appropriate methods based on data characteristics.
"""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def check_assumptions(values, test_type="ttest", values2=None, groups=None):
    """Check statistical assumptions and recommend methods.

    Args:
        values: Primary data values
        test_type: Type of test planned ('ttest', 'anova', 'correlation', etc.)
        values2: Second sample for two-sample tests
        groups: List of groups for ANOVA-type tests

    Returns:
        Dict with assumption checks, pass/fail, and recommendations
    """
    values = np.array(values, dtype=float)
    values = values[np.isfinite(values)]

    result = {
        "test_type": test_type,
        "assumptions": {},
        "recommendations": {},
        "warnings": [],
    }

    # Check normality
    if len(values) >= 3:
        normality_result = _check_normality(values)
        result["assumptions"]["normality"] = normality_result

    # Check sample size
    sample_size_check = _check_sample_size(values, test_type)
    result["assumptions"]["sample_size"] = sample_size_check

    # Check homogeneity of variance (for two-sample or multi-group tests)
    if test_type in ("ttest", "anova") and values2 is not None:
        values2 = np.array(values2, dtype=float)
        values2 = values2[np.isfinite(values2)]
        homogeneity_result = _check_homogeneity(values, values2)
        result["assumptions"]["homogeneity"] = homogeneity_result

    if test_type == "anova" and groups is not None and len(groups) >= 2:
        group_arrays = [np.array(g, dtype=float) for g in groups]
        group_arrays = [g[np.isfinite(g)] for g in group_arrays]
        homogeneity_result = _check_homogeneity_groups(group_arrays)
        result["assumptions"]["homogeneity"] = homogeneity_result

    # Generate recommendations based on assumption checks
    result["recommendations"] = _generate_recommendations(
        result["assumptions"], test_type, values, values2, groups
    )

    # Overall assessment
    all_passed = all(
        check.get("passed", True)
        for check in result["assumptions"].values()
        if isinstance(check, dict) and "passed" in check
    )
    result["overall_assumptions_met"] = all_passed

    return result


def _check_normality(values):
    """Check normality using Shapiro-Wilk test."""
    n = len(values)

    if n < 3:
        return {
            "test": "shapiro_wilk",
            "passed": None,
            "p_value": None,
            "message": "Too few observations for normality test (n < 3)",
        }

    # Shapiro-Wilk test (most powerful for n < 50)
    if n <= 5000:
        stat, p_value = sp_stats.shapiro(values)
        passed = p_value > 0.05
    else:
        # For large samples, use Anderson-Darling
        result = sp_stats.anderson(values, dist='norm')
        # Use 5% significance level (index 2)
        passed = result.statistic < result.critical_values[2]
        p_value = None

    return {
        "test": "shapiro_wilk" if n <= 5000 else "anderson_darling",
        "passed": passed,
        "p_value": r(p_value) if p_value is not None else None,
        "statistic": r(stat) if n <= 5000 else r(result.statistic),
        "interpretation": "Data appears normal" if passed else "Data may not be normal",
    }


def _check_sample_size(values, test_type):
    """Check if sample size is adequate for the planned test."""
    n = len(values)

    min_sizes = {
        "ttest": 2,
        "anova": 3,
        "correlation": 3,
        "regression": 3,
        "normality": 3,
        "capability": 25,
        "control_chart": 20,
    }

    recommended_sizes = {
        "ttest": 30,
        "anova": 30,
        "correlation": 30,
        "regression": 30,
        "normality": 20,
        "capability": 50,
        "control_chart": 30,
    }

    min_n = min_sizes.get(test_type, 2)
    rec_n = recommended_sizes.get(test_type, 30)

    passed = n >= min_n
    adequate = n >= rec_n

    return {
        "n": n,
        "minimum": min_n,
        "recommended": rec_n,
        "passed": passed,
        "adequate": adequate,
        "message": f"n={n}" + (
            f" (minimum {min_n} not met)" if not passed
            else f" (minimum {min_n} met, recommended {rec_n})" if not adequate
            else f" (recommended {rec_n} met)"
        ),
    }


def _check_homogeneity(values1, values2):
    """Check homogeneity of variance using Levene's test."""
    stat, p_value = sp_stats.levene(values1, values2)
    passed = p_value > 0.05

    return {
        "test": "levene",
        "passed": passed,
        "p_value": r(p_value),
        "statistic": r(stat),
        "interpretation": "Variances are equal" if passed else "Variances may be unequal",
    }


def _check_homogeneity_groups(groups):
    """Check homogeneity of variance for multiple groups."""
    stat, p_value = sp_stats.levene(*groups)
    passed = p_value > 0.05

    return {
        "test": "levene",
        "passed": passed,
        "p_value": r(p_value),
        "statistic": r(stat),
        "n_groups": len(groups),
        "interpretation": "Variances are equal across groups" if passed else "Variances may be unequal",
    }


def _generate_recommendations(assumptions, test_type, values, values2=None, groups=None):
    """Generate method recommendations based on assumption checks."""
    recommendations = {}

    normality = assumptions.get("normality", {})
    homogeneity = assumptions.get("homogeneity", {})

    if test_type == "ttest":
        if values2 is not None:
            # Two-sample test
            if normality.get("passed") is False:
                recommendations["primary"] = "mann_whitney"
                recommendations["reason"] = "Data is not normal, use non-parametric Mann-Whitney U test"
                recommendations["alternative"] = "ttest"
                recommendations["alternative_reason"] = "If you must use parametric, Welch t-test is robust to non-normality"
            elif homogeneity.get("passed") is False:
                recommendations["primary"] = "ttest"
                recommendations["variant"] = "welch"
                recommendations["reason"] = "Variances are unequal, use Welch's t-test (equal_var=False)"
            else:
                recommendations["primary"] = "ttest"
                recommendations["variant"] = "standard"
                recommendations["reason"] = "All assumptions met, standard t-test is appropriate"
        else:
            # One-sample test
            if normality.get("passed") is False:
                recommendations["primary"] = "wilcoxon"
                recommendations["reason"] = "Data is not normal, use Wilcoxon signed-rank test"
            else:
                recommendations["primary"] = "ttest"
                recommendations["variant"] = "one_sample"
                recommendations["reason"] = "Assumptions met for one-sample t-test"

    elif test_type == "anova":
        if normality.get("passed") is False:
            recommendations["primary"] = "kruskal_wallis"
            recommendations["reason"] = "Data is not normal, use Kruskal-Wallis test"
            recommendations["alternative"] = "anova"
            recommendations["alternative_reason"] = "ANOVA is robust to moderate non-normality with equal group sizes"
        elif homogeneity.get("passed") is False:
            recommendations["primary"] = "welch_anova"
            recommendations["reason"] = "Variances are unequal, use Welch's ANOVA"
        else:
            recommendations["primary"] = "anova"
            recommendations["variant"] = "one_way"
            recommendations["reason"] = "All assumptions met, standard ANOVA is appropriate"

    elif test_type == "correlation":
        if normality.get("passed") is False:
            recommendations["primary"] = "spearman"
            recommendations["reason"] = "Data is not normal, use Spearman rank correlation"
        else:
            recommendations["primary"] = "pearson"
            recommendations["reason"] = "Data is normal, Pearson correlation is appropriate"

    elif test_type == "capability":
        if normality.get("passed") is False:
            recommendations["warning"] = "Process capability assumes normality. Consider data transformation first."
            recommendations["transform_options"] = ["boxcox", "log", "sqrt"]

    return recommendations


def recommend_test(data_description):
    """Recommend appropriate statistical test based on data description.

    Args:
        data_description: Dict describing the analysis goal and data structure

    Returns:
        Dict with recommended test and rationale
    """
    goal = data_description.get("goal", "")
    n_groups = data_description.get("n_groups", 1)
    paired = data_description.get("paired", False)
    n_variables = data_description.get("n_variables", 1)
    outcome_type = data_description.get("outcome_type", "continuous")

    recommendations = []

    # Compare means
    if "compare" in goal.lower() or "difference" in goal.lower():
        if outcome_type == "categorical":
            recommendations.append({
                "test": "chi_square",
                "reason": "Categorical outcome, use chi-square test of independence",
            })
        elif n_groups == 2:
            if paired:
                recommendations.append({
                    "test": "paired_ttest",
                    "reason": "Two paired groups, use paired t-test",
                })
                recommendations.append({
                    "test": "wilcoxon",
                    "reason": "Non-parametric alternative to paired t-test",
                })
            else:
                recommendations.append({
                    "test": "two_sample_ttest",
                    "reason": "Two independent groups, use two-sample t-test",
                })
                recommendations.append({
                    "test": "mann_whitney",
                    "reason": "Non-parametric alternative to two-sample t-test",
                })
        elif n_groups >= 3:
            recommendations.append({
                "test": "one_way_anova",
                "reason": f"{n_groups} groups, use one-way ANOVA",
            })
            recommendations.append({
                "test": "kruskal_wallis",
                "reason": "Non-parametric alternative to ANOVA",
            })

    # Relationship/correlation
    elif "relationship" in goal.lower() or "correlation" in goal.lower():
        recommendations.append({
            "test": "pearson_correlation",
            "reason": "Measure linear relationship between two continuous variables",
        })
        recommendations.append({
            "test": "spearman_correlation",
            "reason": "Measure monotonic relationship (non-parametric)",
        })

    # Prediction
    elif "predict" in goal.lower() or "regression" in goal.lower():
        if n_variables == 1:
            recommendations.append({
                "test": "simple_linear_regression",
                "reason": "One predictor, use simple linear regression",
            })
        else:
            recommendations.append({
                "test": "multiple_regression",
                "reason": f"{n_variables} predictors, use multiple regression",
            })

    # Normality check
    elif "normal" in goal.lower() or "distribution" in goal.lower():
        recommendations.append({
            "test": "shapiro_wilk",
            "reason": "Test if data follows normal distribution",
        })
        recommendations.append({
            "test": "anderson_darling",
            "reason": "More powerful normality test for larger samples",
        })

    return {
        "goal": goal,
        "recommendations": recommendations,
        "top_recommendation": recommendations[0] if recommendations else None,
    }

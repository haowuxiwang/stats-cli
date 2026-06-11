"""Multi-step statistical workflow engine.

Orchestrates multiple statistical analyses with automatic assumption checking
and method recommendations.
"""

import numpy as np

from stats_engine.assumptions import check_assumptions


def workflow(steps, data=None, values=None, auto_check=True, alpha=0.05):
    """Execute multi-step statistical workflow.

    Args:
        steps: List of analysis steps, each a dict with 'command' and optional 'params'
        data: Input data file path
        values: Input data values (list or array)
        auto_check: Automatically check assumptions before hypothesis tests
        alpha: Significance level for assumption checks

    Returns:
        Dict with step results, assumptions, recommendations, and summary
    """
    from main import handler

    results = []
    assumptions_log = []
    recommendations_log = []
    warnings = []
    context = {"values": values, "data": data}

    for i, step in enumerate(steps):
        command = step.get("command")
        params = dict(step.get("params", {}))

        # Inject data from context if not provided in step
        if "values" not in params and "file" not in params:
            if "values" in context and context["values"] is not None:
                params["values"] = context["values"]
            elif "data" in context and context["data"] is not None:
                params["file"] = context["data"]

        # Auto-check assumptions before hypothesis tests
        if auto_check and command in ("ttest", "anova", "correlation", "nonparametric", "equivalence"):
            assumption_result = _check_step_assumptions(command, params, alpha)
            if assumption_result:
                assumptions_log.append({
                    "step": i,
                    "command": command,
                    "assumptions": assumption_result,
                })

                # Generate warnings if assumptions violated
                if not assumption_result.get("overall_assumptions_met", True):
                    warnings.append({
                        "step": i,
                        "command": command,
                        "message": "Assumptions may be violated. Check recommendations.",
                        "recommendations": assumption_result.get("recommendations", {}),
                    })

                    # Auto-adjust parameters if needed
                    params = _adjust_params_for_assumptions(
                        command, params, assumption_result
                    )

        # Execute the step
        try:
            result = handler({"command": command, "params": params})
            results.append({
                "step": i,
                "command": command,
                "status": result.get("status", "error"),
                "result": result.get("data", {}),
                "params_used": params,
            })

            # Update context with results for chaining
            _update_context(context, command, result.get("data", {}))

        except Exception as e:
            results.append({
                "step": i,
                "command": command,
                "status": "error",
                "error": str(e),
            })

    # Generate summary
    summary = _generate_summary(results, assumptions_log, warnings)

    return {
        "n_steps": len(steps),
        "steps_results": results,
        "assumptions": assumptions_log,
        "recommendations": recommendations_log,
        "warnings": warnings,
        "summary": summary,
        "auto_check_enabled": auto_check,
    }


def _check_step_assumptions(command, params, alpha):
    """Check assumptions for a specific step."""
    values = params.get("values")
    if values is None:
        return None

    values = np.array(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return None

    test_type = command
    values2 = params.get("values2")
    groups = params.get("groups")

    return check_assumptions(values, test_type, values2, groups)


def _adjust_params_for_assumptions(command, params, assumption_result):
    """Auto-adjust parameters based on assumption violations."""
    recommendations = assumption_result.get("recommendations", {})

    if command == "ttest":
        # If variances are unequal, use Welch's t-test
        if recommendations.get("variant") == "welch":
            params["equal_var"] = False
        # If data is not normal, suggest non-parametric
        elif recommendations.get("primary") == "mann_whitney":
            # Keep t-test but add warning
            pass

    elif command == "anova":
        # If variances are unequal, note it in params
        if recommendations.get("primary") == "welch_anova":
            params["welch"] = True

    return params


def _update_context(context, command, result):
    """Update workflow context with step results."""
    if command == "clean":
        if "clean_data" in result:
            context["values"] = result["clean_data"]
    elif command == "transform":
        if "values" in result:
            context["values"] = result["values"]
    elif command == "outlier":
        if "clean_data" in result:
            context["values"] = result["clean_data"]


def _generate_summary(results, assumptions_log, warnings):
    """Generate human-readable summary of workflow results."""
    summary_parts = []

    # Count successes and failures
    n_success = sum(1 for r in results if r.get("status") == "success")
    n_error = sum(1 for r in results if r.get("status") == "error")

    summary_parts.append(f"Completed {n_success}/{len(results)} steps successfully")
    if n_error > 0:
        summary_parts.append(f"{n_error} step(s) failed")

    # Summarize assumptions
    if assumptions_log:
        n_assumptions_met = sum(
            1 for a in assumptions_log
            if a.get("assumptions", {}).get("overall_assumptions_met", True)
        )
        summary_parts.append(
            f"Assumptions: {n_assumptions_met}/{len(assumptions_log)} steps met all assumptions"
        )

    # Summarize warnings
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s) generated")

    # Key findings from results
    key_findings = []
    for r in results:
        if r.get("status") == "success":
            cmd = r.get("command")
            res = r.get("result", {})

            if cmd == "descriptive" and "mean" in res:
                key_findings.append(f"Mean: {res['mean']:.4f}")
            elif cmd == "normality" and "is_normal" in res:
                normal_str = "normal" if res["is_normal"] else "not normal"
                key_findings.append(f"Data is {normal_str}")
            elif cmd == "ttest" and "significant" in res:
                sig_str = "significant" if res["significant"] else "not significant"
                key_findings.append(f"t-test: {sig_str} (p={res.get('p_value', 'N/A')})")
            elif cmd == "capability" and "cpk" in res:
                key_findings.append(f"Cpk: {res['cpk']:.4f}")

    if key_findings:
        summary_parts.append("Key findings: " + "; ".join(key_findings[:5]))

    return ". ".join(summary_parts) + "."


def pipeline(data, steps):
    """Simple pipeline: execute steps in sequence, passing output as input.

    Args:
        data: Initial data (values or file path)
        steps: List of commands to execute in sequence

    Returns:
        List of results from each step
    """
    from main import handler

    results = []
    current_data = data

    for step in steps:
        if isinstance(step, str):
            command = step
            params = {}
        else:
            command = step.get("command")
            params = dict(step.get("params", {}))

        # Inject current data
        if isinstance(current_data, list):
            params["values"] = current_data
        elif isinstance(current_data, str):
            params["file"] = current_data

        result = handler({"command": command, "params": params})
        results.append(result)

        # Update current_data for next step
        if result.get("status") == "success":
            data = result.get("data", {})
            if "clean_data" in data:
                current_data = data["clean_data"]
            elif "values" in data:
                current_data = data["values"]

    return results


def workflow_template(template_name, values=None, **kwargs):
    """Execute a predefined workflow template.

    Args:
        template_name: Name of the template to execute
        values: Input data values (not required for all templates)
        **kwargs: Additional parameters (usl, lsl, target, group2, etc.)

    Returns:
        Dict with workflow results
    """
    templates = {
        "manufacturing": _template_manufacturing,
        "comparison": _template_comparison,
        "capability": _template_capability,
        "exploration": _template_exploration,
        "reliability": _template_reliability,
        "doe": _template_doe,
        "timeseries": _template_timeseries,
        "regression": _template_regression,
        "multivariate": _template_multivariate,
    }

    if template_name not in templates:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available: {list(templates.keys())}"
        )

    return templates[template_name](values, **kwargs)


def _template_manufacturing(values, usl=None, lsl=None, target=None, **kwargs):
    """Manufacturing workflow: clean -> descriptive -> normality -> capability -> report."""
    steps = [
        {"command": "clean", "params": {"method": "drop"}},
        {"command": "descriptive"},
        {"command": "normality"},
    ]

    if usl is not None or lsl is not None:
        cap_params = {}
        if usl is not None:
            cap_params["usl"] = usl
        if lsl is not None:
            cap_params["lsl"] = lsl
        if target is not None:
            cap_params["target"] = target
        steps.append({"command": "capability", "params": cap_params})

    steps.append({"command": "report", "params": {
        "usl": usl, "lsl": lsl, "target": target
    }})

    return workflow(steps=steps, values=values, auto_check=True)


def _template_comparison(values, values2=None, groups=None, paired=False, **kwargs):
    """Comparison workflow: descriptive -> normality -> homogeneity -> ttest/anova."""
    steps = [
        {"command": "clean", "params": {"method": "drop"}},
        {"command": "descriptive"},
        {"command": "normality"},
    ]

    if groups is not None and len(groups) >= 3:
        steps.append({"command": "homogeneity", "params": {
            "test_type": "levene", "groups": groups
        }})
        steps.append({"command": "anova", "params": {
            "anova_type": "one_way", "groups": groups
        }})
    elif values2 is not None:
        steps.append({"command": "homogeneity", "params": {
            "test_type": "levene", "groups": [values, values2]
        }})
        test_type = "paired" if paired else "two_sample"
        steps.append({"command": "ttest", "params": {
            "test_type": test_type, "values": values, "values2": values2
        }})

    return workflow(steps=steps, values=values, auto_check=True)


def _template_capability(values, usl=None, lsl=None, target=None, **kwargs):
    """Capability workflow: clean -> descriptive -> normality -> capability."""
    steps = [
        {"command": "clean", "params": {"method": "drop"}},
        {"command": "descriptive"},
        {"command": "normality"},
    ]

    if usl is not None or lsl is not None:
        cap_params = {}
        if usl is not None:
            cap_params["usl"] = usl
        if lsl is not None:
            cap_params["lsl"] = lsl
        if target is not None:
            cap_params["target"] = target
        steps.append({"command": "capability", "params": cap_params})

    return workflow(steps=steps, values=values, auto_check=True)


def _template_exploration(values, **kwargs):
    """Exploration workflow: clean -> descriptive -> normality -> outlier."""
    steps = [
        {"command": "clean", "params": {"method": "drop"}},
        {"command": "descriptive"},
        {"command": "normality"},
        {"command": "outlier", "params": {"method": "grubbs"}},
    ]

    return workflow(steps=steps, values=values, auto_check=True)


def _template_reliability(values, **kwargs):
    """Reliability workflow: descriptive -> reliability(weibull).

    Args:
        values: Failure/censoring times (used as times parameter)
        **kwargs: times, status parameters
    """
    times = kwargs.get("times", values)
    status = kwargs.get("status", [1] * len(times))

    steps = [
        {"command": "descriptive", "params": {"values": times}},
        {"command": "reliability", "params": {
            "analysis_type": "weibull",
            "times": times,
            "status": status,
        }},
    ]

    return workflow(steps=steps, values=times, auto_check=False)


def _template_doe(values, **kwargs):
    """DOE workflow: doe(full_factorial) -> regression (if responses provided).

    Args:
        values: Not used (placeholder for template interface)
        **kwargs: factors, responses parameters
    """
    factors = kwargs.get("factors", [])
    responses = kwargs.get("responses")

    steps = [
        {"command": "doe", "params": {
            "doe_type": "full_factorial",
            "factors": factors,
        }},
    ]

    if responses is not None:
        steps.append({"command": "regression", "params": {
            "x": list(range(len(responses))),
            "y": responses,
            "reg_type": "linear",
        }})

    return workflow(steps=steps, values=responses or [], auto_check=False)


def _template_timeseries(values, **kwargs):
    """Time series workflow: descriptive -> timeseries(exp_smoothing).

    Args:
        values: Time series data
        **kwargs: frequency, n_forecast parameters
    """
    frequency = kwargs.get("frequency", 12)
    n_forecast = kwargs.get("n_forecast", 5)

    steps = [
        {"command": "descriptive", "params": {"values": values}},
        {"command": "timeseries", "params": {
            "analysis_type": "exp_smoothing",
            "values": values,
            "frequency": frequency,
            "n_forecast": n_forecast,
        }},
    ]

    return workflow(steps=steps, values=values, auto_check=False)


def _template_regression(values, **kwargs):
    """Regression workflow: correlation -> regression.

    Args:
        values: Dependent variable (y)
        **kwargs: x, y, reg_type parameters
    """
    x = kwargs.get("x", list(range(len(values))))
    y = kwargs.get("y", values)
    reg_type = kwargs.get("reg_type", "linear")

    steps = [
        {"command": "descriptive", "params": {"values": y}},
        {"command": "correlation", "params": {"x": x, "y": y}},
        {"command": "regression", "params": {
            "x": x,
            "y": y,
            "reg_type": reg_type,
        }},
    ]

    return workflow(steps=steps, values=y, auto_check=False)


def _template_multivariate(values, **kwargs):
    """Multivariate workflow: pca -> cluster.

    Args:
        values: 2D data matrix
        **kwargs: n_components, n_clusters parameters
    """
    n_components = kwargs.get("n_components")
    n_clusters = kwargs.get("n_clusters", 3)

    steps = [
        {"command": "multivariate", "params": {
            "analysis_type": "pca",
            "values": values,
            "n_components": n_components,
        }},
        {"command": "multivariate", "params": {
            "analysis_type": "cluster",
            "values": values,
            "n_clusters": n_clusters,
            "method": "kmeans",
        }},
    ]

    return workflow(steps=steps, values=values, auto_check=False)

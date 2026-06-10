"""Comprehensive analysis report generation."""


def report(values, usl=None, lsl=None, target=None, **kwargs):
    """Generate comprehensive analysis report.

    Args:
        values: Data values
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value

    Returns:
        Dict with comprehensive analysis results
    """
    from stats_engine.descriptive import descriptive
    from stats_engine.normality import normality
    from stats_engine.outlier import outlier

    result = {
        "analyses": {},
    }

    # Descriptive statistics
    result["analyses"]["descriptive"] = descriptive(values)

    # Normality test
    result["analyses"]["normality"] = normality(values)

    # Outlier detection
    result["analyses"]["outlier"] = outlier(values, method="grubbs")

    # Capability (if spec limits provided)
    if usl is not None or lsl is not None:
        from stats_engine.capability import capability

        result["analyses"]["capability"] = capability(values, usl=usl, lsl=lsl, target=target)

    # Control chart
    from stats_engine.control_chart import control_chart

    result["analyses"]["control_chart"] = control_chart("imr", values)

    # Summary
    desc = result["analyses"]["descriptive"]
    norm = result["analyses"]["normality"]
    result["summary"] = {
        "n": desc["n"],
        "mean": desc["mean"],
        "std": desc["std"],
        "is_normal": norm["is_normal"],
        "n_outliers": result["analyses"]["outlier"]["n_outliers"],
    }

    if "capability" in result["analyses"]:
        cap = result["analyses"]["capability"]
        result["summary"]["cpk"] = cap.get("cpk")
        result["summary"]["rating"] = cap.get("rating")

    return result

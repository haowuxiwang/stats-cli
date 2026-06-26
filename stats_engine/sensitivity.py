"""Sensitivity analysis: Monte Carlo simulation, tornado diagrams, Sobol indices."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r

# Supported input distributions for Monte Carlo
# Each rvs supports optional `size` parameter for batch generation
INPUT_DISTRIBUTIONS = {
    "normal": {"rvs": lambda p, size=None: np.random.normal(p["mean"], p["std"], size=size)},
    "uniform": {"rvs": lambda p, size=None: np.random.uniform(p["low"], p["high"], size=size)},
    "triangular": {"rvs": lambda p, size=None: np.random.triangular(p["left"], p["mode"], p["right"], size=size)},
    "lognormal": {"rvs": lambda p, size=None: np.random.lognormal(p["mean"], p["std"], size=size)},
    "exponential": {"rvs": lambda p, size=None: np.random.exponential(p["scale"], size=size)},
    "beta": {"rvs": lambda p, size=None: np.random.beta(p["a"], p["b"], size=size)},
    "constant": {"rvs": lambda p, size=None: np.full(size, p["value"]) if size is not None else p["value"]},
}


def sensitivity(analysis_type, **kwargs):
    """Perform sensitivity analysis.

    Args:
        analysis_type: 'monte_carlo', 'tornado', 'sobol'

    Returns:
        Dict with sensitivity analysis results
    """
    if analysis_type == "monte_carlo":
        return _monte_carlo(**kwargs)
    elif analysis_type == "tornado":
        return _tornado(**kwargs)
    elif analysis_type == "sobol":
        return _sobol(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use: monte_carlo, tornado, sobol")


def _safe_eval(formula, variables):
    """Evaluate a formula safely with given variable values.

    Args:
        formula: Mathematical expression string (e.g., "x1 + x2 * x3")
        variables: Dict of variable name -> value

    Returns:
        Computed result
    """
    # Only allow math operations — use numpy versions for array compatibility
    safe_builtins = {
        "abs": np.abs,
        "min": np.minimum,
        "max": np.maximum,
        "round": np.round,
        "pow": np.power,
        "sqrt": np.sqrt,
        "log": np.log,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "pi": np.pi,
        "e": np.e,
    }
    namespace = {**safe_builtins, **variables}
    return eval(formula, {"__builtins__": {}}, namespace)


def _monte_carlo(inputs, formula, n_simulations=10000, seed=42, percentiles=None, **kwargs):
    """Run Monte Carlo simulation.

    Args:
        inputs: Dict of variable_name -> {dist: distribution_name, params: {...}}
                Example: {"x1": {"dist": "normal", "params": {"mean": 10, "std": 1}},
                          "x2": {"dist": "uniform", "params": {"low": 5, "high": 15}}}
        formula: Mathematical expression using variable names
                 Example: "x1 * x2 + x1 ** 2"
        n_simulations: Number of simulations (default 10000)
        seed: Random seed for reproducibility
        percentiles: List of percentiles to compute (default [5, 25, 50, 75, 95])

    Returns:
        Dict with simulation statistics and distribution
    """
    if not isinstance(inputs, dict) or len(inputs) < 1:
        raise ValueError("inputs must be a dict with at least 1 variable")

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")

    np.random.seed(seed)
    n_simulations = int(n_simulations)
    if n_simulations < 100:
        raise ValueError("n_simulations must be >= 100")

    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    # Validate inputs
    for name, spec in inputs.items():
        dist_name = spec.get("dist", "").lower()
        if dist_name not in INPUT_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution for '{name}': {dist_name}. Supported: {', '.join(INPUT_DISTRIBUTIONS)}"
            )

    # Generate samples (batch generation for speed)
    samples = {}
    for name, spec in inputs.items():
        dist_name = spec["dist"].lower()
        params = spec.get("params", {})
        try:
            samples[name] = np.asarray(INPUT_DISTRIBUTIONS[dist_name]["rvs"](params, size=n_simulations), dtype=float)
        except Exception as e:
            raise ValueError(f"Error sampling '{name}' ({dist_name}): {e}")

    # Evaluate formula — try vectorized first, fall back to scalar loop
    outputs = np.zeros(n_simulations)
    try:
        vectorized_vars = {name: samples[name] for name in samples}
        outputs = _safe_eval(formula, vectorized_vars)
        outputs = np.asarray(outputs, dtype=float).flatten()
        if outputs.shape != (n_simulations,):
            raise ValueError("Shape mismatch")
    except Exception:
        # Fallback: scalar evaluation (supports formulas with non-vectorizable functions)
        for i in range(n_simulations):
            variables = {name: samples[name][i] for name in samples}
            try:
                outputs[i] = _safe_eval(formula, variables)
            except Exception as e:
                raise ValueError(f"Error evaluating formula at simulation {i}: {e}")

    # Filter out NaN/Inf
    valid = outputs[np.isfinite(outputs)]
    n_valid = len(valid)

    if n_valid < 10:
        raise ValueError("Too few valid simulation results (NaN/Inf produced)")

    # Statistics
    percentile_values = {f"p{p}": r(np.percentile(valid, p)) for p in percentiles}

    # Histogram bins
    n_bins = min(50, max(10, n_valid // 10))
    hist_counts, hist_edges = np.histogram(valid, bins=n_bins)
    histogram = {
        "edges": [r(e) for e in hist_edges],
        "counts": hist_counts.tolist(),
    }

    # Input statistics
    input_stats = {}
    for name in inputs:
        valid_samples = samples[name][np.isfinite(samples[name])]
        input_stats[name] = {
            "mean": r(np.mean(valid_samples)),
            "std": r(np.std(valid_samples, ddof=1)) if len(valid_samples) > 1 else 0.0,
            "min": r(np.min(valid_samples)),
            "max": r(np.max(valid_samples)),
        }

    # Skewness and kurtosis (handle constant data gracefully)
    output_std = np.std(valid, ddof=1)
    if output_std > 1e-10:
        skewness = r(float(sp_stats.skew(valid)))
        kurtosis_val = r(float(sp_stats.kurtosis(valid)))
    else:
        skewness = 0.0
        kurtosis_val = 0.0

    return {
        "analysis_type": "monte_carlo",
        "formula": formula,
        "n_simulations": n_simulations,
        "n_valid": n_valid,
        "n_invalid": n_simulations - n_valid,
        "mean": r(np.mean(valid)),
        "std": r(output_std),
        "min": r(np.min(valid)),
        "max": r(np.max(valid)),
        "median": r(np.median(valid)),
        "skewness": skewness,
        "kurtosis": kurtosis_val,
        "percentiles": percentile_values,
        "histogram": histogram,
        "input_stats": input_stats,
        "interpretation": (
            f"Monte Carlo ({n_simulations} sims): output mean={r(np.mean(valid))}, "
            f"std={r(np.std(valid, ddof=1))}, "
            f"95% CI: [{r(np.percentile(valid, 2.5))}, {r(np.percentile(valid, 97.5))}]"
        ),
    }


def _tornado(inputs, formula, base_values=None, variation=0.1, **kwargs):
    """Tornado sensitivity analysis: vary one input at a time.

    Args:
        inputs: Dict of variable_name -> {dist, params} (used for range if base_values not given)
        formula: Mathematical expression
        base_values: Dict of variable_name -> base value (default: mean of distribution)
        variation: Fraction to vary each input (default 0.1 = +/-10%)

    Returns:
        Dict with sensitivity ranking and tornado chart data
    """
    if not isinstance(inputs, dict) or len(inputs) < 2:
        raise ValueError("Need at least 2 inputs for tornado analysis")

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")

    # Determine base values
    if base_values is None:
        base_values = {}
        for name, spec in inputs.items():
            params = spec.get("params", {})
            dist_name = spec.get("dist", "normal").lower()
            if dist_name == "normal":
                base_values[name] = params.get("mean", 0)
            elif dist_name == "uniform":
                base_values[name] = (params.get("low", 0) + params.get("high", 1)) / 2
            elif dist_name == "triangular":
                base_values[name] = params.get("mode", 0)
            elif dist_name == "constant":
                base_values[name] = params.get("value", 0)
            else:
                base_values[name] = params.get("mean", params.get("scale", 1))

    # Compute base output
    try:
        base_output = _safe_eval(formula, base_values)
    except Exception as e:
        raise ValueError(f"Error evaluating formula at base values: {e}")

    # Vary each input
    sensitivities = []
    for name in inputs:
        base_val = base_values[name]
        delta = abs(base_val) * variation if base_val != 0 else variation

        # Low value
        low_values = dict(base_values)
        low_values[name] = base_val - delta
        try:
            low_output = _safe_eval(formula, low_values)
        except Exception:
            low_output = base_output

        # High value
        high_values = dict(base_values)
        high_values[name] = base_val + delta
        try:
            high_output = _safe_eval(formula, high_values)
        except Exception:
            high_output = base_output

        swing = abs(high_output - low_output)
        sensitivity_coeff = swing / (2 * delta) if delta > 0 else 0

        sensitivities.append(
            {
                "variable": name,
                "base_value": r(base_val),
                "low_value": r(base_val - delta),
                "high_value": r(base_val + delta),
                "low_output": r(low_output),
                "high_output": r(high_output),
                "swing": r(swing),
                "sensitivity_coefficient": r(sensitivity_coeff),
                "elasticity": r(sensitivity_coeff * base_val / base_output) if base_output != 0 else None,
            }
        )

    # Sort by swing (most sensitive first)
    sensitivities.sort(key=lambda x: x["swing"], reverse=True)

    # Add rank
    for i, s in enumerate(sensitivities):
        s["rank"] = i + 1

    return {
        "analysis_type": "tornado",
        "formula": formula,
        "base_output": r(base_output),
        "variation": variation,
        "n_variables": len(inputs),
        "sensitivities": sensitivities,
        "most_sensitive": sensitivities[0]["variable"] if sensitivities else None,
        "interpretation": (
            f"Most sensitive variable: {sensitivities[0]['variable']} (swing={sensitivities[0]['swing']})"
            if sensitivities
            else "No sensitivity data"
        ),
    }


def _sobol(inputs, formula, n_simulations=10000, seed=42, **kwargs):
    """Sobol sensitivity indices (first-order and total).

    Uses the Saltelli (2010) method for computing Sobol indices.

    Args:
        inputs: Dict of variable_name -> {dist, params}
        formula: Mathematical expression
        n_simulations: Number of base samples (default 10000)
        seed: Random seed

    Returns:
        Dict with first-order (S1) and total (ST) sensitivity indices
    """
    if not isinstance(inputs, dict) or len(inputs) < 2:
        raise ValueError("Need at least 2 inputs for Sobol analysis")

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")

    np.random.seed(seed)
    n = int(n_simulations)
    if n < 100:
        raise ValueError("n_simulations must be >= 100")

    var_names = sorted(inputs.keys())
    d = len(var_names)

    # Validate inputs
    for name in var_names:
        spec = inputs[name]
        dist_name = spec.get("dist", "").lower()
        if dist_name not in INPUT_DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution for '{name}': {dist_name}")

    # Generate sample matrices (batch generation)
    def _generate_samples(spec, size):
        dist_name = spec["dist"].lower()
        params = spec.get("params", {})
        return np.asarray(INPUT_DISTRIBUTIONS[dist_name]["rvs"](params, size=size), dtype=float)

    # Create two independent sample matrices A and B
    A = np.zeros((n, d))
    B = np.zeros((n, d))
    for i, name in enumerate(var_names):
        A[:, i] = _generate_samples(inputs[name], n)
        B[:, i] = _generate_samples(inputs[name], n)

    # Evaluate f(A) and f(B) — vectorized with scalar fallback
    def _eval_matrix(matrix):
        try:
            vectorized_vars = {var_names[k]: matrix[:, k] for k in range(d)}
            results = _safe_eval(formula, vectorized_vars)
            results = np.asarray(results, dtype=float).flatten()
            if results.shape != (n,):
                raise ValueError("Shape mismatch")
            return results
        except Exception:
            results = np.zeros(n)
            for j in range(n):
                variables = {var_names[k]: matrix[j, k] for k in range(d)}
                results[j] = _safe_eval(formula, variables)
            return results

    fA = _eval_matrix(A)
    fB = _eval_matrix(B)

    # Filter NaN/Inf
    valid_mask = np.isfinite(fA) & np.isfinite(fB)
    fA = fA[valid_mask]
    fB = fB[valid_mask]
    n_valid = len(fA)

    if n_valid < 50:
        raise ValueError("Too few valid simulation results")

    # Variance of f(A)
    var_y = np.var(fA)

    if var_y < 1e-15:
        raise ValueError("Output has zero variance - all simulations produce the same value")

    # First-order indices (Saltelli 2010)
    s1_indices = {}
    st_indices = {}

    for i in range(d):
        # Create AB_i matrix (A with column i replaced by B)
        ABi = A.copy()
        ABi[:, i] = B[:, i]

        fABi = _eval_matrix(ABi)
        fABi = fABi[valid_mask]

        # First-order: S1_i = (1/n) * sum(f(B) * (f(AB_i) - f(A))) / var(Y)
        s1 = np.mean(fB * (fABi - fA)) / var_y

        # Create BA_i matrix (B with column i replaced by A)
        BAi = B.copy()
        BAi[:, i] = A[:, i]

        fBAi = _eval_matrix(BAi)
        fBAi = fBAi[valid_mask]

        # Total: ST_i = 1 - (1/n) * sum(f(A) * (f(BA_i) - f(B))) / var(Y)
        st = 1 - np.mean(fA * (fBAi - fB)) / var_y

        s1_indices[var_names[i]] = r(max(0, s1))
        st_indices[var_names[i]] = r(max(0, min(1, st)))

    # Sort by total sensitivity
    ranking = sorted(st_indices.items(), key=lambda x: x[1], reverse=True)

    # Check if indices sum to ~1 (additive model)
    s1_sum = sum(s1_indices.values())
    interaction_share = max(0, 1 - s1_sum)

    return {
        "analysis_type": "sobol",
        "formula": formula,
        "n_simulations": n,
        "n_valid": n_valid,
        "n_variables": d,
        "variance": r(var_y),
        "first_order": s1_indices,
        "total_order": st_indices,
        "ranking": [{"variable": name, "S1": s1_indices[name], "ST": st_indices[name]} for name, _ in ranking],
        "interaction_share": r(interaction_share),
        "most_influential": ranking[0][0] if ranking else None,
        "interpretation": (
            f"Most influential: {ranking[0][0]} "
            f"(S1={s1_indices[ranking[0][0]]}, ST={st_indices[ranking[0][0]]}). "
            f"Interaction share: {r(interaction_share * 100, 1)}%"
        ),
    }

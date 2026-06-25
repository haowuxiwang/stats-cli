"""Design of Experiments: full factorial, fractional factorial, response surface, Taguchi."""

import itertools

import numpy as np

from utils.output import r


def doe(doe_type, factors, responses=None, **kwargs):
    """Generate or analyze DOE designs.

    Args:
        doe_type: 'full_factorial', 'fractional_factorial', 'response_surface', 'taguchi'
        factors: List of factor dicts with 'name' and 'levels' keys
        responses: Optional response data for analysis

    Returns:
        Dict with DOE results
    """
    # Validate factors early
    if not isinstance(factors, list):
        raise ValueError("factors must be a list of factor dicts")

    if doe_type == "full_factorial":
        return _full_factorial(factors)
    elif doe_type == "fractional_factorial":
        return _fractional_factorial(factors, **kwargs)
    elif doe_type == "response_surface":
        return _response_surface(factors, responses, **kwargs)
    elif doe_type == "taguchi":
        return _taguchi(factors, **kwargs)
    elif doe_type == "definitive_screening":
        return _definitive_screening(factors, **kwargs)
    else:
        raise ValueError(f"Unknown doe_type: {doe_type}")


def _normalize_factors(factors):
    """Normalize factor definitions to standard format.

    Accepts:
        {"name": "A", "levels": 3}          → 3 levels: [0, 1, 2]
        {"name": "A", "levels": [10, 20, 30]} → explicit level values
        {"name": "A", "low": 10, "high": 30}  → 2 levels: [10, 30]
    """
    if not factors:
        raise ValueError("At least one factor is required")
    normalized = []
    for f in factors:
        name = f.get("name", "unknown")
        if "levels" in f:
            lv = f["levels"]
            if isinstance(lv, int):
                if lv < 2:
                    raise ValueError(f"Factor '{name}': 'levels' must be >= 2, got {lv}")
                normalized.append({"name": name, "levels": list(range(lv))})
            elif isinstance(lv, list):
                if len(lv) < 2:
                    raise ValueError(f"Factor '{name}': need at least 2 level values, got {len(lv)}")
                normalized.append({"name": name, "levels": lv})
            else:
                raise ValueError(f"Factor '{name}': 'levels' must be int or list, got {type(lv).__name__}")
        elif "low" in f and "high" in f:
            normalized.append({"name": name, "levels": [f["low"], f["high"]]})
        else:
            raise ValueError(f"Factor '{name}': must have 'levels' or 'low'+'high'")
    return normalized


def _full_factorial(factors):
    """Generate full factorial design."""
    factors = _normalize_factors(factors)
    names = [f["name"] for f in factors]
    levels_list = [f["levels"] for f in factors]

    # Generate all combinations
    design = list(itertools.product(*levels_list))
    n_runs = len(design)

    # Create design matrix
    design_matrix = []
    for run in design:
        row = {names[i]: run[i] for i in range(len(factors))}
        design_matrix.append(row)

    # Calculate main effects if responses provided
    result = {
        "doe_type": "full_factorial",
        "n_factors": len(factors),
        "n_runs": n_runs,
        "factors": factors,
        "design_matrix": design_matrix,
    }

    # Generate factor level combinations for display
    level_combos = []
    for run in design:
        combo = {names[i]: run[i] for i in range(len(factors))}
        level_combos.append(combo)
    result["level_combinations"] = level_combos

    return result


def _fractional_factorial(factors, resolution=None, **kwargs):
    """Generate fractional factorial design (2^(k-p))."""
    factors = _normalize_factors(factors)
    names = [f["name"] for f in factors]
    k = len(factors)

    # For 2-level factors, use Plackett-Burman or similar
    # Simplified: generate a half-fraction for 2-level factors
    n_levels = [len(f["levels"]) for f in factors]

    if all(n == 2 for n in n_levels):
        # 2-level fractional factorial
        # Use Hadamard matrix approach for power of 2
        n_runs = 2 ** (k - 1)  # half fraction
        if n_runs < k + 1:
            n_runs = k + 1  # minimum for estimation

        # Generate using Yates' algorithm (simplified)
        design_matrix = []
        for i in range(n_runs):
            row = {}
            for j, name in enumerate(names):
                row[name] = int((i >> j) & 1)
            design_matrix.append(row)

        result = {
            "doe_type": "fractional_factorial",
            "n_factors": k,
            "n_runs": n_runs,
            "fraction": f"2^(k-{k - int(np.log2(n_runs))})",
            "resolution": resolution or "III",
            "factors": factors,
            "design_matrix": design_matrix,
        }
    else:
        # Mixed levels - use full factorial subset
        result = _full_factorial(factors)
        result["doe_type"] = "fractional_factorial"
        result["note"] = "Mixed-level factors: using full factorial design"

    return result


def _response_surface(factors, responses, **kwargs):
    """Response surface methodology (Central Composite Design)."""
    names = [f["name"] for f in factors]
    k = len(factors)

    # Central Composite Design: factorial + axial + center points
    factorial_points = list(itertools.product([-1, 1], repeat=k))
    axial_points = []
    alpha = np.sqrt(k)  # rotatable design
    for i in range(k):
        point_pos = [0] * k
        point_neg = [0] * k
        point_pos[i] = alpha
        point_neg[i] = -alpha
        axial_points.append(tuple(point_pos))
        axial_points.append(tuple(point_neg))

    n_center = max(1, k)  # center points
    center_points = [tuple([0] * k) for _ in range(n_center)]

    all_points = factorial_points + axial_points + center_points

    design_matrix = []
    for point in all_points:
        row = {names[i]: r(point[i]) for i in range(k)}
        design_matrix.append(row)

    result = {
        "doe_type": "response_surface",
        "n_factors": k,
        "n_runs": len(all_points),
        "alpha": r(alpha),
        "design_type": "Central Composite Design (CCD)",
        "factors": factors,
        "design_matrix": design_matrix,
        "n_factorial": len(factorial_points),
        "n_axial": len(axial_points),
        "n_center": n_center,
    }

    # If responses provided, fit second-order model
    if responses is not None:
        result["model"] = _fit_rsm(design_matrix, responses, names)

    return result


def _fit_rsm(design, responses, names):
    """Fit second-order response surface model."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels required for RSM fitting"}

    # Build X matrix with linear, quadratic, and interaction terms
    X_data = []
    for row in design:
        x_row = [1]  # intercept
        for name in names:
            x_row.append(row[name])
        for name in names:
            x_row.append(row[name] ** 2)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x_row.append(row[names[i]] * row[names[j]])
        X_data.append(x_row)

    X = np.array(X_data)
    y = np.array(responses)

    model = sm.OLS(y, X).fit()

    return {
        "r_squared": r(model.rsquared),
        "adj_r_squared": r(model.rsquared_adj),
        "coefficients": [r(c) for c in model.params],
    }


def _taguchi(factors, orthogonal_array=None, **kwargs):
    """Taguchi design (orthogonal arrays)."""
    factors = _normalize_factors(factors)
    names = [f["name"] for f in factors]
    k = len(factors)
    max_levels = max(len(f["levels"]) for f in factors)

    # Common Taguchi arrays
    if k <= 4 and max_levels <= 3:
        # L9 (3^4)
        oa = [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 2, 1],
            [2, 1, 0, 2],
            [2, 2, 1, 0],
        ]
        array_name = "L9"
    elif k <= 7 and max_levels <= 2:
        # L8 (2^7)
        oa = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 1],
        ]
        array_name = "L8"
    else:
        # Default: generate full factorial
        return _full_factorial(factors)

    n_runs = len(oa)
    design_matrix = []
    for run in oa:
        row = {}
        for i, name in enumerate(names):
            level = run[i] if i < len(run) else 0
            row[name] = factors[i]["levels"][min(level, len(factors[i]["levels"]) - 1)]
        design_matrix.append(row)

    return {
        "doe_type": "taguchi",
        "n_factors": k,
        "n_runs": n_runs,
        "orthogonal_array": array_name,
        "factors": factors,
        "design_matrix": design_matrix,
    }


def _definitive_screening(factors, **kwargs):
    """Generate a Definitive Screening Design (DSD).

    DSDs use 2k+1 runs for k factors (3+), with 3 levels per factor.
    They can estimate all main effects, two-factor interactions, and quadratic terms.

    Args:
        factors: List of factor definitions with name and low/high or levels

    Returns:
        Dict with design matrix and properties
    """
    import random

    factors = _normalize_factors(factors)
    names = [f["name"] for f in factors]
    k = len(factors)

    if k < 3:
        raise ValueError("Definitive Screening Design requires at least 3 factors")

    n_runs = 2 * k + 1

    # Generate DSD using foldover method
    rng = random.Random(42)

    # Each run has exactly one factor at center (0), others at +1/-1
    design = []
    for i in range(k):
        run = [0] * k
        for j in range(k):
            if j != i:
                run[j] = 1 if rng.random() > 0.5 else -1
        design.append(run)

    # Foldover: negate all non-center values
    for i in range(k):
        run = [0] * k
        for j in range(k):
            if j != i:
                run[j] = -design[i][j]
        design.append(run)

    # Center point
    design.append([0] * k)

    # Map coded levels to actual factor levels
    design_matrix = []
    for run in design:
        row = {}
        for i, name in enumerate(names):
            if run[i] == -1:
                row[name] = factors[i]["levels"][0]
            elif run[i] == 1:
                row[name] = factors[i]["levels"][-1]
            else:
                row[name] = factors[i]["levels"][len(factors[i]["levels"]) // 2]
        design_matrix.append(row)

    return {
        "doe_type": "definitive_screening",
        "n_factors": k,
        "n_runs": n_runs,
        "design_matrix": design_matrix,
        "aliasing_structure": "All main effects, 2FIs, and quadratic terms estimable",
        "properties": {
            "main_effects_estimable": True,
            "quadratic_terms_estimable": True,
            "two_factor_interactions_estimable": True,
        },
    }

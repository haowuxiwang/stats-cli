"""Process capability analysis: Cp, Cpk, Pp, Ppk, Cpm."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array

# d2 constants for subgroup sizes 2..10 (same as control_chart.py)
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
# c4 constants for S-bar method
_C4 = {
    2: 0.7979,
    3: 0.8862,
    4: 0.9213,
    5: 0.9400,
    6: 0.9515,
    7: 0.9594,
    8: 0.9650,
    9: 0.9693,
    10: 0.9727,
}


def capability(
    values, usl=None, lsl=None, target=None, capability_type="normal", sigma_method="mr", subgroup_size=None
):
    """Calculate process capability indices.

    Args:
        values: Data values
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (default: midpoint of specs)
        capability_type: 'normal', 'boxcox', or 'johnson'
        sigma_method: Method for within-subgroup std estimation:
            'mr' (default) - Moving Range (individual data)
            'rbar' - Subgroup R-bar (requires subgroup_size)
            'sbar' - Subgroup S-bar (requires subgroup_size)
        subgroup_size: Subgroup size for rbar/sbar methods (2-10)

    Returns:
        Dict with capability results
    """
    arr = to_array(values, min_n=2, name="values")
    n = len(arr)
    if usl is None and lsl is None:
        raise ValueError("At least one specification limit (USL or LSL) is required")
    if usl is not None and lsl is not None and usl <= lsl:
        raise ValueError(f"USL must be greater than LSL, got USL={usl}, LSL={lsl}")

    mean_val = float(np.mean(arr))
    std_overall = float(np.std(arr, ddof=1))

    if std_overall == 0:
        raise ValueError("Cannot calculate capability: all values are identical (zero variance)")

    # Within-subgroup std estimation
    if sigma_method == "mr":
        # Moving Range method (default, for individual data)
        mr = np.abs(np.diff(arr))
        mr_bar = float(np.mean(mr))
        std_within = mr_bar / _D2[2]
        sigma_desc = "Moving Range (d2=1.128)"
    elif sigma_method == "rbar":
        if subgroup_size is None or subgroup_size < 2 or subgroup_size > 10:
            raise ValueError("subgroup_size must be 2-10 for rbar method")
        # R-bar method: reshape into subgroups, compute mean range
        n_complete = (n // subgroup_size) * subgroup_size
        subgroups = arr[:n_complete].reshape(-1, subgroup_size)
        ranges = np.ptp(subgroups, axis=1)
        r_bar = float(np.mean(ranges))
        std_within = r_bar / _D2[subgroup_size]
        sigma_desc = f"R-bar (d2={_D2[subgroup_size]}, subgroup_size={subgroup_size})"
    elif sigma_method == "sbar":
        if subgroup_size is None or subgroup_size < 2 or subgroup_size > 10:
            raise ValueError("subgroup_size must be 2-10 for sbar method")
        n_complete = (n // subgroup_size) * subgroup_size
        subgroups = arr[:n_complete].reshape(-1, subgroup_size)
        s_vals = np.std(subgroups, axis=1, ddof=1)
        s_bar = float(np.mean(s_vals))
        c4 = _C4[subgroup_size]
        std_within = s_bar / c4
        sigma_desc = f"S-bar (c4={c4}, subgroup_size={subgroup_size})"
    else:
        raise ValueError(f"Unknown sigma_method: {sigma_method}. Use 'mr', 'rbar', or 'sbar'")

    # Default target
    if target is None:
        if usl is not None and lsl is not None:
            target = (usl + lsl) / 2
        else:
            target = mean_val

    result = {
        "mean": r(mean_val),
        "std_within": r(std_within),
        "std_overall": r(std_overall),
        "sigma_method": sigma_method,
        "sigma_method_desc": sigma_desc,
        "usl": usl,
        "lsl": lsl,
        "target": target,
        "n": n,
    }

    # Calculate capability indices
    if usl is not None and lsl is not None:
        # Two-sided spec
        cp = (usl - lsl) / (6 * std_within)
        pp = (usl - lsl) / (6 * std_overall)
        cpu = (usl - mean_val) / (3 * std_within)
        cpl = (mean_val - lsl) / (3 * std_within)
        cpk = min(cpu, cpl)
        ppu = (usl - mean_val) / (3 * std_overall)
        ppl = (mean_val - lsl) / (3 * std_overall)
        ppk = min(ppu, ppl)

        # Cpm (Taguchi)
        tau_sq = np.sum((arr - target) ** 2) / n
        cpm = (usl - lsl) / (6 * np.sqrt(tau_sq))

        result.update(
            {
                "cp": r(cp),
                "cpk": r(cpk),
                "cpu": r(cpu),
                "cpl": r(cpl),
                "pp": r(pp),
                "ppk": r(ppk),
                "ppu": r(ppu),
                "ppl": r(ppl),
                "cpm": r(cpm),
            }
        )
    elif usl is not None:
        # One-sided upper
        cpu = (usl - mean_val) / (3 * std_within)
        ppu = (usl - mean_val) / (3 * std_overall)
        result.update(
            {
                "cp": r(cpu),
                "cpk": r(cpu),
                "cpu": r(cpu),
                "cpl": None,
                "pp": r(ppu),
                "ppk": r(ppu),
                "ppu": r(ppu),
                "ppl": None,
                "cpm": None,
            }
        )
    else:
        # One-sided lower
        cpl = (mean_val - lsl) / (3 * std_within)
        ppl = (mean_val - lsl) / (3 * std_overall)
        result.update(
            {
                "cp": r(cpl),
                "cpk": r(cpl),
                "cpu": None,
                "cpl": r(cpl),
                "pp": r(ppl),
                "ppk": r(ppl),
                "ppu": None,
                "ppl": r(ppl),
                "cpm": None,
            }
        )

    # Confidence intervals
    if n > 2:
        cp_val = result.get("cp")
        if cp_val is not None:
            chi_lower = sp_stats.chi2.ppf(0.025, n - 1)
            chi_upper = sp_stats.chi2.ppf(0.975, n - 1)
            result["cp_ci_lower"] = r(cp_val * np.sqrt((n - 1) / chi_upper))
            result["cp_ci_upper"] = r(cp_val * np.sqrt((n - 1) / chi_lower))

        cpk_val = result.get("cpk")
        if cpk_val is not None:
            se_cpk = np.sqrt(1 / n + cpk_val**2 / (2 * (n - 1)))
            z_crit = sp_stats.norm.ppf(0.975)
            result["cpk_ci_lower"] = r(cpk_val - z_crit * se_cpk)
            result["cpk_ci_upper"] = r(cpk_val + z_crit * se_cpk)

    # Box-Cox transformation for non-normal data
    if capability_type == "boxcox" and n >= 5:
        result["boxcox"] = _boxcox_capability(arr, usl, lsl)

    # Johnson transformation for non-normal data
    if capability_type == "johnson" and n >= 5:
        result["johnson"] = _johnson_capability(arr, usl, lsl)

    # Rating
    cpk_val = result.get("cpk") or 0
    if cpk_val >= 1.67:
        result["rating"] = "Excellent"
        result["rating_desc"] = "Process is highly capable"
        decision_action = "RELEASE"
        decision_confidence = "HIGH"
    elif cpk_val >= 1.33:
        result["rating"] = "Good"
        result["rating_desc"] = "Process is capable (pharma minimum)"
        decision_action = "RELEASE"
        decision_confidence = "HIGH"
    elif cpk_val >= 1.0:
        result["rating"] = "Marginal"
        result["rating_desc"] = "Process is marginally capable, improvement recommended"
        decision_action = "CONDITIONAL_RELEASE"
        decision_confidence = "MEDIUM"
    else:
        result["rating"] = "Poor"
        result["rating_desc"] = "Process is NOT capable, corrective action required"
        decision_action = "HOLD"
        decision_confidence = "HIGH"

    # Structured decision block for AI agents
    result["decision"] = {
        "action": decision_action,
        "confidence": decision_confidence,
        "basis": [
            f"Cpk={r(cpk_val, 3)}",
            f"Cp={result.get('cp', 'N/A')}",
        ],
        "recommendation": (
            f"Process meets capability requirements (Cpk={r(cpk_val, 3)} {'≥' if cpk_val >= 1.33 else '<'} 1.33). "
            f"{'Safe to release batch.' if cpk_val >= 1.33 else 'Investigate process centering or reduce variation before release.'}"
        ),
    }

    # Performance metrics (PPM)
    if usl is not None and lsl is not None:
        z_upper = (usl - mean_val) / std_overall
        z_lower = (mean_val - lsl) / std_overall
        ppm_upper = sp_stats.norm.cdf(-z_upper) * 1e6
        ppm_lower = sp_stats.norm.cdf(-z_lower) * 1e6
        ppm_total = ppm_upper + ppm_lower
        result["performance"] = {
            "z_upper": r(z_upper),
            "z_lower": r(z_lower),
            "ppm_upper": r(ppm_upper, 2),
            "ppm_lower": r(ppm_lower, 2),
            "ppm_total": r(ppm_total, 2),
            "yield_pct": r(100 - ppm_total / 10000),
        }

    # Interpretation
    parts = []
    if result.get("cp") is not None:
        parts.append(f"Cp = {result['cp']}")
    if result.get("cpk") is not None:
        parts.append(f"Cpk = {result['cpk']}")
    if result.get("cpm") is not None:
        parts.append(f"Cpm = {result['cpm']}")
    parts.append(result["rating_desc"])
    if "performance" in result:
        parts.append(f"Expected yield = {result['performance']['yield_pct']}%")
    result["interpretation"] = ". ".join(parts)

    if n < 25:
        result["_warning"] = f"n={n}: capability analysis requires n>=25 for reliable results"

    return result


def _boxcox_capability(arr, usl, lsl):
    """Box-Cox transformation capability analysis."""
    from scipy.stats import boxcox

    # Handle non-positive values
    if np.any(arr <= 0):
        offset = abs(np.min(arr)) + 1
        shifted = arr + offset
    else:
        offset = 0
        shifted = arr

    # Find optimal lambda
    transformed, optimal_lambda = boxcox(shifted)

    mean_t = float(np.mean(transformed))
    sd_t = float(np.std(transformed, ddof=1))

    # Transform spec limits using Box-Cox formula: (x^lambda - 1) / lambda
    def _boxcox_transform(x, lam, off):
        x_shifted = x + off
        if abs(lam) < 1e-10:
            return np.log(x_shifted)
        return (x_shifted**lam - 1) / lam

    usl_t = _boxcox_transform(usl, optimal_lambda, offset) if usl is not None else None
    lsl_t = _boxcox_transform(lsl, optimal_lambda, offset) if lsl is not None else None

    result = {
        "lambda": r(optimal_lambda, 2),
        "offset": offset,
        "mean_transformed": r(mean_t, 6),
        "std_transformed": r(sd_t, 6),
    }

    if usl_t is not None and lsl_t is not None:
        cp_bc = (usl_t - lsl_t) / (6 * sd_t)
        cpu_bc = (usl_t - mean_t) / (3 * sd_t)
        cpl_bc = (mean_t - lsl_t) / (3 * sd_t)
        cpk_bc = min(cpu_bc, cpl_bc)
        result.update(
            {
                "cp": r(cp_bc),
                "cpk": r(cpk_bc),
                "cpu": r(cpu_bc),
                "cpl": r(cpl_bc),
            }
        )

    return result


def _johnson_capability(arr, usl, lsl):
    """Johnson SU transformation capability analysis.

    Uses CDF-then-PPF approach: transform data through Johnson SU CDF to
    uniform, then through normal PPF to get normal scores. Spec limits are
    transformed the same way. Capability indices are computed on the
    transformed (normal) data.
    """
    # Fit Johnson SU distribution
    params = sp_stats.johnsonsu.fit(arr)
    gamma, delta, xi, lam = params

    # Transform data: Johnson CDF -> normal PPF
    uniform = sp_stats.johnsonsu.cdf(arr, *params)
    # Clamp to avoid +/-inf from norm.ppf at 0 or 1
    eps = 1e-10
    uniform = np.clip(uniform, eps, 1 - eps)
    transformed = sp_stats.norm.ppf(uniform)

    mean_t = float(np.mean(transformed))
    sd_t = float(np.std(transformed, ddof=1))

    # Transform spec limits using the same CDF-then-PPF approach
    usl_t = None
    lsl_t = None
    if usl is not None:
        u_unif = float(sp_stats.johnsonsu.cdf(usl, *params))
        u_unif = np.clip(u_unif, eps, 1 - eps)
        usl_t = float(sp_stats.norm.ppf(u_unif))
    if lsl is not None:
        l_unif = float(sp_stats.johnsonsu.cdf(lsl, *params))
        l_unif = np.clip(l_unif, eps, 1 - eps)
        lsl_t = float(sp_stats.norm.ppf(l_unif))

    result = {
        "gamma": r(gamma, 4),
        "delta": r(delta, 4),
        "xi": r(xi, 4),
        "lambda": r(lam, 4),
        "mean_transformed": r(mean_t, 6),
        "std_transformed": r(sd_t, 6),
    }

    if usl_t is not None and lsl_t is not None and sd_t > 0:
        cp_j = (usl_t - lsl_t) / (6 * sd_t)
        cpu_j = (usl_t - mean_t) / (3 * sd_t)
        cpl_j = (mean_t - lsl_t) / (3 * sd_t)
        cpk_j = min(cpu_j, cpl_j)
        result.update(
            {
                "cp": r(cp_j),
                "cpk": r(cpk_j),
                "cpu": r(cpu_j),
                "cpl": r(cpl_j),
            }
        )

    return result

"""Process capability analysis: Cp, Cpk, Pp, Ppk, Cpm."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def capability(values, usl=None, lsl=None, target=None, capability_type="normal"):
    """Calculate process capability indices.

    Args:
        values: Data values
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (default: midpoint of specs)
        capability_type: 'normal' or 'boxcox'

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

    # Within-subgroup std using moving range
    if n >= 2:
        mr = np.abs(np.diff(arr))
        mr_bar = float(np.mean(mr))
        std_within = mr_bar / 1.128
    else:
        std_within = std_overall

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

    # Rating
    cpk_val = result.get("cpk") or 0
    if cpk_val >= 1.67:
        result["rating"] = "Excellent"
        result["rating_desc"] = "Process is highly capable"
    elif cpk_val >= 1.33:
        result["rating"] = "Good"
        result["rating_desc"] = "Process is capable (pharma minimum)"
    elif cpk_val >= 1.0:
        result["rating"] = "Marginal"
        result["rating_desc"] = "Process is marginally capable, improvement recommended"
    else:
        result["rating"] = "Poor"
        result["rating_desc"] = "Process is NOT capable, corrective action required"

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
        "sd_transformed": r(sd_t, 6),
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

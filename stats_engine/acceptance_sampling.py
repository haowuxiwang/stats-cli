"""Acceptance sampling: single plan, double plan, OC curve, plan finding."""

import numpy as np
from scipy.stats import binom, hypergeom

from utils.output import r


def acceptance_sampling(analysis_type, **kwargs):
    """Perform acceptance sampling analysis.

    Args:
        analysis_type: 'single_plan', 'double_plan', 'oc_curve', 'find_plan'

    Returns:
        Dict with acceptance sampling results
    """
    if analysis_type == "single_plan":
        return _single_plan(**kwargs)
    elif analysis_type == "double_plan":
        return _double_plan(**kwargs)
    elif analysis_type == "oc_curve":
        return _oc_curve(**kwargs)
    elif analysis_type == "find_plan":
        return _find_plan(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


# ---------------------------------------------------------------------------
# Single sampling plan
# ---------------------------------------------------------------------------


def _single_plan(n, c, defect_rate, lot_size=None, **kwargs):
    """Compute single sampling plan metrics.

    Args:
        n: Sample size (positive integer)
        c: Accept number (non-negative integer, c < n)
        defect_rate: Fraction defective (0 < defect_rate < 1)
        lot_size: If provided, use hypergeometric model; otherwise binomial

    Returns:
        Dict with accept_prob, reject_prob, oc_curve, aoq, ati
    """
    # --- input validation ---------------------------------------------------
    _validate_positive_int(n, "n")
    _validate_non_negative_int(c, "c")
    if c >= n:
        raise ValueError(f"c ({c}) must be less than n ({n})")
    _validate_defect_rate(defect_rate)
    if lot_size is not None:
        _validate_positive_int(lot_size, "lot_size")
        if lot_size < n:
            raise ValueError(f"lot_size ({lot_size}) must be >= n ({n})")

    # --- probabilities ------------------------------------------------------
    if lot_size is not None:
        # Hypergeometric model
        D = max(1, round(lot_size * defect_rate))
        accept_prob = hypergeom.cdf(c, lot_size, D, n)
        # OC curve points
        p_range = np.linspace(0.001, 0.30, 50)
        oc_points = []
        for p in p_range:
            d = max(1, round(lot_size * p))
            pa = hypergeom.cdf(c, lot_size, d, n)
            oc_points.append({"defect_rate": r(p), "accept_prob": r(pa)})
    else:
        # Binomial model
        accept_prob = binom.cdf(c, n, defect_rate)
        p_range = np.linspace(0.001, 0.30, 50)
        oc_points = [{"defect_rate": r(p), "accept_prob": r(binom.cdf(c, n, p))} for p in p_range]

    reject_prob = 1.0 - accept_prob

    # AOQ = Pa * p * (N - n) / N   (finite lot)  or  Pa * p  (infinite)
    if lot_size is not None:
        aoq = accept_prob * defect_rate * (lot_size - n) / lot_size
    else:
        aoq = accept_prob * defect_rate

    # ATI = n * Pa + N * (1 - Pa)  (inspect whole lot on rejection)
    if lot_size is not None:
        ati = n * accept_prob + lot_size * reject_prob
    else:
        ati = n  # no finite lot, always inspect n

    return {
        "analysis_type": "single_plan",
        "n": n,
        "c": c,
        "defect_rate": defect_rate,
        "lot_size": lot_size,
        "model": "hypergeometric" if lot_size else "binomial",
        "accept_prob": r(accept_prob),
        "reject_prob": r(reject_prob),
        "aoq": r(aoq),
        "ati": r(ati),
        "oc_curve": oc_points,
        "interpretation": (
            f"Single plan n={n}, c={c}: P(accept)={r(accept_prob)} at p={defect_rate}. AOQ={r(aoq)}, ATI={r(ati)}."
        ),
    }


# ---------------------------------------------------------------------------
# Double sampling plan
# ---------------------------------------------------------------------------


def _double_plan(n1, c1, d1, n2, c2, defect_rate, lot_size=None, **kwargs):
    """Compute double sampling plan metrics.

    Decision rules:
    - First sample: accept if x1 <= c1, reject if x1 >= d1, otherwise take
      second sample.
    - After second sample: accept if x1+x2 <= c2, else reject.

    Args:
        n1: First sample size
        c1: Accept number for first sample
        d1: Reject number for first sample (d1 > c1)
        n2: Second sample size
        c2: Combined accept number (c2 >= c1)
        defect_rate: Fraction defective
        lot_size: Optional (not used for probability; kept for consistency)
    """
    # --- input validation ---------------------------------------------------
    _validate_positive_int(n1, "n1")
    _validate_non_negative_int(c1, "c1")
    _validate_positive_int(d1, "d1")
    _validate_positive_int(n2, "n2")
    _validate_non_negative_int(c2, "c2")
    _validate_defect_rate(defect_rate)
    if d1 <= c1:
        raise ValueError(f"d1 ({d1}) must be > c1 ({c1})")
    if c2 < c1:
        raise ValueError(f"c2 ({c2}) must be >= c1 ({c1})")
    if lot_size is not None:
        _validate_positive_int(lot_size, "lot_size")

    # --- stage 1 probabilities ----------------------------------------------
    # P(accept on first sample) = P(x1 <= c1)
    pa1 = binom.cdf(c1, n1, defect_rate)
    # P(reject on first sample) = P(x1 >= d1) = 1 - P(x1 <= d1-1)
    pr1 = 1.0 - binom.cdf(d1 - 1, n1, defect_rate)
    # P(go to second sample) = 1 - pa1 - pr1
    p_continue = 1.0 - pa1 - pr1

    # --- stage 2 probabilities ----------------------------------------------
    # P(accept on second sample | went to second)
    # = sum_{x1=c1+1}^{d1-1} P(x1) * P(x2 <= c2 - x1)
    pa2 = 0.0
    for x1 in range(c1 + 1, d1):
        p_x1 = binom.pmf(x1, n1, defect_rate)
        max_x2 = c2 - x1
        if max_x2 >= 0:
            pa2 += p_x1 * binom.cdf(max_x2, n2, defect_rate)
        # else: impossible to accept, contribution is 0

    total_accept = pa1 + pa2
    total_reject = 1.0 - total_accept

    # --- ASN (Average Sample Number) ----------------------------------------
    asn = n1 + p_continue * n2

    # --- OC curve -----------------------------------------------------------
    p_range = np.linspace(0.001, 0.30, 50)
    oc_points = []
    for p in p_range:
        pa1_p = binom.cdf(c1, n1, p)
        pa2_p = 0.0
        for x1 in range(c1 + 1, d1):
            p_x1 = binom.pmf(x1, n1, p)
            max_x2 = c2 - x1
            if max_x2 >= 0:
                pa2_p += p_x1 * binom.cdf(max_x2, n2, p)
        oc_points.append({"defect_rate": r(p), "accept_prob": r(pa1_p + pa2_p)})

    return {
        "analysis_type": "double_plan",
        "n1": n1,
        "c1": c1,
        "d1": d1,
        "n2": n2,
        "c2": c2,
        "defect_rate": defect_rate,
        "prob_accept_first": r(pa1),
        "prob_reject_first": r(pr1),
        "prob_continue": r(p_continue),
        "prob_accept_second": r(pa2),
        "total_accept_prob": r(total_accept),
        "total_reject_prob": r(total_reject),
        "asn": r(asn),
        "oc_curve": oc_points,
        "interpretation": (
            f"Double plan ({n1},{c1},{d1})+({n2},{c2}): P(accept)={r(total_accept)} at p={defect_rate}. ASN={r(asn)}."
        ),
    }


# ---------------------------------------------------------------------------
# OC curve generation
# ---------------------------------------------------------------------------


def _oc_curve(n, c, defect_rate_range=None, lot_size=None, **kwargs):
    """Generate an Operating Characteristic curve.

    Args:
        n: Sample size
        c: Accept number
        defect_rate_range: [low, high] or list of rates (default [0, 0.30])
        lot_size: If provided, use hypergeometric model
    """
    _validate_positive_int(n, "n")
    _validate_non_negative_int(c, "c")
    if c >= n:
        raise ValueError(f"c ({c}) must be less than n ({n})")

    if defect_rate_range is None:
        defect_rate_range = [0.0, 0.30]

    if isinstance(defect_rate_range, (list, tuple)) and len(defect_rate_range) == 2:
        p_values = np.linspace(
            max(defect_rate_range[0], 0.001),
            defect_rate_range[1],
            50,
        )
    else:
        p_values = np.array(defect_rate_range, dtype=float)

    oc_points = []
    for p in p_values:
        if lot_size is not None:
            d = max(1, round(lot_size * p))
            pa = hypergeom.cdf(c, lot_size, d, n)
        else:
            pa = binom.cdf(c, n, p)
        oc_points.append({"defect_rate": r(p), "accept_prob": r(pa)})

    return {
        "analysis_type": "oc_curve",
        "n": n,
        "c": c,
        "model": "hypergeometric" if lot_size else "binomial",
        "n_points": len(oc_points),
        "oc_curve": oc_points,
        "interpretation": f"OC curve for n={n}, c={c} with {len(oc_points)} points.",
    }


# ---------------------------------------------------------------------------
# Find sampling plan
# ---------------------------------------------------------------------------


def _find_plan(AQL, LTPD, alpha=0.05, beta=0.10, **kwargs):
    """Find (n, c) satisfying producer's and consumer's risk constraints.

    Searches c from 1 upward; for each c finds the minimum n such that:
        P(accept | p=AQL) >= 1 - alpha
        P(accept | p=LTPD) <= beta

    Args:
        AQL: Acceptable Quality Level (producer's quality)
        LTPD: Lot Tolerance Percent Defective (consumer's limit)
        alpha: Producer's risk (default 0.05)
        beta: Consumer's risk (default 0.10)
    """
    _validate_defect_rate(AQL)
    _validate_defect_rate(LTPD)
    if LTPD <= AQL:
        raise ValueError(f"LTPD ({LTPD}) must be > AQL ({AQL})")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not (0 < beta < 1):
        raise ValueError(f"beta must be between 0 and 1, got {beta}")

    for c in range(1, 200):
        # Binary search for n
        lo, hi = c + 1, 200_000
        best_n = None
        while lo <= hi:
            mid = (lo + hi) // 2
            pa_aql = binom.cdf(c, mid, AQL)
            pa_ltpd = binom.cdf(c, mid, LTPD)
            if pa_aql >= 1 - alpha and pa_ltpd <= beta:
                best_n = mid
                hi = mid - 1  # try smaller n
            elif pa_ltpd > beta:
                lo = mid + 1  # need larger n to reduce Pa at LTPD
            else:
                hi = mid - 1  # Pa at AQL too low, try smaller n

        if best_n is not None:
            # Verify constraints
            pa_aql = binom.cdf(c, best_n, AQL)
            pa_ltpd = binom.cdf(c, best_n, LTPD)
            return {
                "analysis_type": "find_plan",
                "n": best_n,
                "c": c,
                "AQL": AQL,
                "LTPD": LTPD,
                "alpha": alpha,
                "beta": beta,
                "pa_at_aql": r(pa_aql),
                "pa_at_ltpd": r(pa_ltpd),
                "interpretation": (
                    f"Recommended plan: n={best_n}, c={c}. "
                    f"P(accept|AQL)={r(pa_aql)} (>= {1 - alpha}), "
                    f"P(accept|LTPD)={r(pa_ltpd)} (<= {beta})."
                ),
            }

    raise ValueError("No feasible plan found within search limits (c up to 200, n up to 200000)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_positive_int(val, name):
    if not isinstance(val, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {type(val).__name__}")
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")


def _validate_non_negative_int(val, name):
    if not isinstance(val, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer, got {type(val).__name__}")
    if val < 0:
        raise ValueError(f"{name} must be non-negative, got {val}")


def _validate_defect_rate(p):
    if not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError(f"defect_rate must be numeric, got {type(p).__name__}")
    if not (0 < p < 1):
        raise ValueError(f"defect_rate must be between 0 and 1 (exclusive), got {p}")

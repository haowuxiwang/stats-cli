"""Reliability and survival analysis: Weibull, Kaplan-Meier, distribution fitting, stability."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def reliability(analysis_type, **kwargs):
    """Perform reliability/survival analysis.

    Args:
        analysis_type: 'weibull', 'kaplan_meier', 'distribution', 'stability'

    Returns:
        Dict with reliability results
    """
    if analysis_type == "weibull":
        return _weibull(**kwargs)
    elif analysis_type == "kaplan_meier":
        return _kaplan_meier(**kwargs)
    elif analysis_type == "distribution":
        # Accept either 'times' or 'values' as input
        if "values" in kwargs and "times" not in kwargs:
            kwargs["times"] = kwargs.pop("values")
        return _distribution_fit(**kwargs)
    elif analysis_type == "stability":
        return _shelf_life(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _weibull(times, status=None, **kwargs):
    """Weibull analysis."""
    times = np.array(times, dtype=float)
    n = len(times)

    if status is None:
        status = np.ones(n)
    status = np.array(status, dtype=float)

    # Fit Weibull distribution using MLE
    # Weibull: f(t) = (beta/eta) * (t/eta)^(beta-1) * exp(-(t/eta)^beta)
    def neg_log_likelihood(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e10
        ll = 0
        for i in range(n):
            if status[i] == 1:  # failure
                ll += np.log(beta / eta) + (beta - 1) * np.log(times[i] / eta) - (times[i] / eta) ** beta
            else:  # censored
                ll -= (times[i] / eta) ** beta
        return -ll

    # Initial estimates
    beta_init = 1.0
    eta_init = np.mean(times)

    from scipy.optimize import minimize

    result = minimize(neg_log_likelihood, [beta_init, eta_init], bounds=[(0.01, 100), (0.01, max(times) * 10)])
    beta, eta = result.x

    # B-lives (reliability at specific times)
    b_lives = {}
    for b in [1, 5, 10, 50]:
        t_b = eta * (-np.log(1 - b / 100)) ** (1 / beta)
        b_lives[f"B{b}"] = r(t_b)

    # MTTF
    from scipy.special import gamma as gamma_func

    mttf = eta * gamma_func(1 + 1 / beta)

    return {
        "analysis_type": "weibull",
        "n": n,
        "n_failures": int(np.sum(status)),
        "n_censored": int(np.sum(1 - status)),
        "beta": r(beta),
        "eta": r(eta),
        "mttf": r(mttf),
        "b_lives": b_lives,
        "reliability_at_mttf": r(np.exp(-((mttf / eta) ** beta))),
        "interpretation": f"Weibull beta={r(beta, 2)}, eta={r(eta, 2)}, MTTF={r(mttf, 2)}",
    }


def _kaplan_meier(times, status=None, **kwargs):
    """Kaplan-Meier survival analysis."""
    times = np.array(times, dtype=float)
    if status is None:
        status = np.ones(len(times))
    status = np.array(status, dtype=float)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    status = status[order]

    n = len(times)
    unique_times = np.unique(times)

    survival = 1.0
    km_table = []
    at_risk = n

    for t in unique_times:
        events = np.sum((times == t) & (status == 1))
        censored = np.sum((times == t) & (status == 0))

        if events > 0:
            survival *= (at_risk - events) / at_risk

        km_table.append(
            {
                "time": r(t),
                "at_risk": at_risk,
                "events": int(events),
                "censored": int(censored),
                "survival": r(survival),
            }
        )

        at_risk -= events + censored

    # Median survival time
    median_time = None
    for row in km_table:
        if row["survival"] <= 0.5:
            median_time = row["time"]
            break

    return {
        "analysis_type": "kaplan_meier",
        "n": n,
        "n_events": int(np.sum(status)),
        "n_censored": int(np.sum(1 - status)),
        "median_survival": median_time,
        "km_table": km_table,
        "interpretation": f"Median survival time = {median_time}"
        if median_time
        else "Median survival time not reached",
    }


def _distribution_fit(times, status=None, **kwargs):
    """Fit multiple distributions and compare."""
    times = np.array(times, dtype=float)
    n = len(times)

    results = {}

    # Weibull
    try:
        weibull = _weibull(times, status)
        results["weibull"] = {
            "parameters": {"beta": weibull["beta"], "eta": weibull["eta"]},
            "mttf": weibull["mttf"],
        }
    except Exception as e:
        results["weibull"] = {"error": str(e)}

    # Lognormal
    try:
        log_times = np.log(times[times > 0])
        mu, sigma = sp_stats.norm.fit(log_times)
        mttf = np.exp(mu + sigma**2 / 2)
        results["lognormal"] = {
            "parameters": {"mu": r(mu), "sigma": r(sigma)},
            "mttf": r(mttf),
        }
    except Exception as e:
        results["lognormal"] = {"error": str(e)}

    # Exponential
    try:
        scale = np.mean(times)
        results["exponential"] = {
            "parameters": {"lambda": r(1 / scale)},
            "mttf": r(scale),
        }
    except Exception as e:
        results["exponential"] = {"error": str(e)}

    # Normal
    try:
        mu, sigma = sp_stats.norm.fit(times)
        results["normal"] = {
            "parameters": {"mu": r(mu), "sigma": r(sigma)},
            "mttf": r(mu),
        }
    except Exception as e:
        results["normal"] = {"error": str(e)}

    # Determine best fit (prefer weibull, then lognormal, then exponential, then normal)
    best_fit = None
    for candidate in ["weibull", "lognormal", "exponential", "normal"]:
        if candidate in results:
            best_fit = candidate
            break

    return {
        "analysis_type": "distribution_fit",
        "n": n,
        "distributions": results,
        "best_fit": best_fit,
    }


def _shelf_life(times, values, lsl=None, usl=None, **kwargs):
    """Stability / shelf life analysis."""
    times = np.array(times, dtype=float)
    values = np.array(values, dtype=float)

    # Linear regression: value vs time
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(times, values)

    # Calculate shelf life (time when limit is reached)
    shelf_life = None
    if lsl is not None:
        shelf_life = (lsl - intercept) / slope if abs(slope) > 1e-12 else None
    elif usl is not None:
        shelf_life = (usl - intercept) / slope if abs(slope) > 1e-12 else None

    # 95% CI for shelf life
    ci_lower = None
    ci_upper = None
    if shelf_life is not None:
        # Approximate CI using Fieller's theorem
        n = len(times)
        t_crit = sp_stats.t.ppf(0.975, n - 2)
        x_mean = np.mean(times)
        se_pred = std_err * np.sqrt(1 + 1 / n + (shelf_life - x_mean) ** 2 / np.sum((times - x_mean) ** 2))
        ci_lower = shelf_life - t_crit * se_pred
        ci_upper = shelf_life + t_crit * se_pred

    return {
        "analysis_type": "stability",
        "n": len(times),
        "slope": r(slope),
        "intercept": r(intercept),
        "r_squared": r(r_value**2),
        "p_value": r(p_value),
        "lsl": lsl,
        "usl": usl,
        "shelf_life": r(shelf_life, 2) if shelf_life is not None else None,
        "shelf_life_ci_lower": r(ci_lower, 2) if ci_lower is not None else None,
        "shelf_life_ci_upper": r(ci_upper, 2) if ci_upper is not None else None,
        "interpretation": (
            f"Estimated shelf life = {r(shelf_life, 2)} time units"
            if shelf_life is not None
            else "Shelf life cannot be estimated (limit not reached)"
        ),
    }

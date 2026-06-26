"""Reliability and survival analysis: Weibull, Kaplan-Meier, distribution fitting, stability, ALT, Crow-AMSAA."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def reliability(analysis_type, **kwargs):
    """Perform reliability/survival analysis.

    Args:
        analysis_type: 'weibull', 'kaplan_meier', 'distribution', 'stability', 'alt', 'crow'

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
    elif analysis_type == "alt":
        return _alt(**kwargs)
    elif analysis_type == "crow":
        return _crow_amsaa(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _weibull(times, status=None, **kwargs):
    """Weibull analysis."""
    times = to_array(times, min_n=2, name="times")
    if np.any(times <= 0):
        raise ValueError("All times must be positive for Weibull analysis")
    n = len(times)

    if status is None:
        status = np.ones(n)
    status = np.array(status, dtype=float)
    if len(status) != n:
        raise ValueError(f"status length ({len(status)}) must match times length ({n})")

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
    times = to_array(times, min_n=2, name="times")
    if np.any(times < 0):
        raise ValueError("Times must be non-negative for Kaplan-Meier analysis")
    n = len(times)
    if status is None:
        status = np.ones(n)
    status = np.array(status, dtype=float)
    if len(status) != n:
        raise ValueError(f"status length ({len(status)}) must match times length ({n})")

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
    times = to_array(times, min_n=1, name="times")
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


def _alt(stress_levels, failure_times, stress_model="arrhenius", use_stress=None, **kwargs):
    """Accelerated Life Testing (ALT) analysis.

    Fits a life-stress relationship and extrapolates to use conditions.

    Args:
        stress_levels: List of stress values (e.g., temperatures in Kelvin)
        failure_times: List of median failure times at each stress level
        stress_model: 'arrhenius', 'inverse_power', or 'log_linear'
        use_stress: Normal use-condition stress value for extrapolation

    Returns:
        Dict with ALT results
    """
    stress_levels = to_array(stress_levels, min_n=2, name="stress_levels")
    failure_times = to_array(failure_times, min_n=2, name="failure_times")

    if len(stress_levels) != len(failure_times):
        raise ValueError("stress_levels and failure_times must have the same length")

    # Transform stress and failure times based on model
    if stress_model == "arrhenius":
        # ln(t) = a + b*(1/T)  where T is temperature in Kelvin
        x = 1.0 / stress_levels
        y = np.log(failure_times)
        model_name = "Arrhenius"
    elif stress_model == "inverse_power":
        # ln(t) = a + b*ln(S)
        if np.any(stress_levels <= 0):
            raise ValueError("Inverse Power Law requires positive stress levels")
        x = np.log(stress_levels)
        y = np.log(failure_times)
        model_name = "Inverse Power Law"
    elif stress_model == "log_linear":
        # ln(t) = a + b*S
        x = stress_levels
        y = np.log(failure_times)
        model_name = "Log-Linear"
    else:
        raise ValueError(f"Unknown stress_model: {stress_model}. Use arrhenius, inverse_power, or log_linear")

    # Linear regression on transformed data
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    a = intercept  # intercept
    b = slope  # slope

    result = {
        "analysis_type": "alt",
        "stress_model": stress_model,
        "model_name": model_name,
        "n_stress_levels": len(stress_levels),
        "model_params": {"a": r(a), "b": r(b)},
        "r_squared": r(r_value**2),
        "p_value": r(p_value),
        "std_err": r(std_err),
    }

    # Extrapolate to use stress if provided
    if use_stress is not None:
        if stress_model == "arrhenius":
            x_use = 1.0 / use_stress
        elif stress_model == "inverse_power":
            x_use = np.log(use_stress)
        else:  # log_linear
            x_use = use_stress

        ln_life_at_use = a + b * x_use
        median_life_at_use = np.exp(ln_life_at_use)
        result["use_stress"] = r(use_stress)
        result["median_life_at_use"] = r(median_life_at_use)

        # Acceleration factor = life at use / life at highest stress
        # Use the highest stress level as the accelerated condition
        highest_stress_idx = np.argmax(stress_levels)
        highest_stress = stress_levels[highest_stress_idx]
        if stress_model == "arrhenius":
            x_high = 1.0 / highest_stress
        elif stress_model == "inverse_power":
            x_high = np.log(highest_stress)
        else:
            x_high = highest_stress
        ln_life_high = a + b * x_high
        life_high = np.exp(ln_life_high)
        acceleration_factor = median_life_at_use / life_high
        result["acceleration_factor"] = r(acceleration_factor)
        result["highest_stress"] = r(highest_stress)
        result["life_at_highest_stress"] = r(life_high)

    result["interpretation"] = f"{model_name} model: ln(t) = {r(a, 4)} + {r(b, 4)}*x_transformed, R^2={r(r_value**2)}"

    return result


def _crow_amsaa(cumulative_times, cumulative_failures, **kwargs):
    """Crow-AMSAA (NHPP power-law) analysis for repairable systems.

    Model: lambda(t) = rho * beta * t^(beta-1)

    Args:
        cumulative_times: List of cumulative failure times
        cumulative_failures: Cumulative failure counts at each time

    Returns:
        Dict with Crow-AMSAA results
    """
    cumulative_times = to_array(cumulative_times, min_n=2, name="cumulative_times")
    cumulative_failures = to_array(cumulative_failures, min_n=2, name="cumulative_failures")

    if len(cumulative_times) != len(cumulative_failures):
        raise ValueError("cumulative_times and cumulative_failures must have the same length")
    if not np.all(np.diff(cumulative_times) > 0):
        raise ValueError("cumulative_times must be monotonically increasing")

    n = len(cumulative_times)
    T = cumulative_times[-1]  # total observation time
    N = cumulative_failures[-1]  # total failures

    if T <= 0 or N <= 0:
        raise ValueError("Total observation time and failure count must be positive")

    # MLE estimation
    # beta_hat = n / sum(ln(T / t_i)) for i = 1..n
    # For cumulative data, use the individual failure times
    # If cumulative_failures increments by 1 each time, these are exact failure times
    # Otherwise, use the provided cumulative data
    t_i = cumulative_times
    log_sum = np.sum(np.log(T / t_i[t_i > 0]))
    if log_sum <= 0:
        raise ValueError("Cannot estimate beta: log sum is non-positive")

    beta_hat = n / log_sum
    rho_hat = N / (T**beta_hat)

    # Current instantaneous MTBF at time T
    # lambda(T) = rho * beta * T^(beta-1)
    lambda_T = rho_hat * beta_hat * T ** (beta_hat - 1)
    current_mtbf = 1.0 / lambda_T if lambda_T > 0 else float("inf")

    # Growth rate: improvement if beta < 1
    growth_rate = 1 - beta_hat

    # Laplace test for goodness-of-fit (H0: NHPP vs H1: non-homogeneous)
    # U = (sum(t_i/T) - n/2) / (n / sqrt(12*n))
    # Simplified: use the test statistic based on uniformity of failure times
    if n >= 2:
        laplace_stat = (np.sum(t_i / T) - n / 2) / (n / np.sqrt(12 * n))
        gof_p_value = 2 * (1 - sp_stats.norm.cdf(abs(laplace_stat)))
    else:
        laplace_stat = None
        gof_p_value = None

    # Interpretation
    if beta_hat < 1:
        interp = "Improving system (beta < 1): reliability is increasing over time"
    elif beta_hat > 1:
        interp = "Deteriorating system (beta > 1): reliability is decreasing over time"
    else:
        interp = "Stable system (beta = 1): constant failure rate (HPP)"

    return {
        "analysis_type": "crow_amsaa",
        "n_failures": int(N),
        "observation_time": r(T),
        "beta": r(beta_hat),
        "rho": r(rho_hat),
        "current_mtbf": r(current_mtbf),
        "growth_rate": r(growth_rate),
        "laplace_statistic": r(laplace_stat) if laplace_stat is not None else None,
        "gof_p_value": r(gof_p_value) if gof_p_value is not None else None,
        "interpretation": interp,
    }

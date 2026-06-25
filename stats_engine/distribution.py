"""Distribution fitting, goodness-of-fit tests, and distribution selection."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array

# Supported distributions with their scipy stats objects and parameter names
DISTRIBUTIONS = {
    "normal": {"dist": sp_stats.norm, "param_names": ["loc", "scale"], "bounds": {}},
    "lognormal": {"dist": sp_stats.lognorm, "param_names": ["s", "loc", "scale"], "bounds": {"s": (0.001, None)}},
    "exponential": {"dist": sp_stats.expon, "param_names": ["loc", "scale"], "bounds": {}},
    "gamma": {"dist": sp_stats.gamma, "param_names": ["a", "loc", "scale"], "bounds": {"a": (0.001, None)}},
    "weibull": {"dist": sp_stats.weibull_min, "param_names": ["c", "loc", "scale"], "bounds": {"c": (0.001, None)}},
    "beta": {"dist": sp_stats.beta, "param_names": ["a", "b", "loc", "scale"], "bounds": {"a": (0.001, None), "b": (0.001, None)}},
    "logistic": {"dist": sp_stats.logistic, "param_names": ["loc", "scale"], "bounds": {}},
    "gumbel": {"dist": sp_stats.gumbel_r, "param_names": ["loc", "scale"], "bounds": {}},
}


def distribution(analysis_type, **kwargs):
    """Perform distribution analysis.

    Args:
        analysis_type: 'fit', 'gof', or 'select'

    Returns:
        Dict with distribution analysis results
    """
    if analysis_type == "fit":
        return _fit(**kwargs)
    elif analysis_type == "gof":
        return _gof(**kwargs)
    elif analysis_type == "select":
        return _select(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use: fit, gof, select")


def _fit(values, dist_name="normal", method="mle", **kwargs):
    """Fit a distribution to data using MLE or MOM.

    Args:
        values: Numeric data
        dist_name: Distribution name (normal, lognormal, exponential, gamma, weibull, beta, logistic, gumbel)
        method: Fitting method - 'mle' (maximum likelihood) or 'mom' (method of moments)

    Returns:
        Dict with fitted parameters, log-likelihood, AIC, BIC
    """
    arr = to_array(values, min_n=5, name="values")
    dist_name = dist_name.lower()

    if dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {dist_name}. Supported: {', '.join(DISTRIBUTIONS)}")

    method = method.lower()
    if method not in ("mle", "mom"):
        raise ValueError(f"Unknown method: {method}. Use 'mle' or 'mom'")

    entry = DISTRIBUTIONS[dist_name]
    dist = entry["dist"]
    param_names = entry["param_names"]

    if method == "mle":
        params = dist.fit(arr)
        log_lik = np.sum(dist.logpdf(arr, *params))
    else:
        # Method of moments: use dist.fit with floc=0/floc=None to fix location
        # For simplicity, fall back to MLE for complex distributions
        # and use analytical moments for simple ones
        if dist_name == "normal":
            mu = np.mean(arr)
            sigma = np.std(arr, ddof=1)
            params = (mu, sigma)
            log_lik = np.sum(dist.logpdf(arr, *params))
        elif dist_name == "exponential":
            scale = np.mean(arr)
            params = (0, scale)
            log_lik = np.sum(dist.logpdf(arr, *params))
        elif dist_name == "lognormal":
            log_arr = np.log(arr[arr > 0])
            mu = np.mean(log_arr)
            sigma = np.std(log_arr, ddof=1)
            params = (sigma, 0, np.exp(mu))
            log_lik = np.sum(dist.logpdf(arr, *params))
        else:
            # Fall back to MLE for complex distributions
            params = dist.fit(arr)
            log_lik = np.sum(dist.logpdf(arr, *params))

    n = len(arr)
    k = len(params)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    # Build parameter dict
    param_dict = {}
    for i, name in enumerate(param_names):
        param_dict[name] = r(params[i])

    # Compute goodness-of-fit statistics
    ks_stat, ks_p = sp_stats.kstest(arr, dist_name if dist_name in ("norm", "expon") else dist.cdf, args=params)

    return {
        "analysis_type": "fit",
        "distribution": dist_name,
        "method": method,
        "n": n,
        "parameters": param_dict,
        "log_likelihood": r(log_lik),
        "aic": r(aic),
        "bic": r(bic),
        "ks_statistic": r(ks_stat),
        "ks_p_value": r(ks_p),
        "mean": r(dist.mean(*params)),
        "variance": r(dist.var(*params)),
        "interpretation": f"{dist_name} fitted via {method.upper()}: log-likelihood={r(log_lik, 4)}, AIC={r(aic, 4)}, BIC={r(bic, 4)}",
    }


def _gof(values, dist_name="normal", **kwargs):
    """Run goodness-of-fit tests for a distribution.

    Args:
        values: Numeric data
        dist_name: Distribution to test against

    Returns:
        Dict with KS, Anderson-Darling, and chi-square test results
    """
    arr = to_array(values, min_n=5, name="values")
    dist_name = dist_name.lower()

    if dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {dist_name}. Supported: {', '.join(DISTRIBUTIONS)}")

    entry = DISTRIBUTIONS[dist_name]
    dist = entry["dist"]

    # Fit the distribution first
    params = dist.fit(arr)
    n = len(arr)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = sp_stats.kstest(arr, dist.cdf, args=params)

    # Anderson-Darling test (for normal distribution)
    ad_result = None
    if dist_name == "normal":
        ad_test = sp_stats.anderson(arr, dist="norm")
        ad_result = {
            "statistic": r(ad_test.statistic),
            "critical_values": {f"{sl}%": r(cv) for sl, cv in zip(ad_test.significance_level, ad_test.critical_values)},
        }

    # Chi-square test (binned)
    n_bins = max(5, int(np.sqrt(n)))
    observed, bin_edges = np.histogram(arr, bins=n_bins)
    expected = np.zeros(n_bins)
    for i in range(n_bins):
        expected[i] = n * (dist.cdf(bin_edges[i + 1], *params) - dist.cdf(bin_edges[i], *params))

    # Merge bins with expected < 5
    while np.any(expected < 5) and len(expected) > 3:
        # Find the bin with smallest expected
        min_idx = np.argmin(expected)
        if min_idx == 0:
            merge_with = 1
        elif min_idx == len(expected) - 1:
            merge_with = min_idx - 1
        else:
            merge_with = min_idx - 1 if expected[min_idx - 1] < expected[min_idx + 1] else min_idx + 1

        lo = min(min_idx, merge_with)
        hi = max(min_idx, merge_with)
        observed = np.concatenate([observed[:lo], [observed[lo] + observed[hi]], observed[hi + 1:]])
        expected = np.concatenate([expected[:lo], [expected[lo] + expected[hi]], expected[hi + 1:]])

    if len(expected) > 1 and np.all(expected > 0):
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        chi2_df = len(expected) - 1 - len(params)
        chi2_p = 1 - sp_stats.chi2.cdf(chi2_stat, max(chi2_df, 1))
    else:
        chi2_stat = None
        chi2_p = None

    result = {
        "analysis_type": "gof",
        "distribution": dist_name,
        "n": n,
        "parameters": {name: r(params[i]) for i, name in enumerate(entry["param_names"])},
        "ks_statistic": r(ks_stat),
        "ks_p_value": r(ks_p),
        "chi2_statistic": r(chi2_stat) if chi2_stat is not None else None,
        "chi2_p_value": r(chi2_p) if chi2_p is not None else None,
        "n_bins": len(expected),
    }

    if ad_result:
        result["anderson_darling"] = ad_result

    # Interpretation
    if ks_p > 0.05:
        result["interpretation"] = f"KS test p={r(ks_p, 4)} > 0.05: data is consistent with {dist_name} distribution"
    else:
        result["interpretation"] = f"KS test p={r(ks_p, 4)} <= 0.05: data may not follow {dist_name} distribution"

    return result


def _select(values, distributions=None, criterion="aic", **kwargs):
    """Compare multiple distributions and rank by information criterion.

    Args:
        values: Numeric data
        distributions: List of distribution names to compare (default: all supported)
        criterion: Ranking criterion - 'aic' or 'bic'

    Returns:
        Dict with ranked distributions and their fit statistics
    """
    arr = to_array(values, min_n=5, name="values")
    criterion = criterion.lower()
    if criterion not in ("aic", "bic"):
        raise ValueError(f"Unknown criterion: {criterion}. Use 'aic' or 'bic'")

    if distributions is None:
        dist_list = list(DISTRIBUTIONS.keys())
    else:
        dist_list = [d.lower() for d in distributions]
        for d in dist_list:
            if d not in DISTRIBUTIONS:
                raise ValueError(f"Unknown distribution: {d}. Supported: {', '.join(DISTRIBUTIONS)}")

    n = len(arr)
    results = []

    for dist_name in dist_list:
        entry = DISTRIBUTIONS[dist_name]
        dist = entry["dist"]
        param_names = entry["param_names"]

        try:
            params = dist.fit(arr)
            log_lik = np.sum(dist.logpdf(arr, *params))
            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik

            # KS test
            ks_stat, ks_p = sp_stats.kstest(arr, dist.cdf, args=params)

            results.append({
                "distribution": dist_name,
                "parameters": {name: r(params[i]) for i, name in enumerate(param_names)},
                "log_likelihood": r(log_lik),
                "aic": r(aic),
                "bic": r(bic),
                "ks_statistic": r(ks_stat),
                "ks_p_value": r(ks_p),
            })
        except Exception as e:
            results.append({
                "distribution": dist_name,
                "error": str(e),
            })

    # Sort by criterion (lower is better), put errors at end
    valid = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]
    valid.sort(key=lambda x: x[criterion])
    results = valid + errored

    # Add rank
    for i, r_dict in enumerate(valid):
        r_dict["rank"] = i + 1

    best = results[0] if results and "error" not in results[0] else None

    return {
        "analysis_type": "select",
        "n": n,
        "criterion": criterion,
        "n_distributions": len(dist_list),
        "rankings": results,
        "best_distribution": best["distribution"] if best else None,
        "best_parameters": best["parameters"] if best else None,
        "interpretation": (
            f"Best fit: {best['distribution']} ({criterion.upper()}={best[criterion]})"
            if best
            else "No distribution could be fitted"
        ),
    }

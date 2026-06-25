"""Bayesian statistics: estimation, hypothesis testing, and credible intervals."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r
from utils.validators import to_array


def bayesian(analysis_type, **kwargs):
    """Perform Bayesian analysis.

    Args:
        analysis_type: 'estimate', 'ttest', 'proportion', 'anova'

    Returns:
        Dict with Bayesian analysis results
    """
    if analysis_type == "estimate":
        return _estimate(**kwargs)
    elif analysis_type == "ttest":
        return _bayesian_ttest(**kwargs)
    elif analysis_type == "proportion":
        return _bayesian_proportion(**kwargs)
    elif analysis_type == "anova":
        return _bayesian_anova(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use: estimate, ttest, proportion, anova")


def _estimate(values, prior_mean=None, prior_std=None, credible_level=0.95, **kwargs):
    """Bayesian estimation of a normal mean with conjugate normal prior.

    Uses Normal-Normal conjugate model:
        Prior: mu ~ N(prior_mean, prior_std^2)
        Likelihood: x_i ~ N(mu, sigma^2) with sigma estimated from data
        Posterior: mu | data ~ N(posterior_mean, posterior_std^2)

    Args:
        values: Observed data
        prior_mean: Prior mean (default: sample mean)
        prior_std: Prior standard deviation (default: 10 * sample std, weakly informative)
        credible_level: Credible interval level (default 0.95)

    Returns:
        Dict with posterior parameters and credible interval
    """
    arr = to_array(values, min_n=2, name="values")
    n = len(arr)
    sample_mean = np.mean(arr)
    sample_std = np.std(arr, ddof=1)

    # Default prior: weakly informative
    if prior_mean is None:
        prior_mean = sample_mean
    if prior_std is None:
        prior_std = 10 * sample_std

    # Known variance = sample variance (plug-in)
    sigma2 = sample_std ** 2

    # Posterior parameters (Normal-Normal conjugate)
    prior_precision = 1.0 / prior_std ** 2
    data_precision = n / sigma2
    posterior_precision = prior_precision + data_precision
    posterior_var = 1.0 / posterior_precision
    posterior_std = np.sqrt(posterior_var)
    posterior_mean = (prior_precision * prior_mean + data_precision * sample_mean) / posterior_precision

    # Credible interval
    alpha = 1 - credible_level
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = posterior_mean - z_crit * posterior_std
    ci_upper = posterior_mean + z_crit * posterior_std

    # Prior vs posterior comparison
    prior_ci_lower = prior_mean - z_crit * prior_std
    prior_ci_upper = prior_mean + z_crit * prior_std

    # Bayes factor for H0: mu = 0 vs H1: mu != 0 (BIC approximation)
    log_lik_null = np.sum(sp_stats.norm.logpdf(arr, loc=0, scale=sample_std))
    log_lik_alt = np.sum(sp_stats.norm.logpdf(arr, loc=sample_mean, scale=sample_std))
    bic_null = -2 * log_lik_null + 0 * np.log(n)  # 0 free params under null
    bic_alt = -2 * log_lik_alt + 1 * np.log(n)  # 1 free param (mean)
    log_bf = (bic_null - bic_alt) / 2
    bf_10 = np.exp(log_bf) if abs(log_bf) < 700 else (float("inf") if log_bf > 0 else 0.0)

    return {
        "analysis_type": "estimate",
        "n": n,
        "sample_mean": r(sample_mean),
        "sample_std": r(sample_std),
        "prior_mean": r(prior_mean),
        "prior_std": r(prior_std),
        "posterior_mean": r(posterior_mean),
        "posterior_std": r(posterior_std),
        "credible_level": credible_level,
        "credible_interval": [r(ci_lower), r(ci_upper)],
        "prior_ci": [r(prior_ci_lower), r(prior_ci_upper)],
        "bayes_factor_h0_vs_h1": r(bf_10),
        "interpretation": (
            f"Posterior mean = {r(posterior_mean)} "
            f"({credible_level*100:.0f}% CI: [{r(ci_lower)}, {r(ci_upper)}]). "
            f"BF01 = {r(bf_10)}: {'evidence for H0: mu=0' if bf_10 > 1 else 'evidence against H0: mu=0'}"
        ),
    }


def _bayesian_ttest(values, values2=None, mu=0, paired=False, credible_level=0.95, **kwargs):
    """Bayesian t-test with Bayes Factor.

    Computes Bayes Factor using the JZS (Jeffreys-Zellner-Siow) prior approximation.
    BF10 > 1 favors H1 (difference exists), BF10 < 1 favors H0 (no difference).

    Args:
        values: First sample
        values2: Second sample (None for one-sample test)
        mu: Null hypothesis mean / mean difference (default 0)
        paired: Whether samples are paired
        credible_level: Credible interval level (default 0.95)

    Returns:
        Dict with Bayes Factor and posterior estimates
    """
    arr1 = to_array(values, min_n=2, name="values")

    if values2 is not None:
        arr2 = to_array(values2, min_n=2, name="values2")
        if paired:
            if len(arr1) != len(arr2):
                raise ValueError("Paired test requires equal-length samples")
            diff = arr1 - arr2
            n = len(diff)
            sample_mean = np.mean(diff)
            sample_std = np.std(diff, ddof=1)
            test_label = "paired"
        else:
            n1, n2 = len(arr1), len(arr2)
            sample_mean = np.mean(arr1) - np.mean(arr2)
            sp = np.sqrt(((n1 - 1) * np.var(arr1, ddof=1) + (n2 - 1) * np.var(arr2, ddof=1)) / (n1 + n2 - 2))
            sample_std = sp * np.sqrt(1 / n1 + 1 / n2)
            n = n1 + n2 - 2  # effective df
            test_label = "two_sample"
    else:
        n = len(arr1)
        sample_mean = np.mean(arr1)
        sample_std = np.std(arr1, ddof=1) / np.sqrt(n)
        test_label = "one_sample"

    # t-statistic and classical p-value
    t_stat = (sample_mean - mu) / sample_std if sample_std > 0 else 0
    df = n - 1 if values2 is None or paired else n
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=df))

    # Bayes Factor using BIC approximation (Wagenmakers, 2007)
    # BF10 ≈ exp((BIC_null - BIC_alt) / 2)
    if values2 is not None and not paired:
        # Two-sample: pooled data under H0, separate means under H1
        all_data = np.concatenate([arr1, arr2])
        pooled_mean = np.mean(all_data)
        residuals_null = all_data - pooled_mean
        residuals_alt = np.concatenate([arr1 - np.mean(arr1), arr2 - np.mean(arr2)])
        sigma_null = np.std(residuals_null, ddof=1)
        sigma_alt = np.std(residuals_alt, ddof=1)
        log_lik_null = np.sum(sp_stats.norm.logpdf(residuals_null, loc=0, scale=sigma_null))
        log_lik_alt = np.sum(sp_stats.norm.logpdf(residuals_alt, loc=0, scale=sigma_alt))
        k_null = 1  # pooled mean
        k_alt = 2   # two group means
    else:
        if paired:
            data = diff
        else:
            data = arr1
        residuals_null = data - mu
        residuals_alt = data - np.mean(data)
        sigma_null = np.std(residuals_null, ddof=1)
        sigma_alt = np.std(residuals_alt, ddof=1)
        log_lik_null = np.sum(sp_stats.norm.logpdf(residuals_null, loc=0, scale=sigma_null))
        log_lik_alt = np.sum(sp_stats.norm.logpdf(residuals_alt, loc=0, scale=sigma_alt))
        k_null = 1  # sigma only (mean fixed at mu)
        k_alt = 2   # mean + sigma

    total_n = len(arr1) + (len(arr2) if values2 is not None else 0)
    bic_null = -2 * log_lik_null + k_null * np.log(total_n)
    bic_alt = -2 * log_lik_alt + k_alt * np.log(total_n)
    log_bf = (bic_null - bic_alt) / 2
    bf_10 = np.exp(log_bf) if abs(log_bf) < 700 else (float("inf") if log_bf > 0 else 0.0)

    # Posterior credible interval for the mean difference
    alpha = 1 - credible_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
    ci_lower = sample_mean - t_crit * sample_std
    ci_upper = sample_mean + t_crit * sample_std

    # Effect size (Cohen's d)
    if values2 is not None:
        if paired:
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            pooled_std = np.sqrt(((n1 - 1) * np.var(arr1, ddof=1) + (n2 - 1) * np.var(arr2, ddof=1)) / (n1 + n2 - 2))
            cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std
    else:
        cohens_d = (np.mean(arr1) - mu) / np.std(arr1, ddof=1)

    # Interpretation of Bayes Factor
    if bf_10 > 100:
        bf_label = "extreme evidence for H1"
    elif bf_10 > 30:
        bf_label = "very strong evidence for H1"
    elif bf_10 > 10:
        bf_label = "strong evidence for H1"
    elif bf_10 > 3:
        bf_label = "moderate evidence for H1"
    elif bf_10 > 1:
        bf_label = "anecdotal evidence for H1"
    elif bf_10 > 1 / 3:
        bf_label = "anecdotal evidence for H0"
    elif bf_10 > 1 / 10:
        bf_label = "moderate evidence for H0"
    elif bf_10 > 1 / 30:
        bf_label = "strong evidence for H0"
    elif bf_10 > 1 / 100:
        bf_label = "very strong evidence for H0"
    else:
        bf_label = "extreme evidence for H0"

    result = {
        "analysis_type": "bayesian_ttest",
        "test_type": test_label,
        "n": total_n,
        "mean_difference": r(sample_mean),
        "std_error": r(sample_std),
        "t_statistic": r(t_stat),
        "df": r(float(df), 1),
        "p_value": r(p_value),
        "cohens_d": r(cohens_d),
        "bayes_factor_10": r(bf_10),
        "bayes_factor_01": r(1 / bf_10) if bf_10 > 0 else None,
        "bf_interpretation": bf_label,
        "credible_level": credible_level,
        "credible_interval": [r(ci_lower), r(ci_upper)],
        "interpretation": f"BF10 = {r(bf_10)} ({bf_label}). {credible_level*100:.0f}% CI: [{r(ci_lower)}, {r(ci_upper)}]",
    }

    return result


def _bayesian_proportion(successes, n, prior_alpha=1, prior_beta=1, credible_level=0.95, **kwargs):
    """Bayesian proportion analysis using Beta-Binomial conjugate model.

    Prior: p ~ Beta(prior_alpha, prior_beta)
    Likelihood: X ~ Binomial(n, p)
    Posterior: p | X ~ Beta(prior_alpha + successes, prior_beta + n - successes)

    Args:
        successes: Number of successes
        n: Total number of trials
        prior_alpha: Beta prior alpha parameter (default 1 = uniform)
        prior_beta: Beta prior beta parameter (default 1 = uniform)
        credible_level: Credible interval level (default 0.95)

    Returns:
        Dict with posterior parameters and credible interval
    """
    successes = int(successes)
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be between 0 and n, got {successes}")

    # Posterior parameters
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + n - successes

    # Posterior statistics
    post_mean = post_alpha / (post_alpha + post_beta)
    post_var = (post_alpha * post_beta) / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
    post_std = np.sqrt(post_var)

    # Credible interval
    alpha = 1 - credible_level
    ci_lower = sp_stats.beta.ppf(alpha / 2, post_alpha, post_beta)
    ci_upper = sp_stats.beta.ppf(1 - alpha / 2, post_alpha, post_beta)

    # MAP estimate
    if post_alpha > 1 and post_beta > 1:
        map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        map_estimate = post_mean

    # Prior mean
    prior_mean = prior_alpha / (prior_alpha + prior_beta)

    # Bayes Factor for H0: p = p0 vs H1: p != p0 (using p0 = 0.5 as default)
    p0 = 0.5
    # Savage-Dickey density ratio approximation
    prior_at_p0 = sp_stats.beta.pdf(p0, prior_alpha, prior_beta)
    post_at_p0 = sp_stats.beta.pdf(p0, post_alpha, post_beta)
    bf_01 = post_at_p0 / prior_at_p0 if prior_at_p0 > 0 else None

    return {
        "analysis_type": "bayesian_proportion",
        "n": n,
        "successes": successes,
        "sample_proportion": r(successes / n),
        "prior_alpha": r(prior_alpha),
        "prior_beta": r(prior_beta),
        "prior_mean": r(prior_mean),
        "posterior_alpha": r(post_alpha),
        "posterior_beta": r(post_beta),
        "posterior_mean": r(post_mean),
        "posterior_std": r(post_std),
        "map_estimate": r(map_estimate),
        "credible_level": credible_level,
        "credible_interval": [r(ci_lower), r(ci_upper)],
        "bayes_factor_01_p0_05": r(bf_01) if bf_01 is not None else None,
        "interpretation": (
            f"Posterior: Beta({r(post_alpha)}, {r(post_beta)}). "
            f"Mean = {r(post_mean)}, MAP = {r(map_estimate)}, "
            f"{credible_level*100:.0f}% CI: [{r(ci_lower)}, {r(ci_upper)}]"
        ),
    }


def _bayesian_anova(groups, credible_level=0.95, **kwargs):
    """Bayesian one-way ANOVA using Bayes Factor approximation.

    Computes BF10 for the hypothesis that group means differ,
    using the BIC approximation (Wagenmakers, 2007).

    Args:
        groups: List of groups, each group is an array of values
        credible_level: Credible interval level (default 0.95)

    Returns:
        Dict with Bayes Factor and group-level posterior estimates
    """
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("Need at least 2 groups")

    # Convert and validate groups
    clean_groups = []
    for i, g in enumerate(groups):
        arr = np.array(g, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            raise ValueError(f"Group {i} must have at least 2 valid values")
        clean_groups.append(arr)

    k = len(clean_groups)
    all_data = np.concatenate(clean_groups)
    grand_mean = np.mean(all_data)
    n_total = len(all_data)

    # Classical ANOVA for reference
    f_stat, p_value = sp_stats.f_oneway(*clean_groups)

    # BIC approximation for Bayes Factor
    # H0: all group means equal (grand mean only)
    # H1: each group has its own mean
    residuals_null = all_data - grand_mean
    sse_null = np.sum(residuals_null ** 2)

    sse_alt = 0
    for g in clean_groups:
        sse_alt += np.sum((g - np.mean(g)) ** 2)

    # Log-likelihoods
    sigma_null = np.sqrt(sse_null / n_total)
    sigma_alt = np.sqrt(sse_alt / n_total)

    log_lik_null = np.sum(sp_stats.norm.logpdf(residuals_null, loc=0, scale=sigma_null)) if sigma_null > 0 else -1e10
    log_lik_alt = 0
    for g in clean_groups:
        log_lik_alt += np.sum(sp_stats.norm.logpdf(g, loc=np.mean(g), scale=sigma_alt)) if sigma_alt > 0 else -1e10

    bic_null = -2 * log_lik_null + 1 * np.log(n_total)  # 1 param: grand mean
    bic_alt = -2 * log_lik_alt + k * np.log(n_total)     # k params: group means

    bf_10 = np.exp((bic_null - bic_alt) / 2)

    # Posterior estimates for each group (conjugate normal)
    alpha = 1 - credible_level
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    group_posteriors = []
    for i, g in enumerate(clean_groups):
        n_i = len(g)
        mean_i = np.mean(g)
        std_i = np.std(g, ddof=1)
        se_i = std_i / np.sqrt(n_i)

        group_posteriors.append({
            "group": i,
            "n": n_i,
            "sample_mean": r(mean_i),
            "sample_std": r(std_i),
            "posterior_mean": r(mean_i),
            "posterior_se": r(se_i),
            "credible_interval": [r(mean_i - z_crit * se_i), r(mean_i + z_crit * se_i)],
        })

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean_groups)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Interpretation
    if bf_10 > 10:
        bf_label = "strong evidence for group differences"
    elif bf_10 > 3:
        bf_label = "moderate evidence for group differences"
    elif bf_10 > 1:
        bf_label = "anecdotal evidence for group differences"
    elif bf_10 > 1 / 3:
        bf_label = "anecdotal evidence for no group differences"
    elif bf_10 > 1 / 10:
        bf_label = "moderate evidence for no group differences"
    else:
        bf_label = "strong evidence for no group differences"

    return {
        "analysis_type": "bayesian_anova",
        "n_groups": k,
        "n_total": n_total,
        "f_statistic": r(f_stat),
        "p_value": r(p_value),
        "eta_squared": r(eta_squared),
        "bayes_factor_10": r(bf_10),
        "bayes_factor_01": r(1 / bf_10) if bf_10 > 0 else None,
        "bf_interpretation": bf_label,
        "credible_level": credible_level,
        "group_posteriors": group_posteriors,
        "interpretation": f"BF10 = {r(bf_10)} ({bf_label}). Eta-squared = {r(eta_squared)}",
    }

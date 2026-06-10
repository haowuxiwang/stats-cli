"""Power analysis and sample size calculation."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def power(analysis_type, effect_size=None, alpha=0.05, power=None, n=None, power_val=None, k_groups=3, df=1):
    """Calculate power or required sample size.

    Args:
        analysis_type: 't_test', 'anova', 'chi_square', 'proportion', 'correlation'
        effect_size: Effect size (Cohen's d, f, w, h, or r)
        alpha: Significance level
        power: Desired power (1 - beta) - primary parameter name
        n: Sample size
        power_val: Alias for power (backward compatibility)
        k_groups: Number of groups for ANOVA (default 3)
        df: Degrees of freedom for chi-square (default 1)

    Returns:
        Dict with power analysis results
    """
    # Accept both 'power' (user-facing) and 'power_val' (legacy)
    if power is not None and power_val is None:
        power_val = power

    if effect_size is not None and effect_size < 0:
        raise ValueError(f"effect_size must be non-negative, got {effect_size}")
    if effect_size == 0:
        return {
            "analysis_type": analysis_type,
            "effect_size": 0,
            "alpha": alpha,
            "power": alpha,
            "n": n,
            "warning": "effect_size=0 means no effect; power equals alpha level",
        }

    if analysis_type == "t_test":
        return _t_test_power(effect_size, alpha, power_val, n)
    elif analysis_type == "anova":
        return _anova_power(effect_size, alpha, power_val, n, k_groups)
    elif analysis_type == "chi_square":
        return _chi_square_power(effect_size, alpha, power_val, n, df)
    elif analysis_type == "proportion":
        return _proportion_power(effect_size, alpha, power_val, n)
    elif analysis_type == "correlation":
        return _correlation_power(effect_size, alpha, power_val, n)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _t_test_power(effect_size, alpha, power_val, n):
    """Power analysis for t-test."""
    if power_val is None and n is not None:
        # Calculate power
        df = n - 1
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
        ncp = effect_size * np.sqrt(n)
        power_calc = 1 - sp_stats.nct.cdf(t_crit, df, ncp) + sp_stats.nct.cdf(-t_crit, df, ncp)
        return {
            "analysis_type": "t_test",
            "effect_size": effect_size,
            "alpha": alpha,
            "n": n,
            "power": r(power_calc),
            "interpretation": f"Power = {r(power_calc)} with n={n}",
        }
    elif n is None and power_val is not None:
        # Calculate required sample size
        for test_n in range(2, 10000):
            df = test_n - 1
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
            ncp = effect_size * np.sqrt(test_n)
            pwr = 1 - sp_stats.nct.cdf(t_crit, df, ncp) + sp_stats.nct.cdf(-t_crit, df, ncp)
            if pwr >= power_val:
                return {
                    "analysis_type": "t_test",
                    "effect_size": effect_size,
                    "alpha": alpha,
                    "power": power_val,
                    "sample_size": test_n,
                    "interpretation": f"Need n={test_n} to achieve power={power_val}",
                }
        return {"error": True, "message": "Could not find required sample size (tried up to 10000)"}
    else:
        raise ValueError("Provide either 'n' (to calculate power) or 'power_val' (to calculate sample size)")


def _anova_power(effect_size, alpha, power_val, n, k=3):
    """Power analysis for ANOVA."""
    if power_val is None and n is not None:
        # Approximate power using non-central F
        df1 = k - 1
        df2 = n - k
        f_crit = sp_stats.f.ppf(1 - alpha, df1, df2)
        ncp = effect_size**2 * n
        power_calc = 1 - sp_stats.ncf.cdf(f_crit, df1, df2, ncp)
        return {
            "analysis_type": "anova",
            "effect_size": effect_size,
            "alpha": alpha,
            "n": n,
            "n_groups": k,
            "power": r(power_calc),
        }
    elif n is None and power_val is not None:
        for test_n in range(k + 1, 10000):
            df1 = k - 1
            df2 = test_n - k
            f_crit = sp_stats.f.ppf(1 - alpha, df1, df2)
            ncp = effect_size**2 * test_n
            pwr = 1 - sp_stats.ncf.cdf(f_crit, df1, df2, ncp)
            if pwr >= power_val:
                return {
                    "analysis_type": "anova",
                    "effect_size": effect_size,
                    "alpha": alpha,
                    "power": power_val,
                    "n_groups": k,
                    "sample_size": test_n,
                    "interpretation": f"Need n={test_n} ({k} groups) to achieve power={power_val}",
                }
        return {"error": True, "message": "Could not find required sample size"}
    else:
        raise ValueError("Provide either 'n' or 'power_val'")


def _chi_square_power(effect_size, alpha, power_val, n, df=1):
    """Power analysis for chi-square test."""
    if power_val is None and n is not None:
        chi2_crit = sp_stats.chi2.ppf(1 - alpha, df)
        ncp = effect_size**2 * n
        power_calc = 1 - sp_stats.ncx2.cdf(chi2_crit, df, ncp)
        return {
            "analysis_type": "chi_square",
            "effect_size": effect_size,
            "alpha": alpha,
            "n": n,
            "power": r(power_calc),
        }
    elif n is None and power_val is not None:
        for test_n in range(2, 10000):
            chi2_crit = sp_stats.chi2.ppf(1 - alpha, df)
            ncp = effect_size**2 * test_n
            pwr = 1 - sp_stats.ncx2.cdf(chi2_crit, df, ncp)
            if pwr >= power_val:
                return {
                    "analysis_type": "chi_square",
                    "effect_size": effect_size,
                    "alpha": alpha,
                    "power": power_val,
                    "sample_size": test_n,
                }
        return {"error": True, "message": "Could not find required sample size"}
    else:
        raise ValueError("Provide either 'n' or 'power_val'")


def _proportion_power(effect_size, alpha, power_val, n):
    """Power analysis for proportion test."""
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

    if power_val is None and n is not None:
        z_beta = effect_size * np.sqrt(n) - z_alpha
        power_calc = sp_stats.norm.cdf(z_beta)
        return {
            "analysis_type": "proportion",
            "effect_size": effect_size,
            "alpha": alpha,
            "n": n,
            "power": r(power_calc),
        }
    elif n is None and power_val is not None:
        z_beta = sp_stats.norm.ppf(power_val)
        n_calc = ((z_alpha + z_beta) / effect_size) ** 2
        return {
            "analysis_type": "proportion",
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power_val,
            "sample_size": int(np.ceil(n_calc)),
            "interpretation": f"Need n={int(np.ceil(n_calc))} to achieve power={power_val}",
        }
    else:
        raise ValueError("Provide either 'n' or 'power_val'")


def _correlation_power(effect_size, alpha, power_val, n):
    """Power analysis for correlation."""
    if power_val is None and n is not None:
        # Fisher's z transformation
        z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        se = 1 / np.sqrt(n - 3)
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        power_calc = 1 - sp_stats.norm.cdf(z_crit - z_r / se) + sp_stats.norm.cdf(-z_crit - z_r / se)
        return {
            "analysis_type": "correlation",
            "effect_size": effect_size,
            "alpha": alpha,
            "n": n,
            "power": r(power_calc),
        }
    elif n is None and power_val is not None:
        z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        se = 1 / np.sqrt(100 - 3)  # start with n=100
        z_beta = sp_stats.norm.ppf(power_val)
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        n_calc = ((z_crit + z_beta) / z_r) ** 2 + 3
        return {
            "analysis_type": "correlation",
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power_val,
            "sample_size": int(np.ceil(n_calc)),
        }
    else:
        raise ValueError("Provide either 'n' or 'power_val'")

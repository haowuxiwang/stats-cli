"""Regression analysis: linear, multiple, polynomial, stepwise, logistic, nonlinear."""

import numpy as np
from scipy import stats as sp_stats

from utils.output import r


def regression(
    x=None,
    y=None,
    reg_type="linear",
    degree=2,
    file=None,
    x_columns=None,
    y_column=None,
    alpha=0.05,
    reg_alpha=None,
    l1_ratio=0.5,
    method="huber",
):
    """Perform regression analysis.

    Args:
        x: List of x values (simple regression) or 2D array
        y: List of y values
        reg_type: 'linear', 'quadratic', 'polynomial', 'multiple', 'stepwise', 'logistic',
                 'exponential', 'power', 'logarithmic', 'sigmoid',
                 'lasso', 'ridge', 'elastic_net', 'robust'
        degree: Polynomial degree (for polynomial type)
        file: Data file path (for multiple/stepwise)
        x_columns: Column names for x (for multiple/stepwise)
        y_column: Column name for y (for multiple/stepwise)
        alpha: Significance level
        reg_alpha: Regularization strength for penalized regression (lasso/ridge/elastic_net).
                   None = automatic cross-validated selection.
        l1_ratio: Elastic net mixing parameter (0=ridge, 1=lasso, default 0.5)
        method: M-estimator for robust regression ('huber' or 'tukey')

    Returns:
        Dict with regression results
    """
    _scalar_types = ("linear", "quadratic", "polynomial", "logistic", "exponential", "power", "logarithmic", "sigmoid")
    _penalized_types = ("lasso", "ridge", "elastic_net", "robust")

    # Filter NaN/Inf from x and y together for array-based regression types
    if reg_type in _scalar_types or reg_type in _penalized_types:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if x_arr.ndim == 2:
            x_finite = np.all(np.isfinite(x_arr), axis=1)
            y_finite = np.isfinite(y_arr)
            mask = x_finite & y_finite
            if mask.sum() < len(y_arr):
                x_arr = x_arr[mask]
                y_arr = y_arr[mask]
            if len(y_arr) < 2:
                raise ValueError(
                    f"After removing NaN/Inf, only {len(y_arr)} valid data points remain (need at least 2)"
                )
        else:
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask.sum() < len(x_arr):
                x_arr = x_arr[mask]
                y_arr = y_arr[mask]
            if len(x_arr) < 2:
                raise ValueError(
                    f"After removing NaN/Inf, only {len(x_arr)} valid data points remain (need at least 2)"
                )
    else:
        x_arr = np.array(x, dtype=float) if x is not None else None
        y_arr = np.array(y, dtype=float) if y is not None else None

    if reg_type == "linear":
        return _simple_linear(x_arr, y_arr, alpha)
    elif reg_type == "quadratic":
        return _polynomial(x_arr, y_arr, 2, alpha)
    elif reg_type == "polynomial":
        return _polynomial(x_arr, y_arr, degree, alpha)
    elif reg_type in ("multiple", "stepwise"):
        if file is None:
            raise ValueError("file is required for multiple/stepwise regression")
        return _multiple_regression(file, x_columns, y_column, reg_type, alpha)
    elif reg_type == "logistic":
        return _logistic(x_arr, y_arr, alpha)
    elif reg_type in ("exponential", "power", "logarithmic", "sigmoid"):
        return _nonlinear(x_arr, y_arr, reg_type, alpha)
    elif reg_type == "lasso":
        return _lasso(x_arr, y_arr, reg_alpha)
    elif reg_type == "ridge":
        return _ridge(x_arr, y_arr, reg_alpha)
    elif reg_type == "elastic_net":
        return _elastic_net(x_arr, y_arr, reg_alpha, l1_ratio)
    elif reg_type == "robust":
        return _robust(x_arr, y_arr, method)
    else:
        raise ValueError(f"Unknown regression type: {reg_type}")


def _simple_linear(x, y, alpha):
    """Simple linear regression."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    if n < 2:
        raise ValueError("At least 2 data points are required for linear regression")

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

    if n < 3:
        # n=2: perfect fit but adj_r_squared, f_statistic, p_value, CIs are undefined
        return {
            "regression_type": "linear",
            "n": n,
            "coefficients": {"intercept": r(intercept), "slope": r(slope)},
            "std_errors": {"intercept": None, "slope": None},
            "ci_intercept": [None, None],
            "ci_slope": [None, None],
            "r_squared": 1.0,
            "r": r(r_value),
            "adj_r_squared": None,
            "f_statistic": None,
            "p_value": None,
            "significant": None,
            "residual_std": None,
            "equation": f"y = {r(slope)} * x + {r(intercept)}",
            "_warning": "n=2: perfect fit but inferential statistics are undefined",
        }
    r_sq = r_value**2

    # Predictions
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # ANOVA for regression
    ss_reg = np.sum((y_pred - np.mean(y)) ** 2)
    ss_res = np.sum(residuals**2)
    ms_reg = ss_reg / 1
    ms_res = ss_res / (n - 2)
    f_stat = ms_reg / ms_res if ms_res > 0 else float("inf")
    p_f = 1 - sp_stats.f.cdf(f_stat, 1, n - 2)

    # Confidence intervals for coefficients
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 2)
    se_slope = std_err
    se_intercept = np.sqrt(ms_res * (1 / n + np.mean(x) ** 2 / np.sum((x - np.mean(x)) ** 2)))

    return {
        "regression_type": "linear",
        "n": n,
        "coefficients": {
            "intercept": r(intercept),
            "slope": r(slope),
        },
        "std_errors": {
            "intercept": r(se_intercept),
            "slope": r(se_slope),
        },
        "ci_intercept": [
            r(intercept - t_crit * se_intercept),
            r(intercept + t_crit * se_intercept),
        ],
        "ci_slope": [
            r(slope - t_crit * se_slope),
            r(slope + t_crit * se_slope),
        ],
        "r_squared": r(r_sq),
        "r": r(r_value),
        "adj_r_squared": r(1 - (1 - r_sq) * (n - 1) / (n - 2)),
        "f_statistic": r(f_stat),
        "p_value": r(p_f),
        "significant": bool(p_f < alpha),
        "residual_std": r(np.sqrt(ms_res)),
        "equation": f"y = {r(slope)} * x + {r(intercept)}",
    }


def _polynomial(x, y, degree, alpha):
    """Polynomial regression."""
    if len(x) < 2:
        raise ValueError("At least 2 data points are required for polynomial regression")
    if len(x) <= degree:
        raise ValueError(f"Need at least {degree + 1} data points for degree-{degree} polynomial, got {len(x)}")

    n = len(x)

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    residuals = y - y_pred

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - degree - 1)

    return {
        "regression_type": f"polynomial_{degree}",
        "n": n,
        "degree": degree,
        "coefficients": [r(c) for c in coeffs],
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "residual_std": r(np.std(residuals, ddof=degree + 1)),
        "equation": "y = " + " + ".join(f"{r(c)}*x^{degree - i}" for i, c in enumerate(coeffs)),
    }


def _multiple_regression(file, x_columns, y_column, reg_type, alpha):
    """Multiple or stepwise regression using statsmodels."""
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for multiple regression")

    from utils.data_loader import read_dataframe

    df = read_dataframe(file)

    y = df[y_column]
    X = df[x_columns]

    if reg_type == "stepwise":
        X = _stepwise_selection(X, y, alpha)

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    coefficients = {}
    for name, val in model.params.items():
        coefficients[name] = r(val)

    return {
        "regression_type": reg_type,
        "n": len(y),
        "coefficients": coefficients,
        "r_squared": r(model.rsquared),
        "adj_r_squared": r(model.rsquared_adj),
        "f_statistic": r(model.fvalue),
        "p_value": r(model.f_pvalue),
        "significant": model.f_pvalue < alpha,
        "aic": r(model.aic),
        "bic": r(model.bic),
        "selected_variables": list(X.columns) if reg_type == "stepwise" else x_columns,
    }


def _stepwise_selection(X, y, alpha_in=0.05, alpha_out=0.1):
    """Forward stepwise variable selection."""
    import statsmodels.api as sm

    included = []
    remaining = list(X.columns)

    while remaining:
        best_p = alpha_in
        best_var = None

        for var in remaining:
            trial = included + [var]
            X_trial = sm.add_constant(X[trial])
            model = sm.OLS(y, X_trial).fit()
            p_val = model.pvalues[var]
            if p_val < best_p:
                best_p = p_val
                best_var = var

        if best_var is not None:
            included.append(best_var)
            remaining.remove(best_var)
        else:
            break

    return X[included] if included else X


def _logistic(x, y, alpha):
    """Logistic regression."""
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for logistic regression")

    X = sm.add_constant(x)
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    return {
        "regression_type": "logistic",
        "n": len(y),
        "coefficients": {
            "intercept": r(model.params[0]),
            "slope": r(model.params[1]),
        },
        "p_values": {
            "intercept": r(model.pvalues[0]),
            "slope": r(model.pvalues[1]),
        },
        "aic": r(model.aic),
        "deviance": r(model.deviance),
        "significant": model.pvalues[1] < alpha,
    }


def _nonlinear(x, y, model, alpha):
    """Nonlinear regression using scipy.optimize.curve_fit."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        raise ImportError("scipy.optimize is required for nonlinear regression")

    n = len(x)

    if model == "exponential":

        def func(x, a, b, c):
            return a * np.exp(b * x) + c

        p0 = [1.0, 0.1, float(np.min(y))]
        param_names = ["a", "b", "c"]
    elif model == "power":

        def func(x, a, b):
            return a * np.power(x, b)

        p0 = [1.0, 1.0]
        param_names = ["a", "b"]
    elif model == "logarithmic":

        def func(x, a, b):
            return a * np.log(x) + b

        p0 = [1.0, 0.0]
        param_names = ["a", "b"]
    elif model == "sigmoid":

        def func(x, a, b, c, d):
            return d + (a - d) / (1 + (x / c) ** b)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        x_mid = float(np.median(x))
        p0 = [y_min, 1.0, x_mid, y_max]
        param_names = ["a", "b", "c", "d"]
    else:
        raise ValueError(f"Unknown nonlinear model: {model}")

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
        convergence_warning = any("OptimizeWarning" in str(warning.category) for warning in w)
    y_pred = func(x, *popt)
    residuals = y - y_pred

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    k = len(popt)
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k) if n > k else 0

    # Standard errors from covariance matrix diagonal
    perr = np.sqrt(np.diag(pcov))

    # Build equation string
    if model == "exponential":
        equation = f"y = {r(popt[0])} * exp({r(popt[1])} * x) + {r(popt[2])}"
    elif model == "power":
        equation = f"y = {r(popt[0])} * x^{r(popt[1])}"
    elif model == "logarithmic":
        equation = f"y = {r(popt[0])} * ln(x) + {r(popt[1])}"
    elif model == "sigmoid":
        equation = f"y = {r(popt[3])} + ({r(popt[0])} - {r(popt[3])}) / (1 + (x / {r(popt[2])})^{r(popt[1])})"

    result = {
        "regression_type": model,
        "n": n,
        "coefficients": {name: r(val) for name, val in zip(param_names, popt)},
        "coefficients_std": {name: r(val) for name, val in zip(param_names, perr)},
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "residual_std": r(np.std(residuals, ddof=k)),
        "equation": equation,
    }
    if convergence_warning:
        result["_warning"] = "OptimizeWarning: curve_fit may not have converged"
    return result


def _prepare_sklearn_xy(x, y):
    """Reshape x to 2D for sklearn and return (X_2d, y_arr)."""
    X_2d = x.reshape(-1, 1) if x.ndim == 1 else x
    return X_2d, y


def _lasso(x, y, reg_alpha):
    """LASSO regression (L1 penalty) using sklearn."""
    from sklearn.linear_model import Lasso, LassoCV

    X_2d, y_arr = _prepare_sklearn_xy(x, y)

    if reg_alpha is not None:
        model = Lasso(alpha=reg_alpha, max_iter=10000)
        model.fit(X_2d, y_arr)
        chosen_alpha = reg_alpha
    else:
        model = LassoCV(cv=5, max_iter=10000)
        model.fit(X_2d, y_arr)
        chosen_alpha = model.alpha_

    y_pred = model.predict(X_2d)
    residuals = y_arr - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(y_arr)
    n_coefs = X_2d.shape[1]
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - n_coefs - 1) if n > n_coefs + 1 else 0
    n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))

    # Build coefficient dict
    if X_2d.shape[1] == 1:
        coefficients = {"intercept": r(model.intercept_), "slope": r(model.coef_[0])}
    else:
        coefficients = {"intercept": r(model.intercept_)}
        for i in range(X_2d.shape[1]):
            coefficients[f"x{i + 1}"] = r(model.coef_[i])

    return {
        "regression_type": "lasso",
        "n": n,
        "coefficients": coefficients,
        "alpha": r(chosen_alpha),
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "n_nonzero_coefs": n_nonzero,
        "residual_std": r(np.std(residuals, ddof=n_coefs + 1)),
    }


def _ridge(x, y, reg_alpha):
    """Ridge regression (L2 penalty) using sklearn."""
    from sklearn.linear_model import Ridge, RidgeCV

    X_2d, y_arr = _prepare_sklearn_xy(x, y)

    if reg_alpha is not None:
        model = Ridge(alpha=reg_alpha)
        model.fit(X_2d, y_arr)
        chosen_alpha = reg_alpha
    else:
        model = RidgeCV(cv=5)
        model.fit(X_2d, y_arr)
        chosen_alpha = model.alpha_

    y_pred = model.predict(X_2d)
    residuals = y_arr - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(y_arr)
    n_coefs = X_2d.shape[1]
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - n_coefs - 1) if n > n_coefs + 1 else 0

    if X_2d.shape[1] == 1:
        coefficients = {"intercept": r(model.intercept_), "slope": r(model.coef_[0])}
    else:
        coefficients = {"intercept": r(model.intercept_)}
        for i in range(X_2d.shape[1]):
            coefficients[f"x{i + 1}"] = r(model.coef_[i])

    return {
        "regression_type": "ridge",
        "n": n,
        "coefficients": coefficients,
        "alpha": r(chosen_alpha),
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "residual_std": r(np.std(residuals, ddof=n_coefs + 1)),
    }


def _elastic_net(x, y, reg_alpha, l1_ratio):
    """Elastic Net regression (L1 + L2 penalty) using sklearn."""
    from sklearn.linear_model import ElasticNet, ElasticNetCV

    X_2d, y_arr = _prepare_sklearn_xy(x, y)

    if reg_alpha is not None:
        model = ElasticNet(alpha=reg_alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_2d, y_arr)
        chosen_alpha = reg_alpha
        chosen_l1_ratio = l1_ratio
    else:
        model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99], cv=5, max_iter=10000)
        model.fit(X_2d, y_arr)
        chosen_alpha = model.alpha_
        chosen_l1_ratio = model.l1_ratio_

    y_pred = model.predict(X_2d)
    residuals = y_arr - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(y_arr)
    n_coefs = X_2d.shape[1]
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - n_coefs - 1) if n > n_coefs + 1 else 0

    if X_2d.shape[1] == 1:
        coefficients = {"intercept": r(model.intercept_), "slope": r(model.coef_[0])}
    else:
        coefficients = {"intercept": r(model.intercept_)}
        for i in range(X_2d.shape[1]):
            coefficients[f"x{i + 1}"] = r(model.coef_[i])

    return {
        "regression_type": "elastic_net",
        "n": n,
        "coefficients": coefficients,
        "alpha": r(chosen_alpha),
        "l1_ratio": r(chosen_l1_ratio),
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "residual_std": r(np.std(residuals, ddof=n_coefs + 1)),
    }


def _robust(x, y, method):
    """Robust regression using M-estimators (HuberT or TukeyBiweight)."""
    import statsmodels.api as sm
    from statsmodels.robust import norms

    X_2d, y_arr = _prepare_sklearn_xy(x, y)
    X_with_const = sm.add_constant(X_2d)

    if method == "tukey":
        m_estimator = norms.TukeyBiweight()
    else:
        m_estimator = norms.HuberT()

    model = sm.RLM(y_arr, X_with_const, M=m_estimator).fit()
    y_pred = model.predict(X_with_const)
    residuals = y_arr - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(y_arr)
    n_coefs = X_2d.shape[1]
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - n_coefs - 1) if n > n_coefs + 1 else 0

    # Outliers: points where the robust weights are low (< 0.5)
    weights = model.weights
    n_outliers = int(np.sum(weights < 0.5))

    if X_2d.shape[1] == 1:
        coefficients = {"intercept": r(model.params[0]), "slope": r(model.params[1])}
    else:
        coefficients = {"intercept": r(model.params[0])}
        for i in range(X_2d.shape[1]):
            coefficients[f"x{i + 1}"] = r(model.params[i + 1])

    return {
        "regression_type": "robust",
        "n": n,
        "coefficients": coefficients,
        "method": method,
        "r_squared": r(r_sq),
        "adj_r_squared": r(adj_r_sq),
        "n_outliers": n_outliers,
        "residual_std": r(np.std(residuals, ddof=n_coefs + 1)),
    }

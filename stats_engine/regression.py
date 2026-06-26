"""Regression analysis: linear, multiple, polynomial, stepwise, logistic, nonlinear, PLS, GLM."""

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
    n_components=2,
    family="poisson",
    cv=None,
    scoring="r2",
):
    """Perform regression analysis.

    Args:
        x: List of x values (simple regression) or 2D array
        y: List of y values
        reg_type: 'linear', 'quadratic', 'polynomial', 'multiple', 'stepwise', 'logistic',
                 'exponential', 'power', 'logarithmic', 'sigmoid',
                 'lasso', 'ridge', 'elastic_net', 'robust', 'pls',
                 'poisson', 'gamma', 'negbin', 'cross_validate'
        degree: Polynomial degree (for polynomial type)
        file: Data file path (for multiple/stepwise)
        x_columns: Column names for x (for multiple/stepwise)
        y_column: Column name for y (for multiple/stepwise)
        alpha: Significance level
        reg_alpha: Regularization strength for penalized regression (lasso/ridge/elastic_net).
                   None = automatic cross-validated selection.
        l1_ratio: Elastic net mixing parameter (0=ridge, 1=lasso, default 0.5)
        method: M-estimator for robust regression ('huber' or 'tukey')
        n_components: Number of PLS components (default 2)
        family: GLM family for poisson/gamma/negbin types
        cv: Number of cross-validation folds (default 5). When provided with a
            standard reg_type, wraps it in k-fold CV. Also usable via
            reg_type='cross_validate'.
        scoring: Scoring metric for cross-validation (default 'r2').
                 Options: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'

    Returns:
        Dict with regression results
    """
    _scalar_types = ("linear", "quadratic", "polynomial", "logistic", "exponential", "power", "logarithmic", "sigmoid")
    _penalized_types = ("lasso", "ridge", "elastic_net", "robust")
    _glm_types = ("poisson", "gamma", "negbin")

    # Cross-validation mode: either explicit reg_type='cross_validate' or cv parameter
    if reg_type == "cross_validate" or (cv is not None and reg_type != "cross_validate"):
        inner_type = "linear" if reg_type == "cross_validate" else reg_type
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        # Filter NaN/Inf
        if x_arr.ndim == 2:
            mask = np.all(np.isfinite(x_arr), axis=1) & np.isfinite(y_arr)
        else:
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < len(y_arr):
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]
        return _cross_validate(x_arr, y_arr, reg_type=inner_type, cv=cv or 5, scoring=scoring)

    # Filter NaN/Inf from x and y together for array-based regression types
    if reg_type in _scalar_types or reg_type in _penalized_types or reg_type == "pls" or reg_type in _glm_types:
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
    elif reg_type == "pls":
        return _pls(x_arr, y_arr, n_components)
    elif reg_type in _glm_types:
        return _glm(x_arr, y_arr, reg_type)
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

    # Filter rows with any NaN/Inf in x or y columns
    all_cols = list(x_columns) + [y_column]
    mask = np.all(np.isfinite(df[all_cols].values), axis=1)
    df = df[mask]

    if len(df) < 2:
        raise ValueError(f"After removing NaN/Inf, only {len(df)} valid rows remain (need at least 2)")

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


def _pls(x, y, n_components):
    """Partial Least Squares regression using sklearn."""
    from sklearn.cross_decomposition import PLSRegression

    X_2d, y_arr = _prepare_sklearn_xy(x, y)
    n, p = X_2d.shape

    # Ensure n_components does not exceed min(n, p)
    max_comp = min(n, p, n_components)
    if n_components > max_comp:
        n_components = max_comp

    model = PLSRegression(n_components=n_components)
    model.fit(X_2d, y_arr)

    y_pred = model.predict(X_2d)
    if y_pred.ndim == 2:
        y_pred = y_pred.ravel()

    # R-squared
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # VIP scores: VIP_j = sqrt(p * sum(w_jk^2 * SSY_k) / sum(SSY_k))
    # w = model.x_weights_ (p x n_components), SSY_k = sum of y_scores_k^2
    w = model.x_weights_  # (p, n_components)
    y_scores = model.y_scores_  # (n, n_components)
    ssy_k = np.sum(y_scores**2, axis=0)  # (n_components,)
    total_ssy = np.sum(ssy_k)

    vip_scores = np.zeros(p)
    for j in range(p):
        vip_scores[j] = np.sqrt(p * np.sum(w[j, :] ** 2 * ssy_k) / total_ssy) if total_ssy > 0 else 0

    # Coefficients -- sklearn PLSRegression.coef_ shape is (n_targets, p)
    coefficients = {
        "intercept": r(float(model.intercept_[0]))
        if hasattr(model.intercept_, "__len__")
        else r(float(model.intercept_))
    }
    for i in range(p):
        coefficients[f"x{i + 1}"] = r(float(model.coef_[0, i])) if model.coef_.ndim == 2 else r(float(model.coef_[i]))

    vip_dict = {}
    for i in range(p):
        vip_dict[f"x{i + 1}"] = r(float(vip_scores[i]))

    return {
        "regression_type": "pls",
        "n": n,
        "n_components": n_components,
        "coefficients": coefficients,
        "vip_scores": vip_dict,
        "r_squared": r(r_sq),
        "x_scores": [[r(float(v)) for v in row] for row in model.x_scores_],
        "y_scores": [[r(float(v)) for v in row] for row in model.y_scores_],
    }


def _glm(x, y, family):
    """Generalized Linear Model using statsmodels.

    Args:
        x: 2D array of predictors
        y: 1D array of response
        family: 'poisson', 'gamma', or 'negbin'
    """
    import statsmodels.api as sm

    X_2d, y_arr = _prepare_sklearn_xy(x, y)
    X_with_const = sm.add_constant(X_2d)

    if family == "poisson":
        fam = sm.families.Poisson()
    elif family == "gamma":
        fam = sm.families.Gamma()
    elif family == "negbin":
        fam = sm.families.NegativeBinomial()
    else:
        raise ValueError(f"Unknown GLM family: {family}. Use 'poisson', 'gamma', or 'negbin'")

    model = sm.GLM(y_arr, X_with_const, family=fam).fit()

    # Pseudo R-squared (McFadden): 1 - ll_model / ll_null
    try:
        null_model = sm.GLM(y_arr, np.ones(len(y_arr)), family=fam).fit()
        pseudo_r_sq = 1 - (model.llf / null_model.llf) if null_model.llf != 0 else 0
    except Exception:
        pseudo_r_sq = None

    n = len(y_arr)
    p = X_2d.shape[1]

    # Build coefficient dict
    coefficients = {"intercept": r(model.params[0])}
    for i in range(p):
        coefficients[f"x{i + 1}"] = r(model.params[i + 1])

    # Summary table as list of lists: [name, coef, std_err, z, p_value, ci_lower, ci_upper]
    summary_table = [["name", "coef", "std_err", "z", "p_value", "ci_lower", "ci_upper"]]
    param_names = ["const"] + [f"x{i + 1}" for i in range(p)]
    ci = model.conf_int()
    for i in range(len(model.params)):
        name = param_names[i] if i < len(param_names) else f"param_{i}"
        summary_table.append(
            [
                name,
                r(model.params[i]),
                r(model.bse[i]),
                r(model.tvalues[i]),
                r(model.pvalues[i]),
                r(ci[i, 0]) if isinstance(ci, np.ndarray) else r(ci.iloc[i, 0]),
                r(ci[i, 1]) if isinstance(ci, np.ndarray) else r(ci.iloc[i, 1]),
            ]
        )

    return {
        "regression_type": f"glm_{family}",
        "n": n,
        "family": family,
        "coefficients": coefficients,
        "deviance": r(model.deviance),
        "aic": r(model.aic),
        "bic": r(model.bic),
        "pseudo_r_squared": r(pseudo_r_sq) if pseudo_r_sq is not None else None,
        "summary_table": summary_table,
    }


# Map of supported inner regression types for cross-validation to sklearn estimators
_CV_ESTIMATORS = {
    "linear": lambda: __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression(),
    "ridge": lambda: __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(),
    "lasso": lambda: __import__("sklearn.linear_model", fromlist=["Lasso"]).Lasso(max_iter=10000),
    "elastic_net": lambda: __import__("sklearn.linear_model", fromlist=["ElasticNet"]).ElasticNet(max_iter=10000),
    "robust": lambda: __import__("sklearn.linear_model", fromlist=["HuberRegressor"]).HuberRegressor(max_iter=10000),
}

_CV_SCORING_MAP = {
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
}


def _cross_validate(x, y, reg_type="linear", cv=5, scoring="r2"):
    """K-fold cross-validation for regression models.

    Args:
        x: 1D or 2D array of predictors
        y: 1D array of response
        reg_type: Inner regression type ('linear', 'ridge', 'lasso', 'elastic_net', 'robust')
        cv: Number of folds (default 5)
        scoring: Scoring metric ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
    """
    from sklearn.model_selection import KFold, cross_val_score

    if reg_type not in _CV_ESTIMATORS:
        raise ValueError(
            f"Cross-validation not supported for '{reg_type}'. Supported types: {list(_CV_ESTIMATORS.keys())}"
        )

    X_2d, y_arr = _prepare_sklearn_xy(x, y)
    n = len(y_arr)

    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}")
    if cv > n:
        raise ValueError(f"cv ({cv}) cannot exceed number of samples ({n})")

    sk_scoring = _CV_SCORING_MAP.get(scoring)
    if sk_scoring is None:
        raise ValueError(f"Unknown scoring: {scoring}. Use one of {list(_CV_SCORING_MAP.keys())}")

    estimator = _CV_ESTIMATORS[reg_type]()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, X_2d, y_arr, cv=kf, scoring=sk_scoring)

    # For negative metrics, convert to positive for interpretability
    scores_list = [r(float(s)) for s in scores]
    mean_score = r(float(np.mean(scores)))
    std_score = r(float(np.std(scores, ddof=1)))

    result = {
        "regression_type": "cross_validate",
        "inner_type": reg_type,
        "n": n,
        "cv_folds": cv,
        "scoring": scoring,
        "scores": scores_list,
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": r(float(np.min(scores))),
        "max_score": r(float(np.max(scores))),
    }

    # Add interpretation
    if scoring == "r2":
        result["interpretation"] = f"{cv}-fold CV R^2 = {mean_score} +/- {std_score} (per-fold: {scores_list})"
    else:
        result["interpretation"] = f"{cv}-fold CV {scoring} = {mean_score} +/- {std_score} (per-fold: {scores_list})"

    return result

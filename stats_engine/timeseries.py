"""Time series analysis: exponential smoothing, ARIMA, decomposition, ACF/PACF."""

import numpy as np

from utils.output import r
from utils.validators import to_array


def timeseries(analysis_type, values=None, **kwargs):
    """Perform time series analysis.

    Args:
        analysis_type: 'exp_smoothing', 'arima', 'decomposition', 'acf'

    Returns:
        Dict with time series results
    """
    if values is None:
        raise ValueError("'values' parameter is required for time series analysis")

    if analysis_type == "exp_smoothing":
        return _exp_smoothing(values, **kwargs)
    elif analysis_type == "arima":
        return _arima(values, **kwargs)
    elif analysis_type == "decomposition":
        return _decomposition(values, **kwargs)
    elif analysis_type in ("acf", "acf_pacf"):
        return _acf(values, **kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _exp_smoothing(values, alpha=None, frequency=None, n_forecast=0, **kwargs):
    """Exponential smoothing."""
    values = to_array(values, min_n=3, name="values")
    if n_forecast < 0:
        raise ValueError(f"n_forecast must be >= 0, got {n_forecast}")
    n = len(values)

    if alpha is None:
        # Optimize alpha
        best_alpha = 0.2
        best_sse = float("inf")
        for a in np.arange(0.05, 1.0, 0.05):
            fitted = _simple_exp_smooth(values, a)
            sse = np.sum((values[1:] - fitted[:-1]) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_alpha = a
        alpha = best_alpha

    fitted = _simple_exp_smooth(values, alpha)

    result = {
        "analysis_type": "exp_smoothing",
        "n": n,
        "alpha": r(alpha),
        "fitted_values": [r(v) for v in fitted],
        "residuals": [r(values[i] - fitted[i]) for i in range(n)],
        "interpretation": f"Exponential smoothing: alpha = {r(alpha)}",
    }

    if n_forecast > 0:
        # Simple exponential smoothing: forecast = last fitted value
        last_fitted = fitted[-1]
        forecast = [r(last_fitted)] * n_forecast
        result["forecast"] = forecast
        result["n_forecast"] = n_forecast
        # Approximate CI using residual standard error
        residuals = np.array([values[i] - fitted[i] for i in range(n)])
        res_std = float(np.std(residuals, ddof=1))
        result["forecast_ci_95"] = [
            [r(last_fitted - 1.96 * res_std * np.sqrt(h)), r(last_fitted + 1.96 * res_std * np.sqrt(h))]
            for h in range(1, n_forecast + 1)
        ]

    return result


def _simple_exp_smooth(values, alpha):
    """Simple exponential smoothing."""
    n = len(values)
    fitted = np.zeros(n)
    fitted[0] = values[0]
    for i in range(1, n):
        fitted[i] = alpha * values[i] + (1 - alpha) * fitted[i - 1]
    return fitted


def _arima(values, order=None, n_forecast=0, **kwargs):
    """ARIMA analysis (using statsmodels)."""
    values = to_array(values, min_n=5, name="values")
    if n_forecast < 0:
        raise ValueError(f"n_forecast must be >= 0, got {n_forecast}")
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("statsmodels required for ARIMA")

    if order is None:
        order = (1, 1, 1)

    model = ARIMA(values, order=order)
    fitted = model.fit()

    result = {
        "analysis_type": "arima",
        "n": len(values),
        "order": list(order),
        "aic": r(fitted.aic),
        "bic": r(fitted.bic),
        "parameters": {k: r(v) for k, v in zip(fitted.param_names, fitted.params)},
        "fitted_values": [r(v) for v in fitted.fittedvalues],
        "interpretation": f"ARIMA({','.join(str(o) for o in order)}): {n_forecast}-step forecast",
    }

    if n_forecast > 0:
        # Use get_forecast() to obtain confidence intervals
        forecast_obj = fitted.get_forecast(steps=n_forecast)
        forecast_mean = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        result["forecast"] = [r(v) for v in forecast_mean]
        # conf_int may be DataFrame (pandas) or ndarray depending on statsmodels version
        if hasattr(conf_int, "iloc"):
            ci_pairs = [[r(row.iloc[0]), r(row.iloc[1])] for _, row in conf_int.iterrows()]
        elif hasattr(conf_int, "shape") and conf_int.ndim == 2:
            ci_pairs = [[r(row[0]), r(row[1])] for row in conf_int]
        else:
            ci_pairs = []
        result["forecast_ci_95"] = ci_pairs
        result["n_forecast"] = n_forecast

    return result


def _decomposition(values, frequency=12, model="additive", **kwargs):
    """Time series decomposition."""
    values = to_array(values, min_n=8, name="values")
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        raise ImportError("statsmodels required for decomposition")

    result_decomp = seasonal_decompose(values, model=model, period=frequency)

    return {
        "analysis_type": "decomposition",
        "n": len(values),
        "frequency": frequency,
        "model": model,
        "trend": [r(v) if not np.isnan(v) else None for v in result_decomp.trend],
        "seasonal": [r(v) for v in result_decomp.seasonal],
        "residual": [r(v) if not np.isnan(v) else None for v in result_decomp.resid],
        "interpretation": f"Time series decomposition: {frequency}-period seasonality",
    }


def _acf(values, max_lag=None, **kwargs):
    """ACF and PACF analysis."""
    values = to_array(values, min_n=3, name="values")
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        raise ImportError("statsmodels required for ACF/PACF")

    n = len(values)
    if max_lag is None:
        max_lag = min(n // 2, 40)

    acf_vals = acf(values, nlags=max_lag, fft=True)
    pacf_vals = pacf(values, nlags=max_lag)

    # Confidence interval (95%)
    ci = 1.96 / np.sqrt(n)

    n_significant = sum(1 for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci)

    return {
        "analysis_type": "acf",
        "n": n,
        "max_lag": max_lag,
        "acf": [r(v) for v in acf_vals],
        "pacf": [r(v) for v in pacf_vals],
        "confidence_interval": r(ci),
        "significant_lags_acf": [int(i) for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci],
        "significant_lags_pacf": [int(i) for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci],
        "interpretation": f"ACF: {n_significant} significant lags",
    }

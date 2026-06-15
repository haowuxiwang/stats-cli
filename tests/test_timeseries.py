"""Tests for stats_engine/timeseries.py."""

import numpy as np
import pytest

from stats_engine.timeseries import timeseries


@pytest.fixture
def ts_data():
    np.random.seed(42)
    trend = np.linspace(10, 20, 50)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(50) / 12)
    noise = np.random.normal(0, 0.5, 50)
    return (trend + seasonal + noise).tolist()


def test_exp_smoothing(ts_data):
    result = timeseries(analysis_type="exp_smoothing", values=ts_data, frequency=12)
    assert result["analysis_type"] == "exp_smoothing"
    assert "fitted_values" in result


def test_arima(ts_data):
    result = timeseries(analysis_type="arima", values=ts_data, order=(1, 1, 1))
    assert result["analysis_type"] == "arima"
    assert "fitted_values" in result
    assert "parameters" in result
    assert "aic" in result


def test_decomposition(ts_data):
    result = timeseries(analysis_type="decomposition", values=ts_data, frequency=12)
    assert result["analysis_type"] == "decomposition"


def test_acf_pacf(ts_data):
    result = timeseries(analysis_type="acf_pacf", values=ts_data)
    assert result["analysis_type"] == "acf"


def test_forecast(ts_data):
    result = timeseries(analysis_type="arima", values=ts_data, order=(1, 1, 1), n_forecast=5)
    assert "forecast" in result
    assert len(result["forecast"]) == 5


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        timeseries(analysis_type="invalid", values=[1, 2, 3])


def test_no_values():
    """Timeseries without values should raise error."""
    with pytest.raises(ValueError, match="values"):
        timeseries(analysis_type="exp_smoothing")


def test_arima_default_order(ts_data):
    """ARIMA with default order."""
    result = timeseries(analysis_type="arima", values=ts_data)
    assert result["analysis_type"] == "arima"


def test_acf_default_max_lag(ts_data):
    """ACF with default max_lag."""
    result = timeseries(analysis_type="acf", values=ts_data)
    assert result["analysis_type"] == "acf"


def test_exp_smoothing_no_frequency(ts_data):
    """Exponential smoothing without frequency."""
    result = timeseries(analysis_type="exp_smoothing", values=ts_data)
    assert result["analysis_type"] == "exp_smoothing"

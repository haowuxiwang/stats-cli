"""Additional tests for stats_engine/timeseries.py to improve coverage."""

import numpy as np

from main import handler


class TestTimeseriesEdgeCases:
    """Test timeseries edge cases."""

    def test_timeseries_exp_smoothing(self):
        """Exponential smoothing timeseries."""
        values = np.sin(np.linspace(0, 4 * np.pi, 50)).tolist()
        result = handler({
            "command": "timeseries",
            "params": {
                "values": values,
                "analysis_type": "exp_smoothing",
            },
        })
        assert result["status"] == "success"

    def test_timeseries_arima(self):
        """ARIMA timeseries."""
        values = list(range(1, 21))
        result = handler({
            "command": "timeseries",
            "params": {
                "values": values,
                "analysis_type": "arima",
                "order": [1, 1, 0],
            },
        })
        assert result["status"] == "success"

    def test_timeseries_decomposition(self):
        """Timeseries decomposition."""
        values = (np.sin(np.linspace(0, 4 * np.pi, 50)) + np.linspace(0, 5, 50)).tolist()
        result = handler({
            "command": "timeseries",
            "params": {
                "values": values,
                "analysis_type": "decomposition",
                "period": 10,
            },
        })
        assert result["status"] == "success"

    def test_timeseries_acf(self):
        """ACF analysis."""
        values = np.random.randn(50).tolist()
        result = handler({
            "command": "timeseries",
            "params": {
                "values": values,
                "analysis_type": "acf",
            },
        })
        assert result["status"] == "success"

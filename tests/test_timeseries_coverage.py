"""Additional tests for stats_engine/timeseries.py to improve coverage."""

import builtins
import importlib
import sys

import numpy as np
import pytest

from main import handler


class TestTimeseriesEdgeCases:
    """Test timeseries edge cases."""

    def test_timeseries_exp_smoothing(self):
        """Exponential smoothing timeseries."""
        values = np.sin(np.linspace(0, 4 * np.pi, 50)).tolist()
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": values,
                    "analysis_type": "exp_smoothing",
                },
            }
        )
        assert result["status"] == "success"

    def test_timeseries_arima(self):
        """ARIMA timeseries."""
        values = list(range(1, 21))
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": values,
                    "analysis_type": "arima",
                    "order": [1, 1, 0],
                },
            }
        )
        assert result["status"] == "success"

    def test_timeseries_decomposition(self):
        """Timeseries decomposition."""
        values = (np.sin(np.linspace(0, 4 * np.pi, 50)) + np.linspace(0, 5, 50)).tolist()
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": values,
                    "analysis_type": "decomposition",
                    "period": 10,
                },
            }
        )
        assert result["status"] == "success"

    def test_timeseries_acf(self):
        """ACF analysis."""
        values = np.random.randn(50).tolist()
        result = handler(
            {
                "command": "timeseries",
                "params": {
                    "values": values,
                    "analysis_type": "acf",
                },
            }
        )
        assert result["status"] == "success"


class TestTimeseriesImportErrors:
    """Test ImportError guards in timeseries functions."""

    def test_arima_import_error(self):
        """_arima raises ImportError when statsmodels is missing (lines 83-84)."""

        values = list(range(1, 21))

        # Save and remove statsmodels from sys.modules
        saved = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("statsmodels")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        # Block future imports of statsmodels
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "statsmodels" or name.startswith("statsmodels."):
                raise ImportError(f"mocked: no module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        # Also clear the function's local reference to ARIMA if cached
        importlib.reload(sys.modules["stats_engine.timeseries"])
        from stats_engine.timeseries import _arima as _arima_fresh

        try:
            with pytest.raises(ImportError, match="statsmodels required for ARIMA"):
                _arima_fresh(values, order=(1, 1, 0))
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(sys.modules["stats_engine.timeseries"])

    def test_decomposition_import_error(self):
        """_decomposition raises ImportError when statsmodels is missing (lines 114-115)."""

        values = (np.sin(np.linspace(0, 4 * np.pi, 50)) + np.linspace(0, 5, 50)).tolist()

        saved = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("statsmodels")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "statsmodels" or name.startswith("statsmodels."):
                raise ImportError(f"mocked: no module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        importlib.reload(sys.modules["stats_engine.timeseries"])
        from stats_engine.timeseries import _decomposition as _decomposition_fresh

        try:
            with pytest.raises(ImportError, match="statsmodels required for decomposition"):
                _decomposition_fresh(values, frequency=10)
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(sys.modules["stats_engine.timeseries"])

    def test_acf_import_error(self):
        """_acf raises ImportError when statsmodels is missing (lines 134-135)."""

        values = np.random.randn(50).tolist()

        saved = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("statsmodels")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "statsmodels" or name.startswith("statsmodels."):
                raise ImportError(f"mocked: no module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        importlib.reload(sys.modules["stats_engine.timeseries"])
        from stats_engine.timeseries import _acf as _acf_fresh

        try:
            with pytest.raises(ImportError, match="statsmodels required for ACF/PACF"):
                _acf_fresh(values)
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(sys.modules["stats_engine.timeseries"])

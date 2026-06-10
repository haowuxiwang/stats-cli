"""Standard JSON output wrapper."""

import json
from datetime import date, datetime, time, timezone

import numpy as np
import pandas as pd

VERSION = "1.1.0"

# Global precision for numeric output. All stat modules use r() instead of
# hardcoding round(..., N). Change this single value to adjust all output.
DEFAULT_PRECISION = 6


def r(val, precision=None):
    """Round a numeric value to the configured precision.

    Args:
        val: Value to round (int, float, numpy scalar, or None)
        precision: Decimal places (defaults to DEFAULT_PRECISION)

    Returns:
        Rounded float, or None if val is None
    """
    if val is None:
        return None
    if precision is None:
        precision = DEFAULT_PRECISION
    return round(float(val), precision)


def success(data):
    """Wrap successful result in standard envelope."""
    return {
        "status": "success",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


def error(message, error_type="UNKNOWN_ERROR", suggestion=None):
    """Wrap error in standard envelope."""
    result = {
        "status": "error",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": error_type,
        "message": message,
    }
    if suggestion:
        result["suggestion"] = suggestion
    return result


def to_json(result):
    """Convert result dict to JSON string."""
    return json.dumps(_to_native(result), indent=2, ensure_ascii=False)


def _to_native(obj):
    """Recursively convert numpy/pandas types to Python native types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, float):
        return None if (obj != obj or obj == float("inf") or obj == float("-inf")) else obj
    if isinstance(obj, (pd.Timestamp, datetime, date, time)):
        return str(obj)
    return obj

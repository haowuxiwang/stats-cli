"""Standard JSON output wrapper."""

import json
from datetime import date, datetime, time, timezone

import numpy as np
import pandas as pd

VERSION = "1.2.1"

# Global precision for numeric output. All stat modules use r() instead of
# hardcoding round(..., N). Change this single value to adjust all output.
DEFAULT_PRECISION = 6

# Precision constants for specific field types
PRECISION = {
    "default": 6,
    "percent": 2,
    "statistic": 4,  # F, t, chi2 statistics
    "p_value": 4,
    "effect_size": 3,
    "kappa": 3,
    "ppm": 2,
}


# Standardized error types
class ErrorType:
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_COMMAND = "MISSING_COMMAND"
    PARAM_ERROR = "PARAM_ERROR"
    DATA_ERROR = "DATA_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    MEMORY_ERROR = "MEMORY_ERROR"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


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


def warning(data, message, suggestion=None):
    """Wrap result with status: warning — data is valid but degraded."""
    result = {
        "status": "warning",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": message,
        "data": data,
    }
    if suggestion:
        result["suggestion"] = suggestion
    return result


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


def p_value_context(result, p_value, alpha=0.05, n=None):
    """Add credibility context fields to a result dict containing p_value.

    Mutates and returns result. Adds:
    - p_boundary: True if p is near alpha (within 20% margin)
    - p_boundary_warning: human-readable warning string
    - small_sample_warning: if n < 10
    - _warning: triggers status:warning envelope in main.py handler

    Args:
        result: Dict to augment
        p_value: The computed p-value
        alpha: Significance level (default 0.05)
        n: Sample size (optional, triggers small-sample warning)
    """
    warnings = []
    if p_value is not None and alpha * 0.8 < p_value < alpha * 1.2:
        result["p_boundary"] = True
        result["p_boundary_warning"] = (
            f"p={r(p_value)} is near alpha={alpha}; result is borderline — interpret with caution"
        )
        warnings.append(result["p_boundary_warning"])
    if n is not None and n < 10:
        result["small_sample_warning"] = f"n={n}: statistical power is very low, results may be unreliable"
        warnings.append(result["small_sample_warning"])
    if warnings:
        result["_warning"] = "; ".join(warnings)
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

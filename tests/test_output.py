"""Tests for utils/output.py."""

import numpy as np
import pandas as pd

from utils.output import VERSION, _to_native, error, r, success, to_json


def test_success_envelope():
    result = success({"key": "value"})
    assert result["status"] == "success"
    assert result["version"] == VERSION
    assert "timestamp" in result
    assert result["data"] == {"key": "value"}


def test_error_envelope():
    result = error("test error", "TEST_ERROR")
    assert result["status"] == "error"
    assert result["error_type"] == "TEST_ERROR"
    assert result["message"] == "test error"
    assert "timestamp" in result


def test_error_with_suggestion():
    result = error("fail", "ERR", suggestion="try again")
    assert result["suggestion"] == "try again"


def test_error_without_suggestion():
    result = error("fail", "ERR")
    assert "suggestion" not in result


def test_to_json():
    result = success({"a": 1})
    j = to_json(result)
    assert '"status": "success"' in j
    assert '"a": 1' in j


def test_version_string():
    assert VERSION == "1.3.1"


def test_r_rounding():
    assert r(3.14159, 2) == 3.14
    assert r(3.14159265, 6) == 3.141593  # DEFAULT_PRECISION = 6
    assert r(None) is None


def test_to_native_numpy_bool():
    assert _to_native(np.bool_(True)) is True
    assert _to_native(np.bool_(False)) is False


def test_to_native_numpy_int():
    assert _to_native(np.int64(42)) == 42
    assert isinstance(_to_native(np.int64(42)), int)


def test_to_native_numpy_float():
    assert _to_native(np.float64(3.14)) == 3.14
    assert isinstance(_to_native(np.float64(3.14)), float)


def test_to_native_numpy_nan():
    assert _to_native(np.float64(np.nan)) is None


def test_to_native_numpy_inf():
    assert _to_native(np.float64(np.inf)) is None
    assert _to_native(np.float64(-np.inf)) is None


def test_to_native_python_nan():
    assert _to_native(float("nan")) is None


def test_to_native_python_inf():
    assert _to_native(float("inf")) is None
    assert _to_native(float("-inf")) is None


def test_to_native_numpy_array():
    arr = np.array([1, 2, 3])
    result = _to_native(arr)
    assert result == [1, 2, 3]


def test_to_native_dict():
    d = {"a": np.int64(1), "b": np.float64(2.0)}
    result = _to_native(d)
    assert result == {"a": 1, "b": 2.0}


def test_to_native_list():
    lst = [np.int64(1), np.float64(2.0)]
    result = _to_native(lst)
    assert result == [1, 2.0]


def test_to_native_tuple():
    tpl = (np.int64(1), np.float64(2.0))
    result = _to_native(tpl)
    assert result == [1, 2.0]


def test_to_native_pandas_timestamp():
    ts = pd.Timestamp("2024-01-01")
    result = _to_native(ts)
    assert isinstance(result, str)


def test_to_native_string():
    assert _to_native("hello") == "hello"


def test_to_native_int():
    assert _to_native(42) == 42

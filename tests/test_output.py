"""Tests for utils/output.py."""

from utils.output import VERSION, error, success, to_json


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

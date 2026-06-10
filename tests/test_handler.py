"""Tests for main.py handler and routing."""

from main import handler


def test_handler_string_input():
    result = handler('{"command": "descriptive", "params": {"values": [1, 2, 3]}}')
    assert result["status"] == "success"
    assert "data" in result


def test_handler_dict_input():
    result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "success"


def test_handler_invalid_json():
    result = handler("not json")
    assert result["status"] == "error"
    assert result["error_type"] == "INVALID_INPUT"


def test_handler_missing_command():
    result = handler({"params": {}})
    assert result["status"] == "error"
    assert result["error_type"] == "MISSING_COMMAND"


def test_handler_unknown_command():
    result = handler({"command": "nonexistent"})
    assert result["status"] == "error"
    assert result["error_type"] == "VALIDATION_ERROR"


def test_handler_validation_error():
    result = handler({"command": "capability", "params": {"values": [1, 2, 3]}})
    assert result["status"] == "error"
    assert result["error_type"] == "VALIDATION_ERROR"


def test_handler_discover():
    result = handler({"command": "discover", "params": {}})
    assert result["status"] == "success"
    assert "total_commands" in result["data"]


def test_handler_discover_specific():
    result = handler({"command": "discover", "params": {"command_name": "descriptive"}})
    assert result["status"] == "success"
    assert "description" in result["data"]


def test_handler_discover_category():
    result = handler({"command": "discover", "params": {"category": "basic"}})
    assert result["status"] == "success"
    assert result["data"]["total_commands"] > 0

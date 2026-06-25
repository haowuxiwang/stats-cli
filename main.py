"""stats-cli-py: Pure Python statistical analysis tool for Aily.

Entry point for Aily skill invocation.
Accepts JSON input, routes to statistical modules, returns JSON output.
"""

import json
import logging
import sys

from utils.output import ErrorType, error, success, to_json, warning


def handler(input_data):
    """Main handler function for Aily.

    Args:
        input_data: Dict with 'command' and 'params' keys, or JSON string

    Returns:
        Dict with standard output envelope
    """
    # Parse input if string
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError as e:
            return error(f"Invalid JSON input / 无效的JSON输入: {e}", ErrorType.INVALID_INPUT)

    command = input_data.get("command")
    params = input_data.get("params", {})
    # Protect caller's dict from mutation
    params = dict(params or {})

    if not command:
        return error(
            "Missing 'command' field / 缺少 'command' 字段. Use 'discover' to list available commands.",
            ErrorType.MISSING_COMMAND,
            suggestion="使用 'discover' 命令查看所有可用命令",
        )

    # Route to command handler
    want_chart = params.pop("chart", False)
    try:
        result = _route(command, params)
        # Generate chart if requested
        if want_chart and isinstance(result, dict):
            # Inject raw values for chart generation
            if "values" in params and "_values" not in result:
                result["_values"] = params["values"]
            result["chart_base64"] = _generate_chart(command, result, params)
            result.pop("_values", None)  # Clean up temp field
        # Check for warning marker from module
        warning_msg = result.pop("_warning", None) if isinstance(result, dict) else None
        suggestion = result.pop("_warning_suggestion", None) if isinstance(result, dict) else None
        if warning_msg:
            return warning(result, warning_msg, suggestion)
        return success(result)
    except ValueError as e:
        return error(str(e), _classify_value_error(str(e)))
    except FileNotFoundError as e:
        return error(str(e), ErrorType.FILE_NOT_FOUND)
    except PermissionError as e:
        return error(
            f"Permission denied / 权限不足: {e}",
            ErrorType.PARAM_ERROR,
            suggestion="Check if the path is a directory or file requires elevated permissions",
        )
    except ImportError as e:
        return error(
            str(e),
            ErrorType.MISSING_DEPENDENCY,
            suggestion="pip install -r requirements.txt / 缺少依赖，请运行 pip install -r requirements.txt",
        )
    except MemoryError:
        return error(
            "Out of memory / 内存不足. Try reducing data size or use file-based input.",
            ErrorType.MEMORY_ERROR,
        )
    except TypeError as e:
        msg = str(e)
        # Clean up Python function signatures from error messages
        # "descriptive() missing 1 required positional argument: 'values'" -> "Missing required parameter: 'values'"
        import re

        msg = re.sub(r"\w+\(\)\s*", "", msg)  # Remove function_name()
        if "unexpected keyword argument" in msg:
            return error(msg, ErrorType.PARAM_ERROR)
        if "missing" in msg and "required" in msg and "argument" in msg:
            return error(f"Missing required parameter / 缺少必填参数: {msg}", ErrorType.PARAM_ERROR)
        return error(f"Invalid parameter type / 参数类型错误: {msg}", ErrorType.PARAM_ERROR)
    except Exception as e:
        logging.exception("Unexpected error in command handler")
        return error(f"Internal error / 内部错误: {e}", ErrorType.INTERNAL_ERROR)


def _classify_value_error(msg):
    """Classify ValueError message into granular error type.

    - PARAM_ERROR: parameter missing, invalid type, unknown command
    - DATA_ERROR: insufficient data, no numeric columns, format issues
    - COMPUTATION_ERROR: mathematical limits (zero variance, undefined)
    """
    msg_lower = msg.lower()
    # Data-related errors
    data_keywords = [
        "need at least",
        "no numeric",
        "no valid",
        "not found in file",
        "not found.",
        "after removing",
        "after cleaning",
        "insufficient",
        "no data",
        "empty",
        "column",
        "sheet",
    ]
    if any(kw in msg_lower for kw in data_keywords):
        return ErrorType.DATA_ERROR
    # Computation-related errors
    comp_keywords = [
        "cannot",
        "undefined",
        "zero variance",
        "identical",
        "convergence",
        "singular",
        "infinity",
    ]
    if any(kw in msg_lower for kw in comp_keywords):
        return ErrorType.COMPUTATION_ERROR
    # Default to parameter error
    return ErrorType.PARAM_ERROR


# Command registry: command_name -> (module_path, function_name)
# Commands not in this registry: 'discover' (handled specially), 'run' (handled specially)
COMMAND_REGISTRY = {
    "explore": ("stats_engine.explore", "explore"),
    "power": ("stats_engine.power", "power"),
    "regression": ("stats_engine.regression", "regression"),
    "multivariate": ("stats_engine.multivariate", "multivariate"),
    "descriptive": ("stats_engine.descriptive", "descriptive"),
    "normality": ("stats_engine.normality", "normality"),
    "ttest": ("stats_engine.ttest", "ttest"),
    "anova": ("stats_engine.anova", "anova"),
    "correlation": ("stats_engine.correlation", "correlation"),
    "outlier": ("stats_engine.outlier", "outlier"),
    "control_chart": ("stats_engine.control_chart", "control_chart"),
    "capability": ("stats_engine.capability", "capability"),
    "trend": ("stats_engine.trend", "trend"),
    "nonparametric": ("stats_engine.nonparametric", "nonparametric"),
    "homogeneity": ("stats_engine.homogeneity", "homogeneity"),
    "equivalence": ("stats_engine.equivalence", "equivalence"),
    "multiple_comparison": ("stats_engine.multiple_comparison", "multiple_comparison"),
    "doe": ("stats_engine.doe", "doe"),
    "gage_rr": ("stats_engine.gage_rr", "gage_rr"),
    "reliability": ("stats_engine.reliability", "reliability"),
    "timeseries": ("stats_engine.timeseries", "timeseries"),
    "advanced": ("stats_engine.advanced", "advanced"),
    "clean": ("stats_engine.clean", "clean"),
    "transform": ("stats_engine.transform", "transform"),
    "report": ("stats_engine.report", "report"),
    "workflow": ("stats_engine.workflow", "workflow"),
    "check_assumptions": ("stats_engine.assumptions", "check_assumptions"),
    "recommend": ("stats_engine.assumptions", "recommend_test"),
    "workflow_template": ("stats_engine.workflow", "workflow_template"),
    "export_excel": ("stats_engine.report", "export_excel"),
    "export_pdf": ("stats_engine.report", "export_pdf"),
    "distribution": ("stats_engine.distribution", "distribution"),
    "bayesian": ("stats_engine.bayesian", "bayesian"),
    "mining": ("stats_engine.mining", "mining"),
    "sensitivity": ("stats_engine.sensitivity", "sensitivity"),
    "acceptance_sampling": ("stats_engine.acceptance_sampling", "acceptance_sampling"),
    "functional": ("stats_engine.functional", "functional"),
}

# Commands that don't need file-based data loading
NO_DATA_COMMANDS = {
    "discover",
    "explore",
    "power",
    "multivariate",
    "run",
    "mining",
    "sensitivity",
    "acceptance_sampling",
    "functional",
}

# Commands that need two-column file loading (x and y)
TWO_COLUMN_COMMANDS = {"correlation", "regression"}


def _route(command, params):
    """Route command to appropriate statistical module."""
    import importlib

    # Special commands
    if command == "discover":
        return _discover(params)
    if command == "run":
        return _run_script(params)

    # Lookup command in registry
    if command not in COMMAND_REGISTRY:
        raise ValueError(f"Unknown command: {command}. Use 'discover' to list available commands.")

    # Two-column file loading for correlation/regression
    if command in TWO_COLUMN_COMMANDS and "file" in params and "x" not in params:
        x_col = params.get("x_column")
        y_col = params.get("y_column")
        if x_col and y_col:
            from utils.data_loader import load_data

            header = params.pop("header", 0)
            params.pop("x_column")
            params.pop("y_column")
            data = load_data(
                file=params.pop("file"),
                columns=[x_col, y_col],
                sheet=params.pop("sheet", None),
                header=header,
            )
            params["x"] = data["x"]
            params["y"] = data["y"]
        # else: file stays in params for regression's internal handling (multiple/stepwise)

    # Single-column file loading for other commands
    elif command not in NO_DATA_COMMANDS and "file" in params and "values" not in params:
        from utils.data_loader import load_data

        header = params.pop("header", 0)
        data = load_data(
            file=params.pop("file"),
            column=params.pop("column", None),
            sheet=params.pop("sheet", None),
            header=header,
        )
        params["values"] = data["values"]

    # Lazy import and call
    module_path, func_name = COMMAND_REGISTRY[command]
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func(**params)


def _discover(params):
    """List all available commands and their schemas."""
    from stats_engine.discover import COMMANDS

    command_name = params.get("command_name")

    if command_name:
        if command_name not in COMMANDS:
            raise ValueError(f"Unknown command: {command_name}")
        return COMMANDS[command_name]

    category = params.get("category", "all")
    filtered = {}
    for name, cmd in COMMANDS.items():
        if category == "all" or cmd.get("category") == category:
            filtered[name] = cmd

    return {
        "total_commands": len(filtered),
        "commands": {
            name: {"description": cmd["description"], "category": cmd["category"], "example": cmd.get("example", "")}
            for name, cmd in filtered.items()
        },
    }


def _run_script(params):
    """Run custom Python script with data.

    Security: Blocks dangerous patterns (dunder access, imports, file/network ops).
    Timeout: Uses threading + ctypes to enforce timeout on infinite loops.
    Intended for simple data transformations only, NOT untrusted code execution.
    """
    import ctypes
    import threading

    script = params.get("script")
    data = params.get("data", {})
    timeout = params.get("timeout", 5)  # Default 5 seconds

    if not script:
        raise ValueError("'script' parameter is required")

    # Security: block dangerous patterns
    _DANGEROUS_PATTERNS = [
        "__",  # dunder access (class, subclasses, builtins, import)
        "import ",  # import statements
        "open(",  # file operations
        "exec(",  # nested exec
        "eval(",  # eval
        "compile(",  # compile
        "globals(",  # globals/locals access
        "locals(",
        "getattr(",  # attribute access
        "setattr(",
        "delattr(",
        "os.",  # os module
        "sys.",  # sys module
        "subprocess",  # subprocess
        "shutil",  # file operations
        "pathlib",  # path operations
        "socket",  # network
        "urllib",  # network
        "requests",  # network
    ]
    script_lower = script.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in script_lower:
            raise ValueError(
                f"Script contains blocked pattern: '{pattern}'. "
                "The run command supports simple data transformations only. "
                "Blocked: dunder access, imports, file/network operations."
            )

    # Restricted namespace - no imports, no file access
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace = {"data": data, "result": None, "__builtins__": safe_builtins}

    # Execute with timeout using threading + ctypes
    result_holder = {"result": None, "error": None}

    def _exec():
        try:
            exec(script, namespace)
            result_holder["result"] = namespace.get("result", {"message": "Script executed"})
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=_exec, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Force terminate the thread by raising SystemExit
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident),
                ctypes.py_object(SystemExit),
            )
        except (ValueError, SystemError):
            pass
        thread.join(timeout=1)
        raise ValueError(
            f"Script execution timed out after {timeout} seconds. Increase 'timeout' parameter or optimize your script."
        )

    if result_holder["error"]:
        raise result_holder["error"]

    return result_holder["result"]


def _generate_chart(command, result, params):
    """Generate chart for supported commands. Returns base64 PNG or None."""
    try:
        from stats_engine.chart_handlers import CHART_HANDLERS

        handler = CHART_HANDLERS.get(command)
        if handler:
            return handler(result, params)
    except Exception as e:
        logging.warning("Chart generation failed for %s: %s", command, e)
    return None


def main():
    """CLI entry point - read JSON from stdin or file."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        if len(sys.argv) > 1:
            with open(sys.argv[1]) as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(sys.stdin.read())
    except FileNotFoundError:
        print(to_json(error(f"File not found: {sys.argv[1]}", ErrorType.FILE_NOT_FOUND)))
        return
    except json.JSONDecodeError as e:
        print(to_json(error(f"Invalid JSON input: {e}", ErrorType.INVALID_INPUT)))
        return
    except Exception as e:
        print(to_json(error(f"Failed to read input: {e}", ErrorType.INVALID_INPUT)))
        return

    result = handler(input_data)
    try:
        print(to_json(result))
    except (BrokenPipeError, OSError):
        # Pipe closed (e.g., piped to head/tail), exit silently
        pass


if __name__ == "__main__":
    main()

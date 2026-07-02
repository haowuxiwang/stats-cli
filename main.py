"""stats-cli-py: Pure Python statistical analysis tool for Aily.

Entry point for Aily skill invocation.
Accepts JSON input, routes to statistical modules, returns JSON output.
"""

import copy
import json
import logging
import sys

from utils.output import ErrorType, error, success, to_json, warning

# Maximum input size for in-memory arrays (prevents OOM from accidental large input)
MAX_VALUES_SIZE = 100_000


def _validate_input_size(params):
    """Check input data size limits. Returns error dict or None."""
    values = params.get("values")
    if values and isinstance(values, list) and len(values) > MAX_VALUES_SIZE:
        return error(
            f"Input 'values' exceeds maximum size ({len(values)} > {MAX_VALUES_SIZE}). "
            "Use file-based input or sample your data.",
            ErrorType.DATA_ERROR,
            suggestion=f"Reduce to <={MAX_VALUES_SIZE} points or use 'file' parameter",
        )
    groups = params.get("groups")
    if groups and isinstance(groups, list):
        total = sum(len(g) for g in groups if isinstance(g, list))
        if total > MAX_VALUES_SIZE:
            return error(
                f"Input 'groups' total size ({total}) exceeds maximum ({MAX_VALUES_SIZE}).",
                ErrorType.DATA_ERROR,
                suggestion="Reduce data size or use file-based input",
            )
    for key in ("x", "y", "times", "measurements"):
        arr = params.get(key)
        if arr and isinstance(arr, list) and len(arr) > MAX_VALUES_SIZE:
            return error(
                f"Input '{key}' exceeds maximum size ({len(arr)} > {MAX_VALUES_SIZE}).",
                ErrorType.DATA_ERROR,
                suggestion=f"Reduce to <={MAX_VALUES_SIZE} points or use 'file' parameter",
            )
    return None


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
    # Deep copy to protect caller's nested structures from mutation
    params = copy.deepcopy(input_data.get("params", {}))
    if not isinstance(params, dict):
        params = {}

    if not command:
        return error(
            "Missing 'command' field / 缺少 'command' 字段. Use 'discover' to list available commands.",
            ErrorType.MISSING_COMMAND,
            suggestion="使用 'discover' 命令查看所有可用命令",
        )

    # Validate input size limits
    size_error = _validate_input_size(params)
    if size_error:
        return size_error

    # Extract chart customization params (chart_title, chart_bins, etc.)
    # IMPORTANT: chart_type is a REAL param for some commands (e.g., control_chart) — don't strip it!
    want_chart = params.pop("chart", False)
    _CHART_RESERVED = {"chart_type"}  # params that start with "chart_" but are real command params
    chart_custom = {k: v for k, v in list(params.items()) if k.startswith("chart_") and k not in _CHART_RESERVED}
    # Remove chart custom params so they don't confuse command handlers
    for k in list(chart_custom.keys()):
        params.pop(k, None)

    try:
        result = _route(command, params)
        # _route may return error dict for unknown commands
        if isinstance(result, dict) and result.get("status") == "error":
            return result
        # Generate chart if requested
        chart_error = None
        if want_chart and isinstance(result, dict):
            # Inject raw values for chart generation
            if "values" in params and "_values" not in result:
                result["_values"] = params["values"]
            # Inject chart customization
            if chart_custom:
                result["_chart_custom"] = chart_custom
            try:
                result["chart_base64"] = _generate_chart(command, result, params)
            except Exception as chart_exc:
                chart_error = f"Chart generation failed: {chart_exc}"
                logging.warning("Chart generation failed for %s: %s", command, chart_exc)
            # Clean up temp fields
            result.pop("_values", None)
            result.pop("_chart_custom", None)
        # Check for warning marker from module
        if isinstance(result, dict):
            warning_msg = result.pop("_warning", None)
            warning_suggestion = result.pop("_warning_suggestion", None)
            # Add chart error as warning if present
            if chart_error and "chart_error" not in result:
                result["chart_error"] = chart_error
            if warning_msg:
                return warning(result, warning_msg, warning_suggestion)
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
    "measurement_uncertainty": ("stats_engine.uncertainty", "measurement_uncertainty"),
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
    "measurement_uncertainty",
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
        return error(
            f"Unknown command: '{command}'. Use 'discover' to list available commands.",
            ErrorType.UNKNOWN_COMMAND,
            suggestion=f"Did you mean one of: {', '.join(sorted(COMMAND_REGISTRY.keys())[:5])}...?",
        )

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

    # Security: AST-based analysis (more robust than string blacklist)
    import ast

    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Script syntax error: {e}")

    # Walk AST to block dangerous constructs
    _ALLOWED_NODES = (
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.keyword,
        ast.Attribute,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.For,
        ast.While,
        ast.If,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Try,
        ast.ExceptHandler,
        ast.With,
        ast.withitem,
        ast.ListComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Return,
        ast.Yield,
        ast.Lambda,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.IfExp,
        ast.Starred,
        ast.NamedExpr,
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise ValueError("Script contains import statement. The run command does not support imports.")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Script contains 'from ... import' statement. The run command does not support imports.")
        # Block dunder attribute access (e.g., __class__, __bases__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Script accesses dunder attribute '{node.attr}'. Dunder access is blocked for security.")
        # Block __builtins__ access via Name node
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"Script references '{node.id}'. Dunder names are blocked for security.")

    # Restricted namespace - no print (prevents stdout pollution)
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
            # Pass chart customization from result metadata
            custom = result.get("_chart_custom", {}) or {}
            if custom:
                # Merge customization into params for handler use
                params = {**params, **custom}
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

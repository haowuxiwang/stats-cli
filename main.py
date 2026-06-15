"""stats-cli-py: Pure Python statistical analysis tool for Aily.

Entry point for Aily skill invocation.
Accepts JSON input, routes to statistical modules, returns JSON output.
"""

import json
import logging
import sys

from utils.output import error, success, to_json


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
            return error(f"Invalid JSON input / 无效的JSON输入: {e}", "INVALID_INPUT")

    command = input_data.get("command")
    params = input_data.get("params", {})
    # Protect caller's dict from mutation
    params = dict(params or {})

    if not command:
        return error(
            "Missing 'command' field / 缺少 'command' 字段. Use 'discover' to list available commands.",
            "MISSING_COMMAND",
            suggestion="使用 'discover' 命令查看所有可用命令",
        )

    # Route to command handler
    want_chart = params.pop("chart", False)
    try:
        result = _route(command, params)
        # Generate chart if requested
        if want_chart:
            # Inject raw values for chart generation
            if "values" in params and "_values" not in result:
                result["_values"] = params["values"]
            result["chart_base64"] = _generate_chart(command, result, params)
            result.pop("_values", None)  # Clean up temp field
        return success(result)
    except ValueError as e:
        return error(str(e), "VALIDATION_ERROR")
    except FileNotFoundError as e:
        return error(str(e), "FILE_NOT_FOUND")
    except ImportError as e:
        return error(
            str(e),
            "MISSING_DEPENDENCY",
            suggestion="pip install -r requirements.txt / 缺少依赖，请运行 pip install -r requirements.txt",
        )
    except MemoryError:
        return error(
            "Out of memory / 内存不足. Try reducing data size or use file-based input.",
            "MEMORY_ERROR",
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            return error(str(e), "VALIDATION_ERROR")
        return error(f"{type(e).__name__}: {e}", "INTERNAL_ERROR")
    except Exception as e:
        return error(f"{type(e).__name__}: {e}", "INTERNAL_ERROR")


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
}

# Commands that don't need file-based data loading
NO_DATA_COMMANDS = {"discover", "explore", "power", "regression", "multivariate", "run"}


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

    # Load data from file if needed
    if command not in NO_DATA_COMMANDS and "file" in params and "values" not in params:
        from utils.data_loader import load_data
        header = params.pop("header", 0)
        data = load_data(
            file=params.pop("file"), column=params.pop("column", None),
            sheet=params.pop("sheet", None), header=header,
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
    """Run custom Python script with data."""
    script = params.get("script")
    data = params.get("data", {})

    if not script:
        raise ValueError("'script' parameter is required")

    # Restricted namespace - no imports, no file access
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "format": format,
        "frozenset": frozenset, "int": int, "isinstance": isinstance, "len": len,
        "list": list, "map": map, "max": max, "min": min, "pow": pow,
        "print": print, "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "slice": slice, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    }
    namespace = {"data": data, "result": None, "__builtins__": safe_builtins}
    exec(script, namespace)
    return namespace.get("result", {"message": "Script executed"})


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
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    try:
        if len(sys.argv) > 1:
            with open(sys.argv[1]) as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(sys.stdin.read())
    except FileNotFoundError:
        print(to_json(error(f"File not found: {sys.argv[1]}", "FILE_NOT_FOUND")))
        return
    except json.JSONDecodeError as e:
        print(to_json(error(f"Invalid JSON input: {e}", "INVALID_JSON")))
        return
    except Exception as e:
        print(to_json(error(f"Failed to read input: {e}", "INPUT_ERROR")))
        return

    result = handler(input_data)
    print(to_json(result))


if __name__ == "__main__":
    main()

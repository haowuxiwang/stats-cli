"""stats-cli-py: Pure Python statistical analysis tool for Aily.

Entry point for Aily skill invocation.
Accepts JSON input, routes to statistical modules, returns JSON output.
"""

import json
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
    params = dict(params)

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


def _route(command, params):
    """Route command to appropriate statistical module."""
    from utils.data_loader import load_data

    # Commands that don't need data loading
    if command == "discover":
        return _discover(params)
    if command == "explore":
        from stats_engine.explore import explore

        return explore(**params)
    if command == "power":
        from stats_engine.power import power

        return power(**params)
    if command == "regression":
        from stats_engine.regression import regression

        return regression(**params)
    if command == "multivariate":
        from stats_engine.multivariate import multivariate

        return multivariate(**params)

    # Load data if file provided
    if "file" in params and "values" not in params:
        header = params.pop("header", 0)
        data = load_data(
            file=params.pop("file"), column=params.pop("column", None),
            sheet=params.pop("sheet", None), header=header,
        )
        params["values"] = data["values"]

    # Route to specific command
    if command == "descriptive":
        from stats_engine.descriptive import descriptive

        return descriptive(**params)
    elif command == "normality":
        from stats_engine.normality import normality

        return normality(**params)
    elif command == "ttest":
        from stats_engine.ttest import ttest

        return ttest(**params)
    elif command == "anova":
        from stats_engine.anova import anova

        return anova(**params)
    elif command == "correlation":
        from stats_engine.correlation import correlation

        return correlation(**params)
    elif command == "outlier":
        from stats_engine.outlier import outlier

        return outlier(**params)
    elif command == "control_chart":
        from stats_engine.control_chart import control_chart

        return control_chart(**params)
    elif command == "capability":
        from stats_engine.capability import capability

        return capability(**params)
    elif command == "trend":
        from stats_engine.trend import trend

        return trend(**params)
    elif command == "nonparametric":
        from stats_engine.nonparametric import nonparametric

        return nonparametric(**params)
    elif command == "homogeneity":
        from stats_engine.homogeneity import homogeneity

        return homogeneity(**params)
    elif command == "equivalence":
        from stats_engine.equivalence import equivalence

        return equivalence(**params)
    elif command == "multiple_comparison":
        from stats_engine.multiple_comparison import multiple_comparison

        return multiple_comparison(**params)
    elif command == "doe":
        from stats_engine.doe import doe

        return doe(**params)
    elif command == "gage_rr":
        from stats_engine.gage_rr import gage_rr

        return gage_rr(**params)
    elif command == "reliability":
        from stats_engine.reliability import reliability

        return reliability(**params)
    elif command == "timeseries":
        from stats_engine.timeseries import timeseries

        return timeseries(**params)
    elif command == "advanced":
        from stats_engine.advanced import advanced

        return advanced(**params)
    elif command == "clean":
        from stats_engine.clean import clean

        return clean(**params)
    elif command == "transform":
        from stats_engine.transform import transform

        return transform(**params)
    elif command == "report":
        from stats_engine.report import report

        return report(**params)
    elif command == "run":
        return _run_script(params)
    else:
        raise ValueError(f"Unknown command: {command}. Use 'discover' to list available commands.")


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
        from stats_engine.charts import (
            capability_plot,
            control_chart_plot,
            histogram,
            qq_plot,
            scatter_with_regression,
            time_series_plot,
        )

        if command == "descriptive":
            return histogram(result.get("_values", []), title="Distribution")

        elif command == "normality":
            return qq_plot(result.get("_values", []), title="Q-Q Plot")

        elif command == "control_chart":
            return control_chart_plot(
                result.get("_values", []),
                chart_type=result.get("chart_type", "imr"),
                ucl=result.get("ucl"),
                cl=result.get("cl"),
                lcl=result.get("lcl"),
                out_of_control=result.get("out_of_control", []),
            )

        elif command == "capability":
            return capability_plot(
                result.get("_values", []),
                usl=result.get("usl"),
                lsl=result.get("lsl"),
                target=result.get("target"),
                cp=result.get("cp"),
                cpk=result.get("cpk"),
            )

        elif command == "correlation":
            return scatter_with_regression(
                params.get("x", []), params.get("y", []),
                title="Correlation",
            )

        elif command == "regression":
            coeffs = result.get("coefficients", {})
            slope = intercept = None
            if isinstance(coeffs, dict):
                slope = coeffs.get("slope")
                intercept = coeffs.get("intercept")
            return scatter_with_regression(
                params.get("x", []), params.get("y", []),
                title=f'Regression ({result.get("regression_type", "")})',
                slope=slope, intercept=intercept,
                r_squared=result.get("r_squared"),
            )

        elif command == "timeseries":
            return time_series_plot(
                result.get("_values", []),
                title=f'Time Series ({result.get("analysis_type", "")})',
                fitted=result.get("fitted_values"),
                forecast=result.get("forecast"),
            )

        elif command == "report":
            vals = result.get("_values", [])
            if vals:
                return histogram(vals, title="Report - Distribution")
        elif command == "doe":
            # DOE charts (Pareto of factor effects) not yet implemented
            return None
    except Exception:
        pass
    return None


def main():
    """CLI entry point - read JSON from stdin or file."""
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

"""Data exploration - inspect file structure and basic stats."""

import json
from pathlib import Path

import numpy as np

from utils.output import r


def explore(file, sheet=None, rows=5, header=0):
    """Explore data file structure.

    Args:
        file: Path to data file
        sheet: Sheet name/index for Excel
        rows: Number of sample rows to show
        header: Row number for header (0-indexed, None = no header)

    Returns:
        Dict with file structure info
    """
    path = Path(file)
    if not path.exists():
        return {"error": True, "message": f"File not found: {file}"}

    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return _explore_excel(path, sheet, rows, header)
    elif suffix == ".csv":
        return _explore_csv(path, rows, header)
    elif suffix == ".json":
        return _explore_json(path, rows)
    else:
        return _explore_text(path, rows)


def _explore_excel(path, sheet, rows, header=0):
    """Explore Excel file with multi-sheet awareness."""
    import pandas as pd

    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names

    if sheet is not None:
        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(sheet_names):
                return {"error": True, "message": f"Sheet index {sheet} out of range. Available: {len(sheet_names)} sheets"}
            sheet_name = sheet_names[sheet]
        else:
            sheet_name = str(sheet)
            if sheet_name not in sheet_names:
                return {"error": True, "message": f"Sheet '{sheet_name}' not found. Available: {sheet_names}"}
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    else:
        sheet_name = sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)

    result = _describe_dataframe(df, rows, str(path))
    result["sheet"] = sheet_name
    result["n_sheets"] = len(sheet_names)
    result["sheets"] = sheet_names
    if len(sheet_names) > 1 and sheet is None:
        result["hint"] = f"This workbook has {len(sheet_names)} sheets. Use 'sheet' parameter to explore other sheets: {sheet_names}"
    return result


def _explore_csv(path, rows, header=0):
    """Explore CSV file."""
    import pandas as pd

    from utils.data_loader import detect_delimiter, detect_encoding

    encoding = detect_encoding(str(path))
    delimiter = detect_delimiter(str(path), encoding)

    df = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=header)

    return _describe_dataframe(df, rows, str(path))


def _explore_json(path, rows):
    """Explore JSON file."""
    with open(path) as f:
        data = json.load(f)

    result = {
        "file": str(path),
        "format": "json",
        "type": type(data).__name__,
    }

    if isinstance(data, dict):
        result["keys"] = list(data.keys())
        if "values" in data:
            vals = data["values"]
            result["n_values"] = len(vals)
            result["sample_values"] = vals[:rows]
    elif isinstance(data, list):
        result["n_items"] = len(data)
        result["sample_items"] = data[:rows]

    return result


def _explore_text(path, rows):
    """Explore text file."""
    with open(path) as f:
        lines = f.readlines()

    values = []
    non_numeric = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                values.append(float(line))
            except ValueError:
                non_numeric.append(line)

    result = {
        "file": str(path),
        "format": "text",
        "total_lines": len(lines),
        "n_numeric": len(values),
        "n_non_numeric": len(non_numeric),
        "sample_values": values[:rows],
    }

    if values:
        arr = np.array(values)
        result["basic_stats"] = {
            "mean": r(np.mean(arr)),
            "min": r(np.min(arr)),
            "max": r(np.max(arr)),
        }

    return result


def _describe_dataframe(df, rows, file_path):
    """Describe a pandas DataFrame with smart type inference."""
    import pandas as pd

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Try to convert object columns to numeric (handle mixed data)
    inferred_numeric = {}
    for col in text_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        n_valid = converted.notna().sum()
        n_total = df[col].notna().sum()
        if n_total > 0 and n_valid / n_total >= 0.8:
            inferred_numeric[col] = converted
            numeric_cols.append(col)
            text_cols.remove(col)

    columns = []
    for col in df.columns:
        col_info = {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null": int(df[col].isna().sum()),
        }
        # Use inferred numeric series if available
        series = inferred_numeric.get(col, df[col])
        if col in numeric_cols:
            clean = series.dropna()
            if len(clean) > 0:
                col_info["inferred_numeric"] = col in inferred_numeric
                col_info["mean"] = r(clean.mean())
                col_info["min"] = r(clean.min())
                col_info["max"] = r(clean.max())
        columns.append(col_info)

    return {
        "file": file_path,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "columns": columns,
        "sample_data": df.head(rows).to_dict(orient="records"),
    }

"""Data loading from Excel, CSV, JSON, text files."""

import csv
import io
import json
from pathlib import Path

from utils.data_cleaner import clean_values


def detect_encoding(file_path):
    """Detect file encoding. Try utf-8-sig first, then utf-8, then latin-1."""
    for enc in ["utf-8-sig", "utf-8", "latin-1", "gbk", "gb2312"]:
        try:
            with open(file_path, encoding=enc) as f:
                f.read(1024)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def detect_delimiter(file_path, encoding="utf-8"):
    """Detect CSV delimiter by sampling first few lines."""
    try:
        with open(file_path, encoding=encoding) as f:
            sample = f.read(4096)
    except Exception:
        return ","

    # Count occurrences of common delimiters
    delimiters = [",", "\t", ";", "|"]
    counts = {d: sample.count(d) for d in delimiters}

    # Pick the most common one
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_dataframe(file, columns=None, sheet=None, header=0, max_rows=1_000_000):
    """Read a file into a pandas DataFrame with proper encoding detection.

    Public API for modules that need full DataFrame access (multivariate, regression).
    Handles file existence check, encoding detection, sheet selection, and column validation.

    Args:
        file: Path to data file (Excel, CSV, TSV)
        columns: Optional list of column names to select
        sheet: Sheet name/index for Excel files
        header: Row number for header (0-indexed)
        max_rows: Maximum rows to load (default 1,000,000). Raises ValueError if exceeded.

    Returns:
        pandas DataFrame

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is unsupported, columns not found, or row limit exceeded
    """
    import pandas as pd

    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file}")

    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        if sheet is not None:
            if isinstance(sheet, int):
                if sheet < 0 or sheet >= len(xl.sheet_names):
                    raise ValueError(f"Sheet index {sheet} out of range. Available: {len(xl.sheet_names)} sheets")
                sheet_name = xl.sheet_names[sheet]
            else:
                sheet_name = str(sheet)
                if sheet_name not in xl.sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found. Available: {xl.sheet_names}")
        else:
            sheet_name = xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    elif suffix in (".csv", ".tsv"):
        encoding = detect_encoding(str(path))
        delimiter = detect_delimiter(str(path), encoding)
        df = pd.read_csv(path, delimiter=delimiter, encoding=encoding, header=header)
    else:
        raise ValueError(f"read_dataframe requires Excel or CSV/TSV file, got: {suffix}")

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")
        df = df[columns]

    if len(df) > max_rows:
        raise ValueError(
            f"File has {len(df)} rows, exceeding max_rows limit of {max_rows}. "
            "Use a smaller file or increase max_rows parameter."
        )

    return df


def load_data(values=None, file=None, column=None, columns=None, sheet=None, header=0):
    """Load data from various sources.

    Args:
        values: Direct list of numeric values
        file: Path to data file (Excel, CSV, JSON, text)
        column: Column name or index to extract (single column)
        columns: List of two column names for dual-column loading (returns {"x": [...], "y": [...]})
        sheet: Sheet name/index for Excel files
        header: Row number for header (0-indexed, None = no header)

    Returns:
        Dict with 'values' key (list of floats) and optional '_cleaning_report',
        or Dict with 'x' and 'y' keys when columns is provided
    """
    if values is not None:
        cleaned, report = clean_values(values, min_count=1)
        result = {"values": cleaned}
        if report.has_changes():
            result["_cleaning_report"] = report.to_dict()
        return result

    if file is not None:
        if columns is not None:
            return _load_two_columns(file, columns, sheet, header)
        return _load_file(file, column, sheet, header)

    raise ValueError("Either 'values' or 'file' must be provided")


def _load_file(file_path, column=None, sheet=None, header=0):
    """Load data from a file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return _load_excel(path, column, sheet, header)
    elif suffix in (".csv", ".tsv"):
        return _load_csv(path, column, header)
    elif suffix == ".json":
        return _load_json(path)
    else:
        return _load_text(path)


def _load_two_columns(file_path, columns, sheet=None, header=0):
    """Load two columns from a tabular file for correlation/regression.

    Args:
        file_path: Path to data file
        columns: List of two column names [x_column, y_column]
        sheet: Sheet name/index for Excel files
        header: Row number for header

    Returns:
        Dict with 'x' and 'y' keys (lists of floats)
    """
    if not isinstance(columns, (list, tuple)) or len(columns) != 2:
        raise ValueError("columns must be a list of two column names [x_column, y_column]")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for multi-column file loading")

    if suffix in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        if sheet is not None:
            sheet_name = xl.sheet_names[sheet] if isinstance(sheet, int) else str(sheet)
            df = pd.read_excel(path, sheet_name=sheet_name, header=header)
        else:
            df = pd.read_excel(path, sheet_name=0, header=header)
    elif suffix in (".csv", ".tsv"):
        encoding = detect_encoding(str(path))
        delimiter = detect_delimiter(str(path), encoding)
        df = pd.read_csv(path, delimiter=delimiter, encoding=encoding, header=header)
    else:
        raise ValueError(f"Two-column loading requires Excel or CSV/TSV file, got: {suffix}")

    x_col, y_col = columns
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found. Available: {list(df.columns)}")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found. Available: {list(df.columns)}")

    x_vals = pd.to_numeric(df[x_col], errors="coerce").dropna().tolist()
    y_vals = pd.to_numeric(df[y_col], errors="coerce").dropna().tolist()

    # Align by minimum length after dropping NaN independently
    min_len = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]

    if min_len < 2:
        raise ValueError(f"After cleaning, only {min_len} valid paired values remain (need at least 2)")

    return {"x": x_vals, "y": y_vals}


def _load_excel(path, column=None, sheet=None, header=0):
    """Load data from Excel file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to read Excel files. Install with: pip install pandas openpyxl")

    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names

    if sheet is not None:
        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(sheet_names):
                raise ValueError(f"Sheet index {sheet} out of range. Available: {len(sheet_names)} sheets")
            sheet_name = sheet_names[sheet]
        else:
            sheet_name = str(sheet)
            if sheet_name not in sheet_names:
                raise ValueError(f"Sheet '{sheet_name}' not found. Available: {sheet_names}")
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    else:
        df = pd.read_excel(path, sheet_name=0, header=header)

    if column is not None:
        values = _extract_column(df, column, path)
    else:
        import pandas as pd

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # Try to convert object columns to numeric (handle mixed data)
        if len(numeric_cols) == 0:
            for col in df.select_dtypes(include=["object"]).columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                n_valid = converted.notna().sum()
                n_total = df[col].notna().sum()
                if n_total > 0 and n_valid / n_total >= 0.8:
                    numeric_cols.append(col)
                    df[col] = converted
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in Excel file")
        values = df[numeric_cols[0]].dropna().tolist()

    # Convert to float
    raw = []
    for v in values:
        if isinstance(v, (int, float)):
            raw.append(v)
        elif hasattr(v, "timestamp"):
            raw.append(v.timestamp())
        else:
            raw.append(v)

    cleaned, report = clean_values(raw, min_count=1)
    result = {"values": cleaned}
    if report.has_changes():
        result["_cleaning_report"] = report.to_dict()
    if len(sheet_names) > 1:
        result["_available_sheets"] = sheet_names
    return result


def _extract_column(df, column, path):
    """Extract column from DataFrame by name, numeric name, or index."""
    # Try as column name
    if column in df.columns:
        return df[column].dropna().tolist()

    # Try as numeric column name
    try:
        col_num = float(column)
        for col in df.columns:
            try:
                if float(col) == col_num:
                    return df[col].dropna().tolist()
            except (ValueError, TypeError):
                continue
    except (ValueError, TypeError):
        pass

    # Try as column index
    try:
        col_idx = int(column)
        if 0 <= col_idx < len(df.columns):
            return df.iloc[:, col_idx].dropna().tolist()
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Column '{column}' not found in file")


def _load_csv(path, column=None, header=0):
    """Load data from CSV file."""
    encoding = detect_encoding(str(path))
    delimiter = detect_delimiter(str(path), encoding)

    with open(path, encoding=encoding) as f:
        content = f.read().strip()

    # Strip BOM
    if content.startswith("﻿"):
        content = content[1:]

    if header is None:
        # No header: generate column names
        first_line = content.splitlines()[0]
        n_cols = len(first_line.split(delimiter))
        col_names = [f"col{i}" for i in range(n_cols)]
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter, fieldnames=col_names)
    else:
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)

    if column is not None:
        values = _extract_csv_column(reader, column, content, delimiter)
    else:
        first_row = next(reader)
        col_name = None
        for k, v in first_row.items():
            try:
                float(v)
                col_name = k
                break
            except (ValueError, TypeError):
                continue

        if col_name is None:
            raise ValueError("No numeric columns found in CSV file")

        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        values = []
        for row in reader:
            try:
                values.append(float(row[col_name]))
            except (ValueError, TypeError):
                pass

    cleaned, report = clean_values(values, min_count=1)
    result = {"values": cleaned}
    if report.has_changes():
        result["_cleaning_report"] = report.to_dict()
    return result


def _extract_csv_column(reader, column, content, delimiter):
    """Extract column from CSV by name or index."""
    # Try by name
    try:
        values = []
        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, TypeError):
                pass
        if values:
            return values
    except KeyError:
        pass

    # Try by index
    try:
        col_idx = int(column)
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        values = []
        for row in reader:
            try:
                values.append(float(list(row.values())[col_idx]))
            except (ValueError, TypeError, IndexError):
                pass
        if values:
            return values
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Column '{column}' not found in CSV file")


def _load_json(path):
    """Load data from JSON file."""
    encoding = detect_encoding(str(path))
    with open(path, encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, dict) and "values" in data:
        cleaned, report = clean_values(data["values"], min_count=1)
        data["values"] = cleaned
        if report.has_changes():
            data["_cleaning_report"] = report.to_dict()
    return data


def _load_text(path):
    """Load data from plain text (one value per line)."""
    encoding = detect_encoding(str(path))
    with open(path, encoding=encoding) as f:
        content = f.read().strip()

    raw = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                raw.append(float(line))
            except ValueError:
                pass

    cleaned, report = clean_values(raw, min_count=1)
    result = {"values": cleaned}
    if report.has_changes():
        result["_cleaning_report"] = report.to_dict()
    return result

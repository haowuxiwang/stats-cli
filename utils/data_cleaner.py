"""Data cleaning utilities - handle NaN, Inf, non-numeric values."""

import math


class CleaningReport:
    """Track what was cleaned from the data."""

    def __init__(self):
        self.n_original = 0
        self.n_clean = 0
        self.nan_removed = 0
        self.inf_removed = 0
        self.non_numeric_removed = 0

    def has_changes(self):
        return self.n_original != self.n_clean

    def to_dict(self):
        return {
            "n_original": self.n_original,
            "n_clean": self.n_clean,
            "nan_removed": self.nan_removed,
            "inf_removed": self.inf_removed,
            "non_numeric_removed": self.non_numeric_removed,
        }


def clean_values(raw_values, min_count=1):
    """Clean a list of values: remove NaN, Inf, non-numeric.

    Args:
        raw_values: List of raw values (strings, numbers, etc.)
        min_count: Minimum number of clean values required

    Returns:
        Tuple of (cleaned_list, CleaningReport)

    Raises:
        ValueError: If fewer than min_count clean values remain
    """
    report = CleaningReport()
    report.n_original = len(raw_values)
    cleaned = []

    for v in raw_values:
        # Try to convert to float
        try:
            fv = float(v)
        except (ValueError, TypeError):
            report.non_numeric_removed += 1
            continue

        # Check for NaN
        if math.isnan(fv):
            report.nan_removed += 1
            continue

        # Check for Inf
        if math.isinf(fv):
            report.inf_removed += 1
            continue

        cleaned.append(fv)

    report.n_clean = len(cleaned)

    if len(cleaned) < min_count:
        raise ValueError(
            f"Need at least {min_count} valid numeric values, "
            f"got {len(cleaned)} after cleaning "
            f"(removed {report.nan_removed} NaN, {report.inf_removed} Inf, "
            f"{report.non_numeric_removed} non-numeric)"
        )

    return cleaned, report

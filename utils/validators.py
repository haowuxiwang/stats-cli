"""Input validation and data preparation utilities."""

import numpy as np


def to_array(values, min_n=2, name="values"):
    """Convert values to numpy array, filter non-finite values, validate minimum count.

    Args:
        values: Input data (list, array, or other iterable)
        min_n: Minimum number of valid values required (default 2)
        name: Name for error messages

    Returns:
        numpy array with only finite values

    Raises:
        ValueError: If fewer than min_n valid values remain after filtering
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_n:
        raise ValueError(f"'{name}' must have at least {min_n} valid values, got {len(arr)}")
    return arr


def validate_values(values, min_count=2, name="values"):
    """Validate that values is a list of numbers with minimum count."""
    if not isinstance(values, list):
        raise ValueError(f"'{name}' must be a list, got {type(values).__name__}")
    if len(values) < min_count:
        raise ValueError(f"'{name}' must have at least {min_count} values, got {len(values)}")
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            raise ValueError(f"'{name}[{i}]' must be numeric, got {type(v).__name__}: {v}")


def validate_spec_limits(usl=None, lsl=None):
    """Validate specification limits."""
    if usl is not None and lsl is not None:
        if usl <= lsl:
            raise ValueError(f"USL ({usl}) must be greater than LSL ({lsl})")
    if usl is None and lsl is None:
        raise ValueError("At least one specification limit (USL or LSL) is required")


def validate_groups(groups, min_groups=2):
    """Validate group data for ANOVA-type analyses."""
    if not isinstance(groups, list):
        raise ValueError("'groups' must be a list of lists")
    if len(groups) < min_groups:
        raise ValueError(f"Need at least {min_groups} groups, got {len(groups)}")
    for i, g in enumerate(groups):
        if not isinstance(g, list) or len(g) < 1:
            raise ValueError(f"Group {i} must be a non-empty list")


def validate_alpha(alpha):
    """Validate significance level."""
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")


def validate_subgroup_size(size, n):
    """Validate subgroup size for control charts."""
    if size < 2:
        raise ValueError("Subgroup size must be at least 2")
    if n < size:
        raise ValueError(f"Not enough data ({n}) for subgroup size {size}")

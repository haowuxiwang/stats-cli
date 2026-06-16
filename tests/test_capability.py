"""Tests for stats_engine/capability.py."""

import numpy as np
import pytest

from stats_engine.capability import capability


@pytest.fixture
def capable_data():
    np.random.seed(42)
    return np.random.normal(10, 0.5, 100).tolist()


def test_two_sided(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8)
    assert "cp" in result
    assert "cpk" in result
    assert "pp" in result
    assert "ppk" in result
    assert "cpm" in result
    assert "rating" in result
    assert "interpretation" in result
    assert "performance" in result
    assert "yield_pct" in result["performance"]


def test_one_sided_upper(capable_data):
    result = capability(values=capable_data, usl=12)
    assert "cp" in result
    assert "cpk" in result


def test_one_sided_lower(capable_data):
    result = capability(values=capable_data, lsl=8)
    assert "cp" in result
    assert "cpk" in result


def test_no_specs():
    with pytest.raises(ValueError, match="At least one"):
        capability(values=[1, 2, 3])


def test_confidence_intervals(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8)
    assert "cp_ci_lower" in result
    assert "cp_ci_upper" in result
    assert "cpk_ci_lower" in result
    assert "cpk_ci_upper" in result


def test_rating(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8)
    assert result["rating"] in ["Excellent", "Good", "Marginal", "Poor"]


def test_boxcox(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8, capability_type="boxcox")
    assert "boxcox" in result
    assert "lambda" in result["boxcox"]
    assert "cp" in result["boxcox"]


def test_target_default(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8)
    assert result["target"] == 10.0  # midpoint


def test_target_custom(capable_data):
    result = capability(values=capable_data, usl=12, lsl=8, target=10.5)
    assert result["target"] == 10.5


def test_cpk_range():
    """Cpk should be between 0 and Cp for centered data."""
    np.random.seed(42)
    values = np.random.normal(10, 0.5, 200).tolist()
    result = capability(values=values, usl=12.0, lsl=8.0)
    assert 0 < result["cp"] < 5, f"Cp={result['cp']}"
    assert 0 < result["cpk"] <= result["cp"], f"Cpk={result['cpk']} > Cp={result['cp']}"


def test_cpk_one_sided_upper():
    """Upper-only spec should give valid Cpk."""
    np.random.seed(42)
    values = np.random.normal(10, 1, 200).tolist()
    result = capability(values=values, usl=15.0)
    assert result["cpk"] is not None
    assert result["cpk"] > 0


def test_capability_rbar():
    """Capability with R-bar sigma method."""
    values = [10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 10.0]
    result = capability(values=values, usl=11.0, lsl=9.0, sigma_method="rbar", subgroup_size=2)
    assert result["sigma_method"] == "rbar"
    assert "R-bar" in result["sigma_method_desc"]
    assert result["std_within"] > 0


def test_capability_sbar():
    """Capability with S-bar sigma method."""
    values = [10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 10.0, 10.1, 10.2, 10.0]
    result = capability(values=values, usl=11.0, lsl=9.0, sigma_method="sbar", subgroup_size=5)
    assert result["sigma_method"] == "sbar"
    assert "S-bar" in result["sigma_method_desc"]
    assert result["std_within"] > 0


def test_capability_rbar_invalid_subgroup():
    """R-bar with invalid subgroup_size raises ValueError."""
    with pytest.raises(ValueError, match="subgroup_size"):
        capability(values=[1, 2, 3, 4, 5], usl=10, lsl=0, sigma_method="rbar", subgroup_size=1)


def test_capability_boxcox_non_positive():
    """Capability with Box-Cox transformation on data with non-positive values."""
    from stats_engine.capability import capability
    values = [10.1, 10.2, 10.0, 10.3, 10.1, 10.2, 10.0, 10.3, 10.1, 10.2,
              10.0, 10.3, 10.1, 10.2, 10.0, 10.3, 10.1, 10.2, 10.0, 10.3,
              10.1, 10.2, 10.0, 10.3, 10.1]
    result = capability(values=values, usl=11.0, lsl=9.0, capability_type="boxcox")
    assert "boxcox" in result
    assert "lambda" in result["boxcox"]
    assert "std_transformed" in result["boxcox"]


def test_capability_boxcox_with_offset():
    """Box-Cox with non-positive values triggers offset branch."""
    from stats_engine.capability import capability
    # Values include zero/negative to trigger offset path (line 239)
    values = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
              9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23]
    result = capability(values=values, usl=25, lsl=-5, capability_type="boxcox")
    assert "boxcox" in result
    assert result["boxcox"]["offset"] > 0


def test_capability_unknown_sigma_method():
    """Unknown sigma method raises ValueError."""
    from stats_engine.capability import capability
    with pytest.raises(ValueError, match="Unknown sigma_method"):
        capability(values=[1, 2, 3, 4, 5], usl=10, lsl=0, sigma_method="invalid")


def test_capability_rbar_subgroup_too_large():
    """R-bar with subgroup_size > 10 raises ValueError."""
    from stats_engine.capability import capability
    with pytest.raises(ValueError, match="subgroup_size"):
        capability(values=[1, 2, 3, 4, 5], usl=10, lsl=0, sigma_method="rbar", subgroup_size=11)


def test_capability_sbar_subgroup_none():
    """S-bar without subgroup_size raises ValueError."""
    from stats_engine.capability import capability
    with pytest.raises(ValueError, match="subgroup_size"):
        capability(values=[1, 2, 3, 4, 5], usl=10, lsl=0, sigma_method="sbar")

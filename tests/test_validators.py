"""Tests for utils/validators.py."""

import pytest

from utils.validators import (
    validate_alpha,
    validate_groups,
    validate_spec_limits,
    validate_subgroup_size,
    validate_values,
)


class TestValidateValues:
    def test_valid(self):
        validate_values([1, 2, 3])  # should not raise

    def test_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_values("not a list")

    def test_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_values([1])

    def test_min_count(self):
        validate_values([1], min_count=1)  # should not raise

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="must be numeric"):
            validate_values([1, "a", 3])

    def test_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_values([])


class TestValidateSpecLimits:
    def test_both_valid(self):
        validate_spec_limits(usl=10, lsl=5)  # should not raise

    def test_usl_less_than_lsl(self):
        with pytest.raises(ValueError, match="USL.*must be greater"):
            validate_spec_limits(usl=5, lsl=10)

    def test_neither(self):
        with pytest.raises(ValueError, match="At least one"):
            validate_spec_limits()

    def test_only_usl(self):
        validate_spec_limits(usl=10)  # should not raise

    def test_only_lsl(self):
        validate_spec_limits(lsl=5)  # should not raise


class TestValidateGroups:
    def test_valid(self):
        validate_groups([[1, 2], [3, 4]])  # should not raise

    def test_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_groups("not")

    def test_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_groups([[1, 2]])

    def test_empty_group(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_groups([[1, 2], []])


class TestValidateAlpha:
    def test_valid(self):
        validate_alpha(0.05)  # should not raise

    def test_zero(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_alpha(0)

    def test_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_alpha(1)

    def test_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_alpha(-0.1)


class TestValidateSubgroupSize:
    def test_valid(self):
        validate_subgroup_size(5, 20)  # should not raise

    def test_too_small(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_subgroup_size(1, 20)

    def test_not_enough_data(self):
        with pytest.raises(ValueError, match="Not enough data"):
            validate_subgroup_size(5, 3)

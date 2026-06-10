"""Tests for stats_engine/gage_rr.py."""

import numpy as np
import pytest

from stats_engine.gage_rr import gage_rr


@pytest.fixture
def gage_data():
    np.random.seed(42)
    parts = list(range(1, 11)) * 6  # 10 parts, 3 operators, 2 replicates
    operators = (["O1"] * 10 + ["O2"] * 10 + ["O3"] * 10) * 2
    measurements = np.random.normal(100, 2, 60).tolist()
    return measurements, parts, operators


def test_crossed(gage_data):
    measurements, parts, operators = gage_data
    result = gage_rr(analysis_type="crossed", measurements=measurements,
                     parts=parts, operators=operators)
    assert result["analysis_type"] == "crossed"
    assert "variance_components" in result
    assert "contribution" in result
    assert "study_variation" in result
    assert "ndc" in result
    assert "rating" in result


def test_nested(gage_data):
    measurements, parts, operators = gage_data
    result = gage_rr(analysis_type="nested", measurements=measurements,
                     parts=parts, operators=operators)
    assert result["analysis_type"] == "nested"
    assert "variance_components" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        gage_rr(analysis_type="invalid", measurements=[1, 2], parts=[1, 2], operators=["A", "B"])

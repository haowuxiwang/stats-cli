"""Tests for stats_engine/control_chart.py."""

import numpy as np
import pytest

from stats_engine.control_chart import control_chart


@pytest.fixture
def spc_data():
    np.random.seed(42)
    return np.random.normal(100, 2, 100).tolist()


def test_xbar(spc_data):
    result = control_chart("xbar", spc_data, subgroup_size=5)
    assert result["chart_type"] == "xbar"
    assert "chart" in result
    assert "summary" in result
    assert "points" in result["chart"]
    assert "center" in result["chart"]
    assert "ucl" in result["chart"]
    assert "lcl" in result["chart"]


def test_r(spc_data):
    result = control_chart("r", spc_data, subgroup_size=5)
    assert result["chart_type"] == "r"
    assert "chart" in result


def test_imr(spc_data):
    result = control_chart("imr", spc_data)
    assert result["chart_type"] == "imr"
    assert "chart" in result
    assert "mr_chart" in result


def test_p():
    defects = [2, 3, 1, 4, 2, 3, 1, 2, 3, 2]
    result = control_chart("p", defects, sample_size=100)
    assert result["chart_type"] == "p"


def test_np():
    defects = [2, 3, 1, 4, 2, 3, 1, 2, 3, 2]
    result = control_chart("np", defects, sample_size=100)
    assert result["chart_type"] == "np"


def test_c():
    defects = [2, 3, 1, 4, 2, 3, 1, 2, 3, 2]
    result = control_chart("c", defects)
    assert result["chart_type"] == "c"


def test_u():
    defects = [2, 3, 1, 4, 2, 3, 1, 2, 3, 2]
    result = control_chart("u", defects, sample_size=100)
    assert result["chart_type"] == "u"


def test_ewma(spc_data):
    result = control_chart("ewma", spc_data, target=100, lambda_=0.2)
    assert result["chart_type"] == "ewma"
    assert "chart" in result
    assert "lambda" in result["chart"]
    assert "target" in result["chart"]
    # EWMA should have array UCL/LCL
    assert isinstance(result["chart"]["ucl"], list)
    assert isinstance(result["chart"]["lcl"], list)


def test_cusum(spc_data):
    result = control_chart("cusum", spc_data, target=100, k=0.5, h=5)
    assert result["chart_type"] == "cusum"
    assert "chart" in result
    assert "cusum_pos" in result["chart"]
    assert "cusum_neg" in result["chart"]


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown chart type"):
        control_chart("invalid", [1, 2, 3])


def test_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        control_chart("imr", [1.0])


def test_xbar_too_few_subgroups():
    with pytest.raises(ValueError):
        control_chart("xbar", [1, 2, 3, 4, 5], subgroup_size=5)


def test_out_of_control_detection():
    # Create data with clear out-of-control point
    values = [100] * 20 + [150] + [100] * 19
    result = control_chart("imr", values)
    assert len(result["chart"]["out_of_control_points"]) > 0

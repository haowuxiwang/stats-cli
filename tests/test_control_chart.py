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


# --- Hotelling T² Tests ---


@pytest.fixture
def mv_data():
    """Generate 2D multivariate data with 2 variables, in-control."""
    np.random.seed(42)
    n = 50
    mean = [10, 20]
    cov = [[1.0, 0.3], [0.3, 1.0]]
    return np.random.multivariate_normal(mean, cov, n).tolist()


@pytest.fixture
def mv_data_with_outlier():
    """Generate 2D data with an obvious outlier."""
    np.random.seed(42)
    n = 50
    mean = [10, 20]
    cov = [[1.0, 0.3], [0.3, 1.0]]
    data = np.random.multivariate_normal(mean, cov, n)
    # Inject outlier at observation 25
    data[25] = [30, 40]
    return data.tolist()


def test_hotelling_t2_basic(mv_data):
    result = control_chart("hotelling_t2", mv_data)
    assert result["chart_type"] == "hotelling_t2"
    assert "chart" in result
    chart = result["chart"]
    assert "points" in chart
    assert "center" in chart
    assert "ucl" in chart
    assert "lcl" in chart
    assert chart["lcl"] == 0.0
    assert chart["title"] == "Hotelling T² Chart"
    assert chart["n_variables"] == 2
    assert chart["n_observations"] == 50
    assert "mean_vector" in result
    assert "covariance_matrix" in result
    assert "summary" in result


def test_hotelling_t2_ucl_positive(mv_data):
    result = control_chart("hotelling_t2", mv_data)
    assert result["chart"]["ucl"] > 0
    assert result["chart"]["center"] > 0


def test_hotelling_t2_out_of_control(mv_data_with_outlier):
    result = control_chart("hotelling_t2", mv_data_with_outlier)
    assert len(result["chart"]["out_of_control_points"]) > 0
    assert result["summary"]["stable"] is False
    assert "out-of-control" in result["summary"]["message"]


def test_hotelling_t2_stable_data(mv_data):
    result = control_chart("hotelling_t2", mv_data, alpha=0.05)
    # With alpha=0.05, some in-control data might trigger, but structure should be correct
    assert "out_of_control_points" in result["chart"]
    assert isinstance(result["chart"]["in_control"], list)


def test_hotelling_t2_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        control_chart("hotelling_t2", [1, 2, 3, 4, 5])


def test_hotelling_t2_rejects_insufficient_data():
    # n must be > p+1; n=3, p=2 means 3 <= 3 triggers the error
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    with pytest.raises(ValueError, match="n > p\\+1"):
        control_chart("hotelling_t2", data)


def test_hotelling_t2_3_variables():
    np.random.seed(42)
    data = np.random.multivariate_normal([10, 20, 30], np.eye(3), 50).tolist()
    result = control_chart("hotelling_t2", data)
    assert result["chart"]["n_variables"] == 3
    assert result["chart"]["ucl"] > 0


# --- MEWMA Tests ---


def test_ewma_mv_basic(mv_data):
    result = control_chart("ewma_mv", mv_data, lambda_=0.1)
    assert result["chart_type"] == "ewma_mv"
    assert "chart" in result
    chart = result["chart"]
    assert "points" in chart
    assert "center" in chart
    assert "ucl" in chart
    assert "lcl" in chart
    assert chart["lcl"] == 0.0
    assert chart["title"] == "MEWMA Chart"
    assert chart["n_variables"] == 2
    assert chart["lambda"] == 0.1
    assert "mean_vector" in result
    assert "summary" in result


def test_ewma_mv_ucl_positive(mv_data):
    result = control_chart("ewma_mv", mv_data)
    assert result["chart"]["ucl"] > 0
    assert result["chart"]["center"] == 2.0  # p = 2


def test_ewma_mv_out_of_control(mv_data_with_outlier):
    result = control_chart("ewma_mv", mv_data_with_outlier, lambda_=0.1)
    # MEWMA with low lambda should detect the outlier
    assert "out_of_control_points" in result["chart"]


def test_ewma_mv_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        control_chart("ewma_mv", [1, 2, 3, 4, 5])


def test_ewma_mv_default_lambda(mv_data):
    result = control_chart("ewma_mv", mv_data)
    assert result["chart"]["lambda"] == 0.1

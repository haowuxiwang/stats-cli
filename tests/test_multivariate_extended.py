"""Tests for Factor Analysis and MANOVA in stats_engine/multivariate.py."""

import json

import numpy as np
import pytest

from stats_engine.multivariate import multivariate


@pytest.fixture
def factor_data():
    """Data with known factor structure: 2 underlying factors, 6 variables."""
    np.random.seed(42)
    n = 100
    # Two latent factors
    f1 = np.random.normal(0, 1, n)
    f2 = np.random.normal(0, 1, n)
    # Six observed variables: v1-v3 load on f1, v4-v6 load on f2
    v1 = 0.8 * f1 + 0.1 * f2 + np.random.normal(0, 0.3, n)
    v2 = 0.7 * f1 + 0.2 * f2 + np.random.normal(0, 0.4, n)
    v3 = 0.6 * f1 + 0.1 * f2 + np.random.normal(0, 0.5, n)
    v4 = 0.1 * f1 + 0.8 * f2 + np.random.normal(0, 0.3, n)
    v5 = 0.2 * f1 + 0.7 * f2 + np.random.normal(0, 0.4, n)
    v6 = 0.1 * f1 + 0.6 * f2 + np.random.normal(0, 0.5, n)
    return np.column_stack([v1, v2, v3, v4, v5, v6])


@pytest.fixture
def manova_data_diff():
    """Groups that should differ significantly on multiple DVs."""
    np.random.seed(42)
    g1 = np.column_stack(
        [
            np.random.normal(5, 1, 30),
            np.random.normal(10, 1, 30),
        ]
    )
    g2 = np.column_stack(
        [
            np.random.normal(8, 1, 30),
            np.random.normal(13, 1, 30),
        ]
    )
    g3 = np.column_stack(
        [
            np.random.normal(11, 1, 30),
            np.random.normal(16, 1, 30),
        ]
    )
    return [g1, g2, g3]


@pytest.fixture
def manova_data_same():
    """Groups that are essentially the same (no real difference)."""
    np.random.seed(42)
    g1 = np.column_stack(
        [
            np.random.normal(10, 1, 50),
            np.random.normal(20, 1, 50),
        ]
    )
    g2 = np.column_stack(
        [
            np.random.normal(10, 1, 50),
            np.random.normal(20, 1, 50),
        ]
    )
    return [g1, g2]


# ---- Factor Analysis Tests ----


def test_factor_analysis_basic(factor_data):
    """Factor analysis with known factor structure."""
    columns = ["v1", "v2", "v3", "v4", "v5", "v6"]
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        columns=columns,
        n_factors=2,
        rotation="varimax",
    )
    assert result["analysis_type"] == "factor_analysis"
    assert result["n_factors"] == 2
    assert result["n_variables"] == 6
    assert result["n_observations"] == 100
    assert result["rotation"] == "varimax"
    assert "loadings" in result
    assert "communalities" in result
    assert "uniquenesses" in result
    assert "variance_explained" in result
    assert "eigenvalues" in result
    assert "interpretation" in result


def test_factor_analysis_loadings_shape(factor_data):
    """Loadings matrix should have correct shape."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
    )
    assert len(result["loadings"]) == 6
    for _var_name, loads in result["loadings"].items():
        assert len(loads) == 2


def test_factor_analysis_communalities_range(factor_data):
    """Communalities should be between 0 and 1."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
    )
    for var_name, comm in result["communalities"].items():
        assert 0 <= comm <= 1.0 + 1e-6, f"Communality for {var_name} = {comm} out of range"


def test_factor_analysis_uniquenesses_sum_to_one(factor_data):
    """Communality + uniqueness should equal 1 for each variable."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
    )
    for var_name in result["communalities"]:
        total = result["communalities"][var_name] + result["uniquenesses"][var_name]
        assert abs(total - 1.0) < 1e-4, f"{var_name}: communality + uniqueness = {total}"


def test_factor_analysis_varimax_orthogonal(factor_data):
    """Varimax rotation should produce roughly orthogonal factors."""
    columns = ["v1", "v2", "v3", "v4", "v5", "v6"]
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        columns=columns,
        n_factors=2,
        rotation="varimax",
    )
    # Check that loadings are orthogonal by computing correlation between factor scores
    # For varimax, the loading columns should have low inner product
    loadings_matrix = np.array([result["loadings"][f"v{i + 1}"] for i in range(6)])
    # Compute angle between loading columns
    col_norms = np.linalg.norm(loadings_matrix, axis=0)
    if col_norms[0] > 0 and col_norms[1] > 0:
        cos_angle = np.dot(loadings_matrix[:, 0], loadings_matrix[:, 1]) / (col_norms[0] * col_norms[1])
        # For orthogonal rotation, columns should be roughly perpendicular
        assert abs(cos_angle) < 0.3, f"Varimax columns not orthogonal: cos(angle) = {cos_angle}"


def test_factor_analysis_promax(factor_data):
    """Promax rotation should work and return factor correlation."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
        rotation="promax",
    )
    assert result["rotation"] == "promax"
    # Promax is oblique, so factor_correlation should be present
    assert "factor_correlation" in result


def test_factor_analysis_no_rotation(factor_data):
    """No rotation should work."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
        rotation="none",
    )
    assert result["rotation"] == "none"


def test_factor_analysis_default_n_factors(factor_data):
    """Default n_factors should use Kaiser criterion (eigenvalue > 1)."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
    )
    assert result["n_factors"] >= 1
    # With our 2-factor data, should find 2 factors
    assert result["n_factors"] == 2


def test_factor_analysis_unknown_rotation(factor_data):
    """Unknown rotation should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown rotation"):
        multivariate(
            analysis_type="factor_analysis",
            values=factor_data.tolist(),
            n_factors=2,
            rotation="invalid",
        )


def test_factor_analysis_no_values():
    """Factor analysis without data should raise ValueError."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="factor_analysis")


def test_factor_analysis_single_variable():
    """Factor analysis with 1 variable should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        multivariate(
            analysis_type="factor_analysis",
            values=[[1], [2], [3]],
        )


def test_factor_analysis_json_serializable(factor_data):
    """Factor analysis output should be JSON serializable."""
    result = multivariate(
        analysis_type="factor_analysis",
        values=factor_data.tolist(),
        n_factors=2,
    )
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["analysis_type"] == "factor_analysis"


# ---- MANOVA Tests ----


def test_manova_diff_groups(manova_data_diff):
    """MANOVA with clearly different groups should be significant."""
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in manova_data_diff],
    )
    assert result["analysis_type"] == "manova"
    assert result["n_groups"] == 3
    assert result["n_variables"] == 2
    assert result["n_total"] == 90
    assert "statistics" in result
    assert "f_values" in result
    assert "p_values" in result
    assert "df" in result
    assert "interpretation" in result
    # Should be significant
    assert result["p_values"]["wilks_lambda"] < 0.05


def test_manova_same_groups(manova_data_same):
    """MANOVA with identical groups should NOT be significant."""
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in manova_data_same],
    )
    assert result["n_groups"] == 2
    # With identical distributions, p should be large
    assert result["p_values"]["wilks_lambda"] > 0.05


def test_manova_statistics_range(manova_data_diff):
    """All four MANOVA statistics should have valid ranges."""
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in manova_data_diff],
    )
    stats = result["statistics"]
    # Wilks' Lambda: 0 to 1
    assert 0 <= stats["wilks_lambda"] <= 1
    # Pillai's Trace: 0 to min(p, k-1)
    assert stats["pillai_trace"] >= 0
    # Hotelling-Lawley: >= 0
    assert stats["hotelling_trace"] >= 0
    # Roy's Root: >= 0
    assert stats["roys_root"] >= 0


def test_manova_p_values_range(manova_data_diff):
    """P-values should be between 0 and 1."""
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in manova_data_diff],
    )
    for test_name, p_val in result["p_values"].items():
        if p_val is not None:
            assert 0 <= p_val <= 1, f"p_value for {test_name} = {p_val}"


def test_manova_json_serializable(manova_data_diff):
    """MANOVA output should be JSON serializable."""
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in manova_data_diff],
    )
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["analysis_type"] == "manova"


def test_manova_no_groups():
    """MANOVA without groups should raise ValueError."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="manova")


def test_manova_single_group():
    """MANOVA with 1 group should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        multivariate(
            analysis_type="manova",
            groups=[[[1, 2], [3, 4]]],
        )


def test_manova_mismatched_variables():
    """MANOVA with different variable counts should raise ValueError."""
    with pytest.raises(ValueError, match="same number"):
        multivariate(
            analysis_type="manova",
            groups=[[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]],
        )


def test_manova_too_few_observations():
    """MANOVA group with 1 observation should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        multivariate(
            analysis_type="manova",
            groups=[[[1, 2]], [[3, 4], [5, 6]]],
        )


def test_manova_three_dvs():
    """MANOVA with 3 dependent variables."""
    np.random.seed(42)
    g1 = np.column_stack(
        [
            np.random.normal(5, 1, 25),
            np.random.normal(10, 1, 25),
            np.random.normal(15, 1, 25),
        ]
    )
    g2 = np.column_stack(
        [
            np.random.normal(8, 1, 25),
            np.random.normal(13, 1, 25),
            np.random.normal(18, 1, 25),
        ]
    )
    result = multivariate(
        analysis_type="manova",
        groups=[g1.tolist(), g2.tolist()],
    )
    assert result["n_variables"] == 3
    assert result["n_groups"] == 2


def test_unknown_analysis_type():
    """Unknown analysis type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        multivariate(analysis_type="nonexistent")


# ---- Coverage: n_factors validation (line 376) ----


def test_factor_analysis_n_factors_too_large(factor_data):
    """n_factors > n_vars should raise ValueError."""
    with pytest.raises(ValueError, match="n_factors must be between"):
        multivariate(
            analysis_type="factor_analysis",
            values=factor_data.tolist(),
            n_factors=100,
        )


def test_factor_analysis_n_factors_zero(factor_data):
    """n_factors < 1 should raise ValueError."""
    with pytest.raises(ValueError, match="n_factors must be between"):
        multivariate(
            analysis_type="factor_analysis",
            values=factor_data.tolist(),
            n_factors=0,
        )


# ---- Coverage: Kaiser 0 fallback (line 373) ----


def test_factor_analysis_kaiser_zero_fallback(monkeypatch):
    """When all eigenvalues <= 1, Kaiser criterion gives 0 and falls back to 1."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (50, 3))

    original_eigh = np.linalg.eigh

    def mock_eigh(a):
        """Return eigenvalues all <= 1 to trigger Kaiser 0 fallback."""
        vals = np.array([0.5, 0.3, 0.2])
        vecs = np.eye(3)
        return vals, vecs

    monkeypatch.setattr(np.linalg, "eigh", mock_eigh)
    result = multivariate(analysis_type="factor_analysis", values=X.tolist())
    assert result["n_factors"] == 1

    # Restore for other tests
    monkeypatch.setattr(np.linalg, "eigh", original_eigh)


# ---- Coverage: factor_analysis file path (lines 341-344) ----


def test_factor_analysis_file_path(tmp_path):
    """Factor analysis with file input should use the file loading branch."""
    import pandas as pd

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x1": np.random.normal(0, 1, 50),
            "x2": np.random.normal(0, 1, 50),
            "x3": np.random.normal(0, 1, 50),
        }
    )
    csv_file = tmp_path / "fa_data.csv"
    df.to_csv(csv_file, index=False)

    result = multivariate(
        analysis_type="factor_analysis",
        file=str(csv_file),
        columns=["x1", "x2", "x3"],
        n_factors=2,
    )
    assert result["analysis_type"] == "factor_analysis"
    assert result["n_variables"] == 3
    assert result["n_observations"] == 50


# ---- Coverage: MANOVA file path (lines 443-450) ----


def test_manova_file_path(tmp_path):
    """MANOVA with file input should use the file loading branch."""
    import pandas as pd

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "dv1": np.concatenate([np.random.normal(5, 1, 20), np.random.normal(8, 1, 20)]),
            "dv2": np.concatenate([np.random.normal(10, 1, 20), np.random.normal(13, 1, 20)]),
            "group": ["A"] * 20 + ["B"] * 20,
        }
    )
    csv_file = tmp_path / "manova_data.csv"
    df.to_csv(csv_file, index=False)

    result = multivariate(
        analysis_type="manova",
        file=str(csv_file),
        columns=["dv1", "dv2"],
        group_column="group",
    )
    assert result["analysis_type"] == "manova"
    assert result["n_groups"] == 2
    assert result["n_total"] == 40


# ---- Coverage: MANOVA 1D group reshape (line 464) ----


def test_manova_1d_group_triggers_reshape():
    """Passing a 1D group after a 2D group triggers reshape then dimension mismatch."""
    # First group is 2D (2 obs, 3 DVs), second group is 1D
    with pytest.raises(ValueError, match="same number|at least 2"):
        multivariate(
            analysis_type="manova",
            groups=[[[1, 2, 3], [4, 5, 6]], [7, 8, 9]],
        )


# ---- Coverage: MANOVA singular W matrix (lines 505-510) + p_hl=None (568) + p_roy=None (577) ----


def test_manova_singular_w_and_none_pvalues():
    """Tiny groups with more DVs than df_error triggers singular W and None p-values."""
    np.random.seed(42)
    # 2 groups, 2 obs each, 3 DVs → W is singular (rank <= 2 < 3)
    # df_error = 4 - 2 = 2, df2_hl = 1*(2-3-1) = -2, df2_roy = 2-3-1 = -2
    g1 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    g2 = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    result = multivariate(analysis_type="manova", groups=[g1.tolist(), g2.tolist()])
    assert result["n_groups"] == 2
    assert result["n_variables"] == 3
    # With singular W, code falls back to pseudo-inverse
    assert result["p_values"]["hotelling_trace"] is None
    assert result["p_values"]["roys_root"] is None


# ---- Coverage: MANOVA s >= 3 chi2 approximation (lines 541-546) ----


def test_manova_s_geq_3_chi2_approx():
    """4 groups with 3 DVs gives s >= 3, triggering Bartlett's chi2 approximation."""
    np.random.seed(42)
    g1 = np.column_stack([np.random.normal(5, 1, 15), np.random.normal(10, 1, 15), np.random.normal(15, 1, 15)])
    g2 = np.column_stack([np.random.normal(8, 1, 15), np.random.normal(13, 1, 15), np.random.normal(18, 1, 15)])
    g3 = np.column_stack([np.random.normal(11, 1, 15), np.random.normal(16, 1, 15), np.random.normal(21, 1, 15)])
    g4 = np.column_stack([np.random.normal(14, 1, 15), np.random.normal(19, 1, 15), np.random.normal(24, 1, 15)])
    result = multivariate(
        analysis_type="manova",
        groups=[g.tolist() for g in [g1, g2, g3, g4]],
    )
    assert result["n_groups"] == 4
    assert result["n_variables"] == 3
    # In the s>=3 branch, wilks f_values is None and chi2_df is used
    assert result["f_values"]["wilks_lambda"] is None
    assert "chi2_df" in result["df"]["wilks_lambda"]
    assert result["p_values"]["wilks_lambda"] is not None


# ---- Coverage: promax LinAlgError fallback (lines 310-311) ----


def test_factor_analysis_promax_singular_fallback(monkeypatch):
    """Promax with singular LtL falls back to varimax + identity correlation."""
    from stats_engine.multivariate import _promax_loadings

    original_solve = np.linalg.solve

    def mock_solve(a, b):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "solve", mock_solve)

    # Create a simple loading matrix
    loadings = np.array([[0.8, 0.1], [0.7, 0.2], [0.1, 0.8], [0.2, 0.7]])
    rotated, factor_corr = _promax_loadings(loadings)
    # On singular LtL, should return varimax loadings + identity
    assert rotated.shape == loadings.shape
    assert factor_corr.shape == (2, 2)
    np.testing.assert_array_almost_equal(factor_corr, np.eye(2))

    monkeypatch.setattr(np.linalg, "solve", original_solve)

"""Tests for stats_engine/anova.py."""

import pytest

from stats_engine.anova import anova


def test_one_way(three_groups):
    result = anova(anova_type="one_way", groups=three_groups)
    assert result["anova_type"] == "one_way"
    assert "f_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert "eta_squared" in result
    assert "interpretation" in result


def test_two_way():
    import pandas as pd

    df = pd.DataFrame(
        {
            "values": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "factor_a": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
            "factor_b": ["X", "X", "X", "Y", "Y", "Y", "X", "X", "X", "Y", "Y", "Y"],
        }
    )
    result = anova(anova_type="two_way", groups=None, data=df)
    assert result["anova_type"] == "two_way"
    assert "sources" in result


def test_one_way_too_few_groups():
    with pytest.raises((TypeError, ValueError)):
        anova(anova_type="one_way", groups=[[1, 2, 3]])


def test_one_way_significant():
    g1 = [1, 2, 3, 4, 5]
    g2 = [10, 11, 12, 13, 14]
    g3 = [20, 21, 22, 23, 24]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert bool(result["significant"]) is True


def test_one_way_not_significant():
    """Overlapping groups should not be significant."""
    g1 = [10, 11, 12, 13, 14]
    g2 = [10.5, 11.5, 12.5, 13.5, 14.5]
    g3 = [11, 12, 13, 14, 15]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert bool(result["significant"]) is False


def test_one_way_unequal_groups():
    """Groups of different sizes should work correctly."""
    import numpy as np

    np.random.seed(99)
    g1 = np.random.normal(10, 1, 5).tolist()
    g2 = np.random.normal(10, 1, 10).tolist()
    g3 = np.random.normal(10, 1, 15).tolist()
    g4 = np.random.normal(10, 1, 20).tolist()
    result = anova(anova_type="one_way", groups=[g1, g2, g3, g4])
    assert result["n_groups"] == 4
    assert result["n_total"] == 50
    assert "f_statistic" in result
    assert "p_value" in result


def test_one_way_eta_squared_value():
    """eta_squared must be in [0, 1]."""
    g1 = [1, 2, 3, 4, 5]
    g2 = [10, 11, 12, 13, 14]
    g3 = [20, 21, 22, 23, 24]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert 0 <= result["eta_squared"] <= 1


def test_one_way_group_stats():
    """Verify per-group n, mean, std are correct."""
    g1 = [10, 20, 30]
    g2 = [40, 50, 60]
    result = anova(anova_type="one_way", groups=[g1, g2])
    stats = result["group_stats"]
    assert len(stats) == 2
    assert stats[0]["n"] == 3
    assert stats[0]["mean"] == 20.0
    assert stats[1]["n"] == 3
    assert stats[1]["mean"] == 50.0
    assert stats[0]["std"] > 0
    assert stats[1]["std"] > 0


def test_one_way_identical_values():
    """All identical values should yield eta_squared = 0."""
    g1 = [5, 5, 5, 5]
    g2 = [5, 5, 5, 5]
    g3 = [5, 5, 5, 5]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert result["eta_squared"] == 0


def test_two_way_interaction():
    """Two-way ANOVA should include interaction term in sources."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "values": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "factor_a": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
            "factor_b": ["X", "X", "X", "Y", "Y", "Y", "X", "X", "X", "Y", "Y", "Y"],
        }
    )
    result = anova(anova_type="two_way", groups=None, data=df)
    source_names = [s["source"] for s in result["sources"]]
    assert any(":" in name for name in source_names), "Interaction term not found"


def test_two_way_r_squared():
    """r_squared should be in [0, 1]."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "values": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "factor_a": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
            "factor_b": ["X", "X", "X", "Y", "Y", "Y", "X", "X", "X", "Y", "Y", "Y"],
        }
    )
    result = anova(anova_type="two_way", groups=None, data=df)
    assert 0 <= result["r_squared"] <= 1


def test_invalid_anova_type():
    """Unknown anova_type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown anova_type"):
        anova(anova_type="invalid", groups=[[1, 2], [3, 4]])


def test_one_way_anova_table():
    """Verify df_between, df_within, ss_between, ss_within are returned."""
    g1 = [1, 2, 3, 4, 5]
    g2 = [10, 11, 12, 13, 14]
    g3 = [20, 21, 22, 23, 24]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert result["df_between"] == 2  # k - 1 = 3 - 1
    assert result["df_within"] == 12  # n_total - k = 15 - 3
    assert "ss_between" in result
    assert "ss_within" in result
    assert result["ss_between"] > 0
    assert result["ss_within"] > 0


def test_one_way_omega_squared():
    """Verify omega_squared is returned and in [0, 1]."""
    g1 = [1, 2, 3, 4, 5]
    g2 = [10, 11, 12, 13, 14]
    g3 = [20, 21, 22, 23, 24]
    result = anova(anova_type="one_way", groups=[g1, g2, g3])
    assert "omega_squared" in result
    assert 0 <= result["omega_squared"] <= 1

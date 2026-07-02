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


class TestRepeatedMeasuresANOVA:
    """Tests for repeated measures ANOVA."""

    def test_repeated_one_way(self):
        """One-way repeated measures ANOVA."""
        import pandas as pd

        # 6 subjects, 3 timepoints
        df = pd.DataFrame(
            {
                "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                "time": ["T0", "T1", "T2"] * 6,
                "value": [10, 12, 14, 11, 13, 15, 10, 11, 13, 12, 14, 16, 11, 12, 14, 10, 13, 15],
            }
        )
        result = anova(anova_type="repeated", data=df, subject="subject", within="time")
        assert result["anova_type"] == "repeated"
        assert result["n_subjects"] == 6
        assert "sources" in result
        assert "time" in [s["source"] for s in result["sources"]]

    def test_repeated_significant_effect(self):
        """Repeated measures detects significant time effect."""
        import pandas as pd

        # Clear increasing trend across timepoints
        subjects = []
        times = []
        values = []
        for s in range(8):
            subjects.extend([s] * 3)
            times.extend(["T0", "T1", "T2"])
            values.extend([10 + s, 15 + s, 20 + s])  # Strong time effect (+5 per step)

        df = pd.DataFrame({"subject": subjects, "time": times, "value": values})
        result = anova(anova_type="repeated", data=df, subject="subject", within="time")
        time_effect = [s for s in result["sources"] if s["source"] == "time"]
        assert len(time_effect) == 1
        assert time_effect[0]["significant"] is True

    def test_repeated_from_dict_data(self):
        """Repeated measures accepts dict data."""
        data = {
            "subject": [1, 1, 2, 2, 3, 3],
            "condition": ["A", "B", "A", "B", "A", "B"],
            "value": [10, 14, 11, 15, 12, 16],  # default response_col="value"
        }
        result = anova(anova_type="repeated", groups={"data": data}, subject="subject", within="condition")
        assert result["anova_type"] == "repeated"

    def test_repeated_missing_subject_raises(self):
        """Missing subject parameter raises ValueError."""
        import pandas as pd

        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(ValueError, match="subject"):
            anova(anova_type="repeated", data=df, within="a")

    def test_repeated_via_handler(self):
        """Repeated measures works through handler()."""
        import pandas as pd

        from main import handler

        df = pd.DataFrame(
            {
                "subject": [s for s in range(6) for _ in range(3)],
                "time": ["T0", "T1", "T2"] * 6,
                "score": [10, 12, 14, 11, 13, 15, 10, 11, 13, 12, 14, 16, 11, 12, 14, 10, 13, 15],
            }
        )
        result = handler(
            {
                "command": "anova",
                "params": {
                    "anova_type": "repeated",
                    "data": df.to_dict("list"),
                    "subject": "subject",
                    "within": "time",
                    "response_col": "score",
                },
            }
        )
        assert result["status"] == "success"
        assert result["data"]["anova_type"] == "repeated"


class TestNWayANOVA:
    """Tests for N-way ANOVA (3+ factors)."""

    def test_n_way_3_factors(self):
        """3-factor ANOVA with interactions."""
        # 2x2x2 design with 2 replicates = 16 observations
        factors = {
            "A": [1, 1, 1, 1, 2, 2, 2, 2] * 2,
            "B": [1, 1, 2, 2, 1, 1, 2, 2] * 2,
            "C": [1, 2, 1, 2, 1, 2, 1, 2] * 2,
        }
        response = [10, 12, 14, 16, 11, 13, 15, 17, 12, 14, 16, 18, 13, 15, 17, 19]
        result = anova(anova_type="n_way", groups={"factors": factors, "response": response})
        assert result["anova_type"] == "n_way"
        assert result["n_factors"] == 3
        assert "A" in result["factor_names"]
        assert "B" in result["factor_names"]
        assert "C" in result["factor_names"]
        assert "sources" in result
        assert len(result["sources"]) > 0  # main effects + interactions
        assert "r_squared" in result
        assert "formula" in result
        assert "interpretation" in result

    def test_n_way_significant_main_effect(self):
        """N-way ANOVA detects significant main effect."""
        # Larger design: 2x2x2 with 4 replicates = 32 observations
        factors = {
            "Treatment": [],
            "Time": [],
            "Gender": [],
        }
        response = []
        for treat in ["Control", "Drug"]:
            for time in ["T0", "T1"]:
                for gender in ["M", "F"]:
                    for rep in range(4):
                        factors["Treatment"].append(treat)
                        factors["Time"].append(time)
                        factors["Gender"].append(gender)
                        base = 10 + (treat == "Drug") * 10 + (time == "T1") * 2
                        response.append(base + rep)
        result = anova(anova_type="n_way", groups={"factors": factors, "response": response})
        assert result["anova_type"] == "n_way"
        # Treatment should be significant (large effect: +10)
        treatment_effect = [s for s in result["sources"] if "Treatment" in s["source"]]
        assert len(treatment_effect) >= 1
        assert any(s.get("significant") for s in treatment_effect)

    def test_n_way_interaction_effect(self):
        """N-way ANOVA detects interaction effect."""
        factors = {
            "A": [1, 1, 2, 2] * 4,
            "B": [1, 2, 1, 2] * 4,
            "C": [1, 1, 1, 1, 2, 2, 2, 2] * 2,
        }
        # Strong A:B interaction
        response = [10, 20, 30, 10, 11, 21, 31, 11, 12, 22, 32, 12, 13, 23, 33, 13]
        result = anova(anova_type="n_way", groups={"factors": factors, "response": response})
        assert result["anova_type"] == "n_way"
        # Check interaction terms exist
        interaction_sources = [s for s in result["sources"] if ":" in s["source"]]
        assert len(interaction_sources) > 0

    def test_n_way_too_many_factors(self):
        """More than 5 factors raises error."""
        factors = {f"F{i}": [1, 2, 3, 4] for i in range(6)}
        response = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Maximum 5 factors"):
            anova(anova_type="n_way", groups={"factors": factors, "response": response})

    def test_n_way_from_dataframe(self):
        """N-way ANOVA accepts DataFrame."""
        import pandas as pd

        # Larger design to avoid over-parameterization
        rows = []
        for a in [1, 2]:
            for b in [1, 2]:
                for c in [1, 2]:
                    for rep in range(4):
                        rows.append(
                            {
                                "A": a,
                                "B": b,
                                "C": c,
                                "response": 10 + a * 2 + b * 3 + c + rep,
                            }
                        )
        df = pd.DataFrame(rows)
        result = anova(anova_type="n_way", groups=None, data=df.rename(columns={"response": "__response__"}))
        assert result["anova_type"] == "n_way"
        assert result["n_factors"] == 3

    def test_n_way_via_handler(self):
        """N-way ANOVA works through handler()."""
        from main import handler

        factors = {
            "A": [1, 1, 2, 2] * 8,
            "B": [1, 2, 1, 2] * 8,
            "C": [1, 1, 1, 1, 2, 2, 2, 2] * 4,
        }
        response = [
            10 + a * 2 + b * 3 + c + r
            for r, (a, b, c) in enumerate(list(zip([1, 1, 2, 2] * 8, [1, 2, 1, 2] * 8, [1, 1, 1, 1, 2, 2, 2, 2] * 4)))
        ]
        result = handler(
            {
                "command": "anova",
                "params": {"anova_type": "n_way", "factors": factors, "response": response},
            }
        )
        assert result["status"] == "success"
        assert result["data"]["anova_type"] == "n_way"

    def test_n_way_single_factor_raises(self):
        """N-way with < 2 factors raises error."""
        with pytest.raises(ValueError, match="at least 2 factors"):
            anova(anova_type="n_way", groups={"factors": {"A": [1, 2, 3, 4]}, "response": [10, 12, 14, 16]})

    def test_n_way_missing_response_raises(self):
        """N-way with missing response raises error."""
        with pytest.raises((ValueError, TypeError)):
            anova(anova_type="n_way", groups={"factors": {"A": [1, 2], "B": [1, 2]}})

    def test_repeated_sphericity_note(self):
        """Repeated measures includes sphericity note."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "subject": [s for s in range(8) for _ in range(3)],
                "time": ["T0", "T1", "T2"] * 8,
                "value": [10, 12, 14] * 8,
            }
        )
        result = anova(anova_type="repeated", data=df, subject="subject", within="time")
        assert "sphericity_note" in result

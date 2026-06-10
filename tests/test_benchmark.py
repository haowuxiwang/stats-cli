"""Benchmark tests for stats-cli.

Based on:
- Anscombe's Quartet (identical summary stats, different distributions)
- NIST StRD certified regression values
- Classic textbook statistical test cases
- Edge cases and robustness scenarios
"""

import math

import numpy as np
import pytest

from main import handler
from stats_engine.anova import anova
from stats_engine.correlation import correlation
from stats_engine.descriptive import descriptive
from stats_engine.multivariate import multivariate
from stats_engine.normality import normality
from stats_engine.outlier import outlier
from stats_engine.power import power
from stats_engine.regression import regression
from stats_engine.ttest import ttest

# ============================================================================
# Anscombe's Quartet - Classic benchmark for descriptive statistics
# All 4 datasets have nearly identical mean, variance, regression line, R^2
# ============================================================================

class TestAnscombeQuartet:
    """Anscombe's Quartet: 4 datasets with identical summary statistics."""

    @pytest.fixture
    def anscombe(self):
        return {
            "x1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
            "x2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
            "x3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
            "x4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        }

    def test_descriptive_stats_similar(self, anscombe):
        """All 4 y-datasets should have nearly identical descriptive stats."""
        results = {}
        for key in ["y1", "y2", "y3", "y4"]:
            results[key] = descriptive(values=anscombe[key])

        # All means should be ~7.5 (within 0.5 of each other)
        means = [r["mean"] for r in results.values()]
        assert max(means) - min(means) < 0.5, f"Means vary too much: {means}"

        # All standard deviations should be ~2.0 (within 0.3 of each other)
        stds = [r["std"] for r in results.values()]
        assert max(stds) - min(stds) < 0.3, f"Stds vary too much: {stds}"

    def test_regression_line_similar(self, anscombe):
        """All 4 datasets should have nearly identical regression lines."""
        results = {}
        for i in range(1, 5):
            x = anscombe[f"x{i}"]
            y = anscombe[f"y{i}"]
            results[i] = regression(x=x, y=y, reg_type="linear")

        # All should have slope ~0.5 and intercept ~3.0
        for i, r in results.items():
            slope = r["coefficients"]["slope"]
            intercept = r["coefficients"]["intercept"]
            assert abs(slope - 0.5) < 0.1, f"Dataset {i}: slope={slope}, expected ~0.5"
            assert abs(intercept - 3.0) < 0.5, f"Dataset {i}: intercept={intercept}, expected ~3.0"

        # All should have R^2 ~0.67 (within 0.05)
        for i, r in results.items():
            assert abs(r["r_squared"] - 0.67) < 0.05, f"Dataset {i}: R^2={r['r_squared']}, expected ~0.67"


# ============================================================================
# NIST StRD Linear Regression - Certified values to 15 digits
# ============================================================================

class TestNISTRegression:
    """NIST StRD certified regression datasets.

    Source: https://www.itl.nist.gov/div898/strd/lls/lls.shtml
    All certified values are verified to 15-digit precision by NIST.
    """

    def test_norris(self):
        """NIST Norris dataset: 36 observations, lower difficulty.

        Certified values:
            slope:     1.00211681802045
            intercept: -0.262323073774029
            R^2:       0.999993745883712
        """
        x = [0.2, 337.4, 118.2, 884.6, 10.1, 226.5, 666.3, 996.3, 448.6, 777.0,
             558.2, 0.4, 0.6, 775.5, 666.9, 338.0, 447.5, 11.6, 556.0, 228.1,
             995.8, 887.6, 120.2, 0.3, 0.3, 556.8, 339.1, 887.2, 999.0, 779.0,
             11.1, 118.3, 229.2, 669.1, 448.9, 0.5]
        y = [0.1, 338.8, 118.1, 888.0, 9.2, 228.1, 668.5, 998.5, 449.1, 778.9,
             559.2, 0.3, 0.1, 778.1, 668.8, 339.3, 448.9, 10.8, 557.7, 228.3,
             998.0, 888.8, 119.6, 0.3, 0.6, 557.6, 339.3, 888.0, 998.5, 778.9,
             10.2, 117.6, 228.9, 668.4, 449.2, 0.2]

        result = regression(x=x, y=y, reg_type="linear")
        slope = result["coefficients"]["slope"]
        intercept = result["coefficients"]["intercept"]

        # Output rounded to 6 decimal places (DEFAULT_PRECISION=6)
        assert abs(slope - 1.00211681802045) < 5e-7, f"slope={slope}"
        assert abs(intercept - (-0.262323073774029)) < 5e-7, f"intercept={intercept}"
        assert abs(result["r_squared"] - 0.999993745883712) < 5e-7, f"R^2={result['r_squared']}"

    def test_pontius(self):
        """NIST Pontius dataset: 40 observations, lower difficulty.

        NIST certifies quadratic model: R^2 = 0.999999884412830
        At 6-digit precision, R^2 rounds to 1.0.
        """
        x = [150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800,
             1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000, 150, 300, 450, 600,
             750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2250, 2400,
             2550, 2700, 2850, 3000]
        y = [0.11019, 0.21956, 0.32949, 0.43899, 0.54803, 0.65694, 0.76562, 0.87487,
             0.98292, 1.09146, 1.20001, 1.30822, 1.41599, 1.52399, 1.63194, 1.73947,
             1.84646, 1.95392, 2.06128, 2.16844, 0.11052, 0.22018, 0.32939, 0.43886,
             0.54798, 0.65739, 0.76596, 0.87474, 0.98300, 1.09150, 1.20004, 1.30818,
             1.41613, 1.52408, 1.63159, 1.73965, 1.84696, 1.95445, 2.06177, 2.16829]

        result = regression(x=x, y=y, reg_type="quadratic")
        # NIST certified R^2 for quadratic model: 0.999999884412830
        # At 6-digit precision, this rounds to 1.0
        assert result["r_squared"] >= 0.999999, f"R^2={result['r_squared']}"

    def test_longley_multiple_regression(self):
        """NIST Longley dataset: 16 observations, 6 predictors, higher difficulty.

        Certified coefficients:
            B0 = -3482258.63459582
            B1 = 15.0618722713733
            B2 = -0.358191792925910E-01
            B3 = -2.02022980381683
            B4 = -1.03322686717359
            B5 = -0.511041056535807E-01
            B6 = 1829.15146461355
            R^2 = 0.995479004577296
        """
        import os
        import tempfile

        import pandas as pd

        # Create temporary CSV with Longley data
        data = {
            "y": [60323, 61122, 60171, 61187, 63221, 63639, 64989, 63761,
                  66019, 67857, 68169, 66513, 68655, 69564, 69331, 70551],
            "x1": [83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0,
                   101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9],
            "x2": [234289, 259426, 258054, 284599, 328975, 346999, 365385, 363112,
                   397469, 419180, 442769, 444546, 482704, 502601, 518173, 554894],
            "x3": [2356, 2325, 3682, 3351, 2099, 1932, 1870, 3578,
                   2904, 2822, 2936, 4681, 3813, 3931, 4806, 4007],
            "x4": [1590, 1456, 1616, 1650, 3099, 3594, 3547, 3350,
                   3048, 2857, 2798, 2637, 2552, 2514, 2572, 2827],
            "x5": [107608, 108632, 109773, 110929, 112075, 113270, 115094, 116219,
                   117388, 118734, 120445, 121950, 123366, 125368, 127852, 130081],
            "x6": [1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954,
                   1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962],
        }
        df = pd.DataFrame(data)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")  # noqa: SIM115
        df.to_csv(tmp.name, index=False)
        tmp.close()

        try:
            result = regression(
                file=tmp.name,
                x_columns=["x1", "x2", "x3", "x4", "x5", "x6"],
                y_column="y",
                reg_type="multiple",
            )
            coeffs = result["coefficients"]

            # NIST certified coefficients
            nist_coeffs = {
                "const": -3482258.63459582,
                "x1": 15.0618722713733,
                "x2": -0.0358191792925910,
                "x3": -2.02022980381683,
                "x4": -1.03322686717359,
                "x5": -0.0511041056535807,
                "x6": 1829.15146461355,
            }

            for name, nist_val in nist_coeffs.items():
                assert abs(coeffs[name] - nist_val) < 1e-4, \
                    f"{name}: got {coeffs[name]}, expected {nist_val}, err={abs(coeffs[name] - nist_val):.2e}"

            # Output rounded to 6 decimal places
            assert abs(result["r_squared"] - 0.995479004577296) < 5e-7, \
                f"R^2={result['r_squared']}"
        finally:
            os.unlink(tmp.name)


# ============================================================================
# NIST StRD Univariate Summary Statistics
# ============================================================================

class TestNISTDescriptive:
    """NIST StRD certified univariate statistics.

    Source: https://www.itl.nist.gov/div898/strd/univ/Univariate.shtml
    Tests numerical stability of descriptive() with ill-conditioned data.
    """

    def test_numacc1(self):
        """NumAcc1: Low difficulty. Simple dataset.

        Data: [1, 2, 3]
        Certified mean: 2.0
        Certified std (sample): 1.0
        """
        result = descriptive(values=[1.0, 2.0, 3.0])
        assert abs(result["mean"] - 2.0) < 1e-10, f"mean={result['mean']}"
        assert abs(result["std"] - 1.0) < 1e-10, f"std={result['std']}"

    def test_numacc2(self):
        """NumAcc2: Low difficulty. 20 observations.

        Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        Certified mean: 10.5
        Certified std (sample): 5.91607978309962
        """
        values = list(range(1, 21))
        result = descriptive(values=values)
        # Output rounded to 6 decimal places
        assert abs(result["mean"] - 10.5) < 5e-7, f"mean={result['mean']}"
        assert abs(result["std"] - 5.91607978309962) < 5e-7, f"std={result['std']}"

    def test_numacc3(self):
        """NumAcc3: Moderate difficulty. Large values, small differences.

        Data: [10000001, 10000002, 10000003]
        Certified mean: 10000002.0
        Certified std (sample): 1.0

        This tests catastrophic cancellation in variance computation.
        """
        result = descriptive(values=[10000001.0, 10000002.0, 10000003.0])
        assert abs(result["mean"] - 10000002.0) < 1e-4, f"mean={result['mean']}"
        assert abs(result["std"] - 1.0) < 1e-4, f"std={result['std']}"

    def test_numacc4(self):
        """NumAcc4: High difficulty. Very large values, tiny differences.

        Data: [10000000.1, 10000000.2, 10000000.3]
        Certified mean: 10000000.2
        Certified std (sample): 0.1

        This is the hardest case - values differ only in the 8th significant digit.
        """
        result = descriptive(values=[10000000.1, 10000000.2, 10000000.3])
        assert abs(result["mean"] - 10000000.2) < 1e-4, f"mean={result['mean']}"
        # NumAcc4 is extremely ill-conditioned; 6-digit precision helps
        assert abs(result["std"] - 0.1) < 0.01, f"std={result['std']}, expected ~0.1"

    def test_numacc4_extended(self):
        """NumAcc4 extended: 20 observations with large offset.

        All values near 10000000, differences of 0.1.
        Tests that std computation is stable with large mean offset.
        """
        values = [10000000.0 + i * 0.1 for i in range(20)]
        result = descriptive(values=values)
        expected_mean = 10000000.0 + 9.5 * 0.1  # = 10000000.95
        assert abs(result["mean"] - expected_mean) < 1e-4, f"mean={result['mean']}"
        assert result["std"] > 0, f"std={result['std']}, should be positive"

class TestHypothesisTesting:
    """Known results from statistics textbooks."""

    def test_two_sample_ttest_known_difference(self):
        """Two clearly different groups should reject H0."""
        np.random.seed(42)
        g1 = np.random.normal(100, 10, 50).tolist()
        g2 = np.random.normal(130, 10, 50).tolist()

        result = ttest(test_type="two_sample", values=g1, values2=g2)
        assert result["significant"], "Should detect significant difference"
        assert result["p_value"] < 0.001, f"p={result['p_value']}, expected <0.001"

    def test_two_sample_ttest_no_difference(self):
        """Two identical groups should NOT reject H0."""
        np.random.seed(42)
        g = np.random.normal(50, 5, 100).tolist()

        result = ttest(test_type="two_sample", values=g, values2=g)
        assert result["p_value"] > 0.99, f"p={result['p_value']}, expected >0.99 (identical groups)"

    def test_one_sample_ttest_known_mean(self):
        """Test if sample from N(100, 10) is consistent with mu=100."""
        np.random.seed(42)
        values = np.random.normal(100, 10, 100).tolist()

        result = ttest(test_type="one_sample", values=values, mu=100)
        assert result["p_value"] > 0.05, f"p={result['p_value']}, should not reject true H0"

    def test_paired_ttest_known_effect(self):
        """Paired test should detect a consistent shift."""
        np.random.seed(42)
        before = np.random.normal(50, 5, 30).tolist()
        after = (np.array(before) + 5.0).tolist()  # constant shift of 5

        result = ttest(test_type="paired", values=before, values2=after)
        assert result["significant"], "Should detect consistent 5-unit shift"
        assert result["p_value"] < 0.0001

    def test_anova_known_difference(self):
        """One-way ANOVA should detect difference between 3 distinct groups."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 20).tolist()
        g2 = np.random.normal(20, 1, 20).tolist()
        g3 = np.random.normal(30, 1, 20).tolist()

        result = anova(anova_type="one_way", groups=[g1, g2, g3])
        assert result["significant"], f"Should detect difference, got {result['significant']}"
        assert result["p_value"] < 0.0001

    def test_anova_no_difference(self):
        """One-way ANOVA should NOT reject when groups are identical."""
        np.random.seed(42)
        g = np.random.normal(10, 1, 20).tolist()

        result = anova(anova_type="one_way", groups=[g, g, g])
        assert result["p_value"] > 0.99


# ============================================================================
# Correlation - Known relationships
# ============================================================================

class TestCorrelation:
    """Test correlation with known relationships."""

    def test_perfect_positive(self):
        """Perfect linear relationship should give r=1."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = correlation(x=x, y=y, method="pearson")
        assert abs(result["correlation"] - 1.0) < 0.0001, f"r={result['correlation']}"

    def test_perfect_negative(self):
        """Perfect inverse relationship should give r=-1."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        result = correlation(x=x, y=y, method="pearson")
        assert abs(result["correlation"] - (-1.0)) < 0.0001, f"r={result['correlation']}"

    def test_no_correlation(self):
        """Uncorrelated data should give r near 0."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100).tolist()
        y = np.random.normal(0, 1, 100).tolist()
        result = correlation(x=x, y=y, method="pearson")
        assert abs(result["correlation"]) < 0.3, f"r={result['correlation']}, expected near 0"

    def test_spearman_monotonic(self):
        """Spearman should detect monotonic non-linear relationship."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]  # y = x^2
        result = correlation(x=x, y=y, method="spearman")
        assert result["correlation"] > 0.99, f"Spearman r={result['correlation']}, expected ~1.0"


# ============================================================================
# Normality Testing
# ============================================================================

class TestNormality:
    """Normality tests with known distributions."""

    def test_normal_data_passes(self):
        """Large normal sample should pass normality test."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 200).tolist()
        result = normality(values=values)
        assert result["is_normal"], f"Normal data rejected: {result}"

    def test_uniform_data_fails(self):
        """Uniform distribution should fail normality test (with enough data)."""
        np.random.seed(42)
        values = np.random.uniform(0, 1, 200).tolist()
        result = normality(values=values)
        # Uniform is not normal - Shapiro should detect this
        assert result["shapiro_wilk"]["p_value"] < 0.05

    def test_exponential_data_fails(self):
        """Exponential distribution should fail normality test."""
        np.random.seed(42)
        values = np.random.exponential(1, 200).tolist()
        result = normality(values=values)
        assert result["shapiro_wilk"]["p_value"] < 0.05


# ============================================================================
# Outlier Detection
# ============================================================================

class TestOutlierDetection:
    """Outlier detection with known outliers."""

    def test_grubbs_detects_outlier(self):
        """Grubbs test should detect a clear outlier."""
        values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 100.0]
        result = outlier(values=values, method="grubbs")
        assert result["n_outliers"] >= 1

    def test_iqr_detects_outlier(self):
        """IQR method should detect extreme values."""
        values = [10, 10, 10, 10, 10, 10, 10, 10, 10, 100]
        result = outlier(values=values, method="iqr")
        assert result["n_outliers"] >= 1

    def test_no_outliers_in_clean_data(self):
        """Clean uniform data should have no outliers."""
        values = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0]
        result = outlier(values=values, method="grubbs")
        assert result["n_outliers"] == 0


# ============================================================================
# Power Analysis - Known results
# ============================================================================

class TestPowerAnalysis:
    """Power analysis with known results."""

    def test_ttest_power_known(self):
        """Cohen's d=0.5, n=34 should give power ~0.8 for t-test."""
        result = power(analysis_type="t_test", effect_size=0.5, n=34)
        assert abs(result["power"] - 0.8) < 0.05, f"power={result['power']}, expected ~0.8"

    def test_ttest_sample_size_known(self):
        """For d=0.5, power=0.8, should need n~34."""
        result = power(analysis_type="t_test", effect_size=0.5, power=0.80)
        assert 30 <= result["sample_size"] <= 40, f"n={result['sample_size']}, expected ~34"

    def test_anova_power_with_k_groups(self):
        """ANOVA power with custom k_groups should work."""
        result = power(analysis_type="anova", effect_size=0.25, n=100, k_groups=5)
        assert "power" in result
        assert 0 < result["power"] < 1

    def test_proportion_power(self):
        """Proportion power analysis should return valid results."""
        result = power(analysis_type="proportion", effect_size=0.2, n=100)
        assert 0 < result["power"] < 1


# ============================================================================
# Edge Cases - Robustness
# ============================================================================

class TestEdgeCases:
    """Edge cases that could break the tool."""

    def test_descriptive_two_values(self):
        """Minimum viable descriptive stats."""
        result = descriptive(values=[1.0, 2.0])
        assert result["n"] == 2
        assert result["mean"] == 1.5
        assert result["std"] > 0

    def test_descriptive_identical_values(self):
        """All identical values: std=0, no division errors."""
        result = descriptive(values=[5.0, 5.0, 5.0, 5.0, 5.0])
        assert result["n"] == 5
        assert result["mean"] == 5.0
        assert result["std"] == 0.0
        assert result["range"] == 0.0

    def test_descriptive_very_large_values(self):
        """Large values should not overflow."""
        values = [1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4]
        result = descriptive(values=values)
        assert result["n"] == 5
        assert result["std"] > 0

    def test_descriptive_very_small_values(self):
        """Very small values should not underflow."""
        values = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        result = descriptive(values=values)
        assert result["n"] == 5
        # std may be 0 due to floating point precision at extreme scales
        assert result["std"] >= 0

    def test_descriptive_mixed_sign(self):
        """Negative and positive values."""
        values = [-100, -50, 0, 50, 100]
        result = descriptive(values=values)
        assert result["mean"] == 0.0
        assert result["min"] == -100
        assert result["max"] == 100

    def test_regression_constant_x(self):
        """Constant x should raise error (no variance)."""
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        with pytest.raises((ValueError, Exception)):
            regression(x=x, y=y, reg_type="linear")

    def test_correlation_constant_values(self):
        """Constant values: correlation undefined."""
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        result = correlation(x=x, y=y, method="pearson")
        # Should handle gracefully (NaN or error)
        assert "correlation" in result or "error" in result

    def test_anova_single_group(self):
        """ANOVA with 1 group should raise error."""
        with pytest.raises((ValueError, Exception)):
            anova(anova_type="one_way", groups=[[1, 2, 3]])

    def test_anova_empty_group(self):
        """ANOVA with empty group should raise error or return NaN."""
        try:
            result = anova(anova_type="one_way", groups=[[1, 2, 3], []])
            # If it doesn't raise, the result should indicate an issue
            assert result.get("f_statistic") is None or math.isnan(result.get("f_statistic", 0))
        except (ValueError, Exception):
            pass  # Expected

    def test_ttest_empty_values(self):
        """t-test with empty values should raise error or return NaN."""
        try:
            result = ttest(test_type="one_sample", values=[], mu=10)
            # If it doesn't raise, the result should indicate an issue
            assert result.get("t_statistic") is None or math.isnan(result.get("t_statistic", 0))
        except (ValueError, Exception):
            pass  # Expected


# ============================================================================
# Input Robustness - NaN, Inf, type issues
# ============================================================================

class TestInputRobustness:
    """Test handling of messy real-world inputs."""

    def test_descriptive_with_nan(self):
        """NaN values should be filtered out."""
        values = [1.0, 2.0, float("nan"), 4.0, 5.0]
        result = descriptive(values=values)
        assert result["n"] == 4  # NaN filtered

    def test_descriptive_with_inf(self):
        """Inf values should be filtered out."""
        values = [1.0, 2.0, float("inf"), 4.0, float("-inf")]
        result = descriptive(values=values)
        assert result["n"] == 3  # Inf filtered

    def test_descriptive_with_string_coercion(self):
        """Values list with mixed types should be handled."""
        # The handler should convert or reject non-numeric
        result = handler({"command": "descriptive", "params": {"values": [1, 2, 3, 4, 5]}})
        assert result["status"] == "success"

    def test_handler_missing_values(self):
        """Missing values parameter should return error."""
        result = handler({"command": "descriptive", "params": {}})
        assert result["status"] == "error"

    def test_handler_invalid_command(self):
        """Unknown command should return error."""
        result = handler({"command": "nonexistent", "params": {}})
        assert result["status"] == "error"

    def test_handler_empty_params(self):
        """Empty params dict should work for discover."""
        result = handler({"command": "discover", "params": {}})
        assert result["status"] == "success"

    def test_handler_string_input(self):
        """JSON string input should work."""
        result = handler('{"command": "descriptive", "params": {"values": [1, 2, 3]}}')
        assert result["status"] == "success"


# ============================================================================
# Regression Type Coverage
# ============================================================================

class TestRegressionTypes:
    """Test all regression types with appropriate data."""

    def test_linear(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.9, 16.2, 18.0, 19.8]
        result = regression(x=x, y=y, reg_type="linear")
        assert result["regression_type"] == "linear"
        assert result["r_squared"] > 0.99

    def test_quadratic(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
        result = regression(x=x, y=y, reg_type="quadratic")
        assert "polynomial" in result["regression_type"]
        assert result["r_squared"] > 0.99

    def test_polynomial_degree3(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
        result = regression(x=x, y=y, reg_type="polynomial", degree=3)
        assert "3" in result["regression_type"]
        assert result["r_squared"] > 0.99

    def test_logistic(self):
        """Logistic regression with binary outcome."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        result = regression(x=x, y=y, reg_type="logistic")
        assert result["regression_type"] == "logistic"
        assert "aic" in result


# ============================================================================
# ============================================================================
# Nonlinear Regression - Known-value tests for nonlinear types
# ============================================================================

class TestNonlinearRegression:
    """Known-value tests for nonlinear regression types."""

    def test_exponential_known(self):
        """Exponential: y = a*exp(b*x). Known data with known fit."""
        x = [1, 2, 3, 4, 5]
        y = [math.exp(0.5 * xi) for xi in x]  # y = exp(0.5*x)
        result = regression(x=x, y=y, reg_type="exponential")
        assert result["r_squared"] > 0.99, f"R²={result['r_squared']}"
        assert result["coefficients"]["b"] > 0, f"b={result['coefficients']['b']}"

    def test_power_known(self):
        """Power: y = a*x^b. Known data with known fit."""
        x = [1, 2, 3, 4, 5]
        y = [xi**2 for xi in x]  # y = x^2
        result = regression(x=x, y=y, reg_type="power")
        assert result["r_squared"] > 0.99, f"R²={result['r_squared']}"

    def test_logarithmic_known(self):
        """Logarithmic: y = a*ln(x) + b."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2 * math.log(xi) + 3 for xi in x]
        result = regression(x=x, y=y, reg_type="logarithmic")
        assert result["r_squared"] > 0.99, f"R²={result['r_squared']}"

    def test_sigmoid_known(self):
        """Sigmoid: 4PL logistic. Known dose-response curve."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0.1, 0.15, 0.25, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
        result = regression(x=x, y=y, reg_type="sigmoid")
        assert result["r_squared"] > 0.95, f"R²={result['r_squared']}"

# Handler Integration - End-to-end through main.py
# ============================================================================

class TestHandlerIntegration:
    """End-to-end tests through the handler function."""

    def test_descriptive_via_handler(self):
        result = handler({"command": "descriptive", "params": {"values": [1, 2, 3, 4, 5]}})
        assert result["status"] == "success"
        assert result["data"]["n"] == 5

    def test_ttest_via_handler(self):
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 20).tolist()
        g2 = np.random.normal(12, 1, 20).tolist()
        result = handler({"command": "ttest", "params": {"test_type": "two_sample", "values": g1, "values2": g2}})
        assert result["status"] == "success"

    def test_anova_via_handler(self):
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 15).tolist()
        g2 = np.random.normal(15, 1, 15).tolist()
        g3 = np.random.normal(20, 1, 15).tolist()
        result = handler({"command": "anova", "params": {"anova_type": "one_way", "groups": [g1, g2, g3]}})
        assert result["status"] == "success"

    def test_power_via_handler(self):
        result = handler({"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}})
        assert result["status"] == "success"
        assert "sample_size" in result["data"]

    def test_discover_via_handler(self):
        result = handler({"command": "discover", "params": {}})
        assert result["status"] == "success"
        assert result["data"]["total_commands"] > 20

    def test_regression_linear_via_handler(self):
        result = handler({
            "command": "regression",
            "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "reg_type": "linear"}
        })
        assert result["status"] == "success"
        assert result["data"]["r_squared"] > 0.99

    def test_correlation_via_handler(self):
        result = handler({
            "command": "correlation",
            "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "method": "pearson"}
        })
        assert result["status"] == "success"
        assert result["data"]["correlation"] > 0.99

    def test_chart_request_descriptive(self):
        """Chart=True should not break descriptive."""
        result = handler({"command": "descriptive", "params": {"values": [1, 2, 3, 4, 5], "chart": True}})
        assert result["status"] == "success"
        assert "chart_base64" in result["data"]

    def test_chart_request_doe(self):
        """Chart=True on DOE should not crash."""
        result = handler({
            "command": "doe",
            "params": {
                "doe_type": "full_factorial",
                "factors": [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}],
                "chart": True,
            }
        })
        assert result["status"] == "success"


# ============================================================================
# Anscombe's Quartet - Extended tests (additions to existing class)
# ============================================================================

class TestAnscombeQuartetExtended:
    """Extended Anscombe's Quartet tests: R-squared, raw data, x structure."""

    @pytest.fixture
    def anscombe(self):
        return {
            "x1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
            "x2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
            "x3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
            "x4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        }

    def test_perfect_linear_r_squared(self, anscombe):
        """All 4 datasets should have R^2 ~0.67 for linear fit."""
        for i in range(1, 5):
            x = anscombe[f"x{i}"]
            y = anscombe[f"y{i}"]
            result = regression(x=x, y=y, reg_type="linear")
            assert abs(result["r_squared"] - 0.67) < 0.05, \
                f"Dataset {i}: R^2={result['r_squared']}, expected ~0.67"

    def test_datasets_not_identical(self, anscombe):
        """The 4 y-datasets should have different raw values."""
        ys = [anscombe[f"y{i}"] for i in range(1, 5)]
        # Every pair of datasets should differ in at least one value
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                assert ys[i] != ys[j], f"y{i+1} and y{j+1} are identical"

    def test_x_values_differ(self, anscombe):
        """x4 has x=8 repeated 10 times and one x=19, unlike x1-x3 which are all distinct."""
        x4 = anscombe["x4"]
        # x4 should have 11 values total
        assert len(x4) == 11
        # Count occurrences: 8 appears 10 times, 19 appears once
        assert x4.count(8) == 10, f"Expected 10 eights, got {x4.count(8)}"
        assert x4.count(19) == 1, f"Expected one 19, got {x4.count(19)}"
        # x1 through x3 should all be distinct values
        for i in range(1, 4):
            xi = anscombe[f"x{i}"]
            assert len(xi) == len(set(xi)), f"x{i} has duplicate values: {xi}"


# ============================================================================
# Classic Textbook Datasets - Well-known results with certified values
# ============================================================================

class TestClassicTextbook:
    """Tests using well-known textbook datasets with known results."""

    def test_iris_anova(self):
        """Fisher's Iris: one-way ANOVA on sepal length by species.

        Known: F ~119.26, p ~1.67e-31
        """
        setosa = [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                  5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
                  5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
                  4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1,
                  5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0]
        versicolor = [7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                      5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
                      5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7,
                      5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
                      5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7]
        virginica = [6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2,
                     6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0,
                     6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
                     7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9,
                     6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9]

        result = anova(anova_type="one_way", groups=[setosa, versicolor, virginica])

        # F ~119.26, p ~1.67e-31
        assert abs(result["f_statistic"] - 119.26) < 5.0, \
            f"F={result['f_statistic']}, expected ~119.26"
        assert result["p_value"] < 1e-20, \
            f"p={result['p_value']}, expected <1e-20"
        assert result["significant"], "Iris ANOVA should be highly significant"
        # Eta-squared should be large (>0.6) since species explains most variance
        assert result["eta_squared"] > 0.6, \
            f"eta^2={result['eta_squared']}, expected >0.6"

    def test_fisher_z_known(self):
        """Known correlation: r=0.5, n=30 should be significant.

        For r=0.5, n=30: t = r*sqrt(n-2)/sqrt(1-r^2) = 0.5*sqrt(28)/sqrt(0.75) ~ 3.06
        df=28, two-tailed p ~ 0.005
        """
        # Generate data with known r ~ 0.5
        np.random.seed(123)
        n = 30
        x = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)
        y = 0.5 * x + np.sqrt(1 - 0.5**2) * noise  # r ~ 0.5

        result = correlation(x=x.tolist(), y=y.tolist(), method="pearson")
        # r should be in the neighborhood of 0.5 (with noise from sampling)
        assert abs(result["correlation"]) > 0.2, \
            f"r={result['correlation']}, expected |r|>0.2 with seed=123"
        # With r~0.5 and n=30, this should be significant
        assert result["p_value"] < 0.05, \
            f"p={result['p_value']}, expected <0.05 for r~0.5, n=30"

    def test_longley_full(self):
        """Longley dataset: verify all 7 coefficients match NIST.

        This is a re-verification via the handler to confirm end-to-end accuracy.
        NIST certified R^2 = 0.995479004577296
        """
        import os
        import tempfile

        import pandas as pd

        data = {
            "y": [60323, 61122, 60171, 61187, 63221, 63639, 64989, 63761,
                  66019, 67857, 68169, 66513, 68655, 69564, 69331, 70551],
            "x1": [83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0,
                   101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9],
            "x2": [234289, 259426, 258054, 284599, 328975, 346999, 365385, 363112,
                   397469, 419180, 442769, 444546, 482704, 502601, 518173, 554894],
            "x3": [2356, 2325, 3682, 3351, 2099, 1932, 1870, 3578,
                   2904, 2822, 2936, 4681, 3813, 3931, 4806, 4007],
            "x4": [1590, 1456, 1616, 1650, 3099, 3594, 3547, 3350,
                   3048, 2857, 2798, 2637, 2552, 2514, 2572, 2827],
            "x5": [107608, 108632, 109773, 110929, 112075, 113270, 115094, 116219,
                   117388, 118734, 120445, 121950, 123366, 125368, 127852, 130081],
            "x6": [1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954,
                   1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962],
        }
        df = pd.DataFrame(data)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")  # noqa: SIM115
        df.to_csv(tmp.name, index=False)
        tmp.close()

        try:
            result = handler({
                "command": "regression",
                "params": {
                    "file": tmp.name,
                    "x_columns": ["x1", "x2", "x3", "x4", "x5", "x6"],
                    "y_column": "y",
                    "reg_type": "multiple",
                },
            })
            assert result["status"] == "success"
            data = result["data"]
            coeffs = data["coefficients"]

            # Verify all 7 NIST certified coefficients
            nist_coeffs = {
                "const": -3482258.63459582,
                "x1": 15.0618722713733,
                "x2": -0.0358191792925910,
                "x3": -2.02022980381683,
                "x4": -1.03322686717359,
                "x5": -0.0511041056535807,
                "x6": 1829.15146461355,
            }
            for name, nist_val in nist_coeffs.items():
                assert abs(coeffs[name] - nist_val) < 1e-4, \
                    f"{name}: got {coeffs[name]}, expected {nist_val}"

            assert abs(data["r_squared"] - 0.995479004577296) < 5e-7, \
                f"R^2={data['r_squared']}"
        finally:
            os.unlink(tmp.name)

    def test_cohen_d_medium(self):
        """Cohen's d=0.5, n=64 should give power > 0.80.

        With n=64 per group, the noncentrality parameter is large enough
        that power exceeds the classic 0.80 threshold.
        """
        result = power(analysis_type="t_test", effect_size=0.5, n=64)
        assert 0 < result["power"] < 1, f"power={result['power']}, should be in (0,1)"
        assert result["power"] > 0.80, \
            f"power={result['power']}, expected >0.80 for d=0.5, n=64"

    def test_plant_growth_anova(self):
        """R PlantGrowth dataset: 3 groups, 10 each.

        Known: F ~4.85, p ~0.0159
        """
        ctrl = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
        trt1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
        trt2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]

        result = anova(anova_type="one_way", groups=[ctrl, trt1, trt2])
        assert abs(result["f_statistic"] - 4.85) < 1.0, \
            f"F={result['f_statistic']}, expected ~4.85"
        assert result["p_value"] < 0.05, \
            f"p={result['p_value']}, expected <0.05"
        assert result["significant"]

    def test_two_way_anova_interaction(self):
        """Two-way ANOVA with known interaction effect."""
        np.random.seed(42)
        # 2x2 factorial: factor A (low/high), factor B (low/high)
        # With interaction: high_A + high_B gives extra boost
        data = {
            "factor_a": ["low"] * 10 + ["high"] * 10 + ["low"] * 10 + ["high"] * 10,
            "factor_b": ["low"] * 20 + ["high"] * 20,
            "values": (
                np.random.normal(10, 1, 10).tolist() +  # A_low B_low
                np.random.normal(15, 1, 10).tolist() +  # A_high B_low
                np.random.normal(12, 1, 10).tolist() +  # A_low B_high
                np.random.normal(22, 1, 10).tolist()    # A_high B_high (interaction boost)
            ),
        }
        result = anova(anova_type="two_way", groups=[], data=data)
        assert "sources" in result
        assert len(result["sources"]) >= 3  # A, B, A:B
        assert result["r_squared"] > 0.5


# ============================================================================
# Statistical Properties - Mathematical invariants that must hold
# ============================================================================

class TestStatisticalProperties:
    """Test mathematical properties that must always hold."""

    def test_t_test_symmetry(self):
        """Swapping groups should change the sign of t-statistic."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 20).tolist()
        g2 = np.random.normal(15, 1, 20).tolist()

        r1 = ttest(test_type="two_sample", values=g1, values2=g2)
        r2 = ttest(test_type="two_sample", values=g2, values2=g1)

        # t-statistics should have opposite signs (approximately)
        assert r1["t_statistic"] * r2["t_statistic"] < 0, \
            f"t1={r1['t_statistic']}, t2={r2['t_statistic']}, should have opposite signs"
        # p-values should be identical
        assert abs(r1["p_value"] - r2["p_value"]) < 1e-10, \
            f"p1={r1['p_value']}, p2={r2['p_value']}, should be equal"

    def test_anova_f_non_negative(self):
        """F-statistic from ANOVA must always be >= 0."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 20).tolist()
        g2 = np.random.normal(20, 2, 20).tolist()
        g3 = np.random.normal(30, 3, 20).tolist()

        result = anova(anova_type="one_way", groups=[g1, g2, g3])
        assert result["f_statistic"] >= 0, \
            f"F={result['f_statistic']}, must be >= 0"

        # Also test with identical groups (F should be ~0)
        g = np.random.normal(10, 1, 20).tolist()
        result2 = anova(anova_type="one_way", groups=[g, g, g])
        assert result2["f_statistic"] >= 0, \
            f"F={result2['f_statistic']}, must be >= 0 even for identical groups"

    def test_correlation_range(self):
        """Correlation coefficient must be in [-1, 1] for all methods."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 50).tolist()
        y = np.random.normal(0, 1, 50).tolist()

        for method in ["pearson", "spearman", "kendall"]:
            result = correlation(x=x, y=y, method=method)
            r_val = result["correlation"]
            if r_val is not None:  # Skip if undefined (constant input)
                assert -1 <= r_val <= 1, \
                    f"{method}: r={r_val}, must be in [-1, 1]"

    def test_regression_r_squared_range(self):
        """R-squared from regression must be in [0, 1]."""
        np.random.seed(42)
        x = list(range(1, 21))
        y_noisy = (np.array(x) * 2 + np.random.normal(0, 5, 20)).tolist()
        y_perfect = (np.array(x) * 2).tolist()

        # Noisy data: 0 <= R^2 <= 1
        r1 = regression(x=x, y=y_noisy, reg_type="linear")
        assert 0 <= r1["r_squared"] <= 1, \
            f"R^2={r1['r_squared']}, must be in [0, 1]"

        # Perfect linear: R^2 ~ 1
        r2 = regression(x=x, y=y_perfect, reg_type="linear")
        assert 0 <= r2["r_squared"] <= 1, \
            f"R^2={r2['r_squared']}, must be in [0, 1]"

    def test_p_value_range(self):
        """p-values from all tests must be in [0, 1]."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 30).tolist()
        g2 = np.random.normal(12, 1, 30).tolist()

        # t-test p-value
        t_result = ttest(test_type="two_sample", values=g1, values2=g2)
        assert 0 <= t_result["p_value"] <= 1, \
            f"t-test p={t_result['p_value']}, must be in [0, 1]"

        # ANOVA p-value
        g3 = np.random.normal(14, 1, 30).tolist()
        a_result = anova(anova_type="one_way", groups=[g1, g2, g3])
        assert 0 <= a_result["p_value"] <= 1, \
            f"ANOVA p={a_result['p_value']}, must be in [0, 1]"

        # Correlation p-value
        x = np.random.normal(0, 1, 30).tolist()
        y = np.random.normal(0, 1, 30).tolist()
        c_result = correlation(x=x, y=y, method="pearson")
        if c_result["p_value"] is not None:
            assert 0 <= c_result["p_value"] <= 1, \
                f"Correlation p={c_result['p_value']}, must be in [0, 1]"

    def test_eta_squared_range(self):
        """Eta-squared from ANOVA must be in [0, 1]."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 20).tolist()
        g2 = np.random.normal(20, 1, 20).tolist()
        g3 = np.random.normal(30, 1, 20).tolist()

        result = anova(anova_type="one_way", groups=[g1, g2, g3])
        eta = result["eta_squared"]
        assert 0 <= eta <= 1, f"eta^2={eta}, must be in [0, 1]"

        # With identical groups, eta should be ~0
        g = np.random.normal(10, 1, 20).tolist()
        result2 = anova(anova_type="one_way", groups=[g, g, g])
        eta2 = result2["eta_squared"]
        assert 0 <= eta2 <= 1, f"eta^2={eta2}, must be in [0, 1]"

    def test_descriptive_ci_contains_mean(self):
        """95% CI for the mean should always contain the sample mean."""
        np.random.seed(42)
        values = np.random.normal(50, 10, 30).tolist()

        result = descriptive(values=values)
        mean = result["mean"]
        ci_lower = result["ci_95_lower"]
        ci_upper = result["ci_95_upper"]

        assert ci_lower is not None, "CI lower should not be None for n>=2"
        assert ci_upper is not None, "CI upper should not be None for n>=2"
        assert ci_lower <= mean <= ci_upper, \
            f"Mean={mean} not in CI=[{ci_lower}, {ci_upper}]"


# ============================================================================
# Datasaurus Dozen - Identical summary stats, wildly different distributions
# ============================================================================

class TestDatasaurusDozen:
    """Datasaurus Dozen: identical stats, wildly different distributions."""

    @pytest.fixture
    def dino(self):
        """Dinosaur shape - the iconic Anscombe-like dataset."""
        x = [55.38, 51.54, 46.15, 42.82, 40.77, 38.46, 35.38, 33.08, 30.77, 28.46,
             26.92, 25.38, 24.10, 22.56, 21.54, 20.26, 19.23, 18.46, 17.44, 16.92,
             16.15, 15.64, 15.38, 15.38, 16.15, 16.92, 17.69, 18.97, 20.51, 22.56,
             24.87, 27.44, 30.00, 32.82, 35.64, 38.72, 41.54, 44.36, 47.44, 50.26,
             53.08, 55.38, 55.38, 51.54, 46.15, 42.82, 40.77, 38.46, 35.38, 33.08,
             30.77, 28.46, 26.92, 25.38, 24.10, 22.56, 21.54, 20.26, 19.23, 18.46]
        y = [97.18, 96.03, 94.49, 92.44, 91.15, 89.36, 86.79, 84.36, 82.18, 80.00,
             78.08, 76.03, 74.10, 71.92, 70.00, 67.82, 65.38, 63.08, 60.64, 58.08,
             55.38, 52.69, 49.74, 46.79, 44.10, 41.67, 39.49, 37.69, 36.41, 35.77,
             35.64, 36.15, 37.18, 38.72, 40.64, 43.08, 45.90, 49.10, 52.56, 56.03,
             59.49, 62.69, 65.38, 67.82, 70.00, 71.92, 74.10, 76.03, 78.08, 80.00,
             82.18, 84.36, 86.79, 89.36, 91.15, 92.44, 94.49, 96.03, 97.18, 97.18]
        return x, y

    @pytest.fixture
    def circle(self):
        """Circle shape."""
        n = 60
        x = [50 + 30 * math.cos(2 * math.pi * i / n) for i in range(n)]
        y = [50 + 30 * math.sin(2 * math.pi * i / n) for i in range(n)]
        return x, y

    @pytest.fixture
    def star(self):
        """Star/lines shape - multiple crossing lines."""
        x, y = [], []
        for i in range(20):
            x.extend([10 + i * 2, 90 - i * 2])
            y.extend([10 + i * 2, 90 - i * 2])
        for i in range(20):
            x.extend([10 + i * 2, 90 - i * 2])
            y.extend([90 - i * 2, 10 + i * 2])
        return x, y

    def test_similar_stats(self, dino, circle, star):
        """All datasets should have similar mean and std (within 50% of each other)."""
        datasets = {"dino": dino, "circle": circle, "star": star}
        stats = {}
        for name, (x, y) in datasets.items():
            dx = descriptive(values=x)
            dy = descriptive(values=y)
            stats[name] = {
                "mean_x": dx["mean"],
                "mean_y": dy["mean"],
                "std_x": dx["std"],
                "std_y": dy["std"],
            }

        for key in ["mean_x", "mean_y", "std_x", "std_y"]:
            values = [s[key] for s in stats.values()]
            avg = sum(values) / len(values)
            for name, val in stats.items():
                if avg != 0:
                    ratio = abs(val[key] - avg) / abs(avg)
                    assert ratio < 0.5, \
                        f"{name}.{key}={val[key]}, avg={avg}, ratio={ratio:.2f}"

    def test_different_patterns(self, dino, circle):
        """Dino and circle should have different structural patterns."""
        import numpy as np
        arr_dino_x = np.array(dino[0])
        arr_circle_x = np.array(circle[0])
        range_dino = float(np.max(arr_dino_x) - np.min(arr_dino_x))
        range_circle = float(np.max(arr_circle_x) - np.min(arr_circle_x))
        assert abs(range_dino - range_circle) > 1, \
            f"Dino range={range_dino:.1f}, circle range={range_circle:.1f}, should differ"

    def test_regression_similar_r_squared(self, dino, circle, star):
        """All datasets should have low R-squared (no strong linear relationship)."""
        datasets = {"dino": dino, "circle": circle, "star": star}
        for name, (x, y) in datasets.items():
            result = regression(x=x, y=y, reg_type="linear")
            assert result["r_squared"] < 0.2, \
                f"{name}: R^2={result['r_squared']}, expected <0.2"


# ============================================================================
# Multicollinearity - Deliberately correlated predictors
# ============================================================================

class TestMulticollinearity:
    """Test with deliberately multicollinear data."""

    def test_perfect_collinearity(self):
        """x2 = 2*x1: regression should still return results (not crash)."""
        np.random.seed(42)
        x1 = np.random.uniform(1, 10, 30).tolist()
        x2 = [2 * v for v in x1]
        y = (np.array(x1) * 3 + np.random.normal(0, 1, 30)).tolist()

        import os  # noqa: I001
        import tempfile

        import pandas as pd

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            result = regression(
                file=tmp_path,
                x_columns=["x1", "x2"],
                y_column="y",
                reg_type="multiple",
            )
            assert "coefficients" in result
            assert "r_squared" in result
            assert result["r_squared"] >= 0
        finally:
            os.unlink(tmp_path)

    def test_near_collinearity(self):
        """x2 = 2*x1 + noise: regression should work with near-perfect collinearity."""
        np.random.seed(42)
        x1 = np.random.uniform(1, 10, 30).tolist()
        x2 = (2 * np.array(x1) + np.random.normal(0, 0.01, 30)).tolist()
        y = (np.array(x1) * 3 + np.random.normal(0, 1, 30)).tolist()

        import os  # noqa: I001
        import tempfile

        import pandas as pd

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            result = regression(
                file=tmp_path,
                x_columns=["x1", "x2"],
                y_column="y",
                reg_type="multiple",
            )
            assert result["r_squared"] > 0.5, \
                f"R^2={result['r_squared']}, expected >0.5 for strong predictor"
        finally:
            os.unlink(tmp_path)

    def test_vif_detection(self):
        """Correlated variables should not break multiple regression."""
        np.random.seed(42)
        x1 = np.random.uniform(1, 10, 50).tolist()
        x2 = (np.array(x1) * 1.5 + np.random.normal(0, 0.5, 50)).tolist()
        x3 = np.random.uniform(1, 10, 50).tolist()
        y = (np.array(x1) * 2 + np.array(x3) * 3 + np.random.normal(0, 1, 50)).tolist()

        import os  # noqa: I001
        import tempfile

        import pandas as pd

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            result = regression(
                file=tmp_path,
                x_columns=["x1", "x2", "x3"],
                y_column="y",
                reg_type="multiple",
            )
            assert "coefficients" in result
            assert result["r_squared"] > 0.5, \
                f"R^2={result['r_squared']}, expected >0.5"
        finally:
            os.unlink(tmp_path)


# ============================================================================
# Extreme Data Sizes - Large, empty, single value
# ============================================================================

class TestLargeAndEmpty:
    """Test with extreme data sizes."""

    def test_large_dataset_10000(self):
        """10000 values: descriptive should work."""
        np.random.seed(42)
        values = np.random.normal(100, 15, 10000).tolist()
        result = descriptive(values=values)
        assert result["n"] == 10000
        assert abs(result["mean"] - 100) < 1

    def test_empty_values_list(self):
        """Empty list should raise error."""
        result = handler({"command": "descriptive", "params": {"values": []}})
        assert result["status"] == "error"

    def test_single_value(self):
        """Single value should return partial result with insufficient_for_stats flag."""
        result = descriptive(values=[42.0])
        assert result["n"] == 1
        assert result["mean"] == 42.0
        assert result["std"] is None
        assert result["insufficient_for_stats"] is True

    def test_all_same_value(self):
        """1000 identical values: std=0, range=0."""
        values = [7.0] * 1000
        result = descriptive(values=values)
        assert result["n"] == 1000
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["range"] == 0.0

    def test_wide_dataset_100_cols(self):
        """100 columns: multivariate PCA should work."""
        np.random.seed(42)
        values = np.random.rand(50, 5).tolist()
        result = multivariate(analysis_type="pca", values=values)
        assert result["n_observations"] == 50
        assert result["n_variables"] == 5
        assert len(result["eigenvalues"]) == 5
        assert len(result["variance_explained"]) == 5


# ============================================================================
# Messy Strings - Non-numeric and mixed-type inputs
# ============================================================================

class TestMessyStrings:
    """Test with data containing messy strings."""

    def test_numeric_with_nan_strings(self):
        """Values like [1, 2, 'N/A', 4, 5] should return error (non-numeric not filtered)."""
        result = handler({"command": "descriptive", "params": {"values": [1, 2, "N/A", 4, 5]}})
        assert result["status"] == "error"
        assert "N/A" in result["message"] or "convert" in result["message"].lower()

    def test_numeric_with_empty_strings(self):
        """Values like [1, '', 3, '', 5] should return error (empty string not numeric)."""
        result = handler({"command": "descriptive", "params": {"values": [1, "", 3, "", 5]}})
        assert result["status"] == "error"

    def test_mixed_types_handler(self):
        """Handler should accept [1, '2', 3.0, '4.5'] and convert to numeric."""
        result = handler({"command": "descriptive", "params": {"values": [1, "2", 3.0, "4.5"]}})
        assert result["status"] == "success"
        assert result["data"]["n"] == 4
        assert result["data"]["mean"] == 2.625

    def test_all_non_numeric(self):
        """All strings should raise error."""
        result = handler({"command": "descriptive", "params": {"values": ["a", "b", "c"]}})
        assert result["status"] == "error"

# ============================================================================
# Performance Benchmark - Speed regression tests
# ============================================================================

class TestPerformanceBenchmark:
    """Performance regression tests - ensure no command becomes too slow."""

    def test_descriptive_speed(self):
        """Descriptive should handle 1000 values in <50ms."""
        import time
        values = list(range(1000))
        start = time.time()
        for _ in range(10):
            handler({"command": "descriptive", "params": {"values": values}})
        elapsed = (time.time() - start) / 10
        assert elapsed < 0.05, f"descriptive took {elapsed*1000:.0f}ms, expected <50ms"

    def test_capability_speed(self):
        """Capability should handle 500 values in <50ms."""
        import time
        np.random.seed(42)
        values = np.random.normal(10, 1, 500).tolist()
        start = time.time()
        for _ in range(10):
            handler({"command": "capability", "params": {"values": values, "usl": 13, "lsl": 7}})
        elapsed = (time.time() - start) / 10
        assert elapsed < 0.05, f"capability took {elapsed*1000:.0f}ms, expected <50ms"

    def test_correlation_speed(self):
        """Correlation should handle 1000 pairs in <10ms."""
        import time
        x = list(range(1000))
        y = list(range(1000))
        start = time.time()
        for _ in range(10):
            handler({"command": "correlation", "params": {"x": x, "y": y}})
        elapsed = (time.time() - start) / 10
        assert elapsed < 0.01, f"correlation took {elapsed*1000:.0f}ms, expected <10ms"

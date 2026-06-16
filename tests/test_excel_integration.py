"""Integration tests: 100 Excel files x multiple commands.

Tests the full handler pipeline: Excel file -> load_data -> command -> result.
Covers: explore, descriptive, normality, outlier, control_chart, capability,
trend, clean, transform, ttest, timeseries, regression, multivariate, report.
"""

import glob
import os

import pytest

from main import handler

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "excel")


def xlsx(name):
    """Get full path to a fixture Excel file."""
    return os.path.join(FIXTURE_DIR, name)


def all_fixtures():
    """List all Excel fixture files."""
    return sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.xlsx")))


def succeed(result):
    """Check that handler returned success/warning or a known acceptable error."""
    if result["status"] in ("success", "warning"):
        return True
    # Acceptable errors: validation errors, missing data, etc.
    acceptable = {"VALIDATION_ERROR", "PARAM_ERROR", "DATA_ERROR", "COMPUTATION_ERROR", "MISSING_DEPENDENCY", "FILE_NOT_FOUND"}
    return result.get("error_type") in acceptable


# ============================================================================
# explore command - tests file structure discovery
# ============================================================================

class TestExplore:
    """Test explore command with various Excel files."""

    @pytest.mark.parametrize("fname", [
        "001_singlecol_5rows.xlsx",
        "008_threecol_5000rows.xlsx",
        "012_50cols_100rows.xlsx",
        "036_missing_none.xlsx",
        "038_missing_random_20pct.xlsx",
        "042_missing_entire_col.xlsx",
        "046_mixed_num_text.xlsx",
        "048_mixed_all_text.xlsx",
        "051_mixed_date_numeric.xlsx",
        "055_mixed_unicode_cols.xlsx",
        "066_sheet_single.xlsx",
        "067_sheet_two.xlsx",
        "069_sheet_chinese_name.xlsx",
        "070_sheet_spaces_name.xlsx",
        "074_header_standard.xlsx",
        "075_header_numeric.xlsx",
        "078_header_duplicate_names.xlsx",
        "100_stat_comprehensive.xlsx",
    ])
    def test_explore(self, fname):
        result = handler({"command": "explore", "params": {"file": xlsx(fname)}})
        assert succeed(result), f"explore failed for {fname}: {result}"

    def test_explore_with_sheet(self):
        result = handler({"command": "explore", "params": {"file": xlsx("067_sheet_two.xlsx"), "sheet": "Summary"}})
        assert succeed(result)

    def test_explore_nonexistent_sheet(self):
        result = handler({"command": "explore", "params": {"file": xlsx("067_sheet_two.xlsx"), "sheet": "NonExistent"}})
        assert result["status"] == "success"  # explore returns error dict, not exception
        assert result.get("data", {}).get("error") is True


# ============================================================================
# descriptive command - single column via generic loader
# ============================================================================

class TestDescriptive:
    """Test descriptive command with Excel file input."""

    @pytest.mark.parametrize("fname", [
        "001_singlecol_5rows.xlsx",
        "002_singlecol_50rows.xlsx",
        "003_singlecol_500rows.xlsx",
        "004_singlecol_5000rows.xlsx",
        "021_dist_normal.xlsx",
        "022_dist_uniform.xlsx",
        "023_dist_exponential.xlsx",
        "024_dist_lognormal.xlsx",
        "025_dist_bimodal.xlsx",
        "026_dist_chisquare.xlsx",
        "031_dist_constant.xlsx",
        "032_dist_near_constant.xlsx",
        "033_dist_integers.xlsx",
        "034_dist_small_decimals.xlsx",
        "056_edge_single_row.xlsx",
        "057_edge_two_rows.xlsx",
        "058_edge_very_large.xlsx",
        "059_edge_very_small.xlsx",
        "060_edge_all_zeros.xlsx",
        "061_edge_all_negative.xlsx",
        "064_edge_scientific.xlsx",
    ])
    def test_descriptive_single_col(self, fname):
        result = handler({"command": "descriptive", "params": {"file": xlsx(fname)}})
        assert succeed(result), f"descriptive failed for {fname}: {result}"

    def test_descriptive_with_column_name(self):
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("035_dist_mixed_scales.xlsx"), "column": "medium"
        }})
        assert succeed(result)

    def test_descriptive_with_column_index(self):
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("008_threecol_5000rows.xlsx"), "column": "0"
        }})
        assert succeed(result)

    def test_descriptive_missing_data(self):
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("038_missing_random_20pct.xlsx")
        }})
        assert succeed(result)
        if result["status"] == "success":
            assert result["data"]["n"] < 200  # some values were NaN

    def test_descriptive_mostly_missing(self):
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("039_missing_random_50pct.xlsx")
        }})
        assert succeed(result)

    def test_descriptive_single_valid(self):
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("045_missing_single_valid.xlsx")
        }})
        # Should either succeed with n=1 or fail gracefully
        assert succeed(result)


# ============================================================================
# normality command
# ============================================================================

class TestNormality:
    """Test normality command with Excel file input."""

    @pytest.mark.parametrize("fname", [
        "021_dist_normal.xlsx",
        "022_dist_uniform.xlsx",
        "023_dist_exponential.xlsx",
        "025_dist_bimodal.xlsx",
        "033_dist_integers.xlsx",
        "060_edge_all_zeros.xlsx",
    ])
    def test_normality(self, fname):
        result = handler({"command": "normality", "params": {"file": xlsx(fname)}})
        assert succeed(result), f"normality failed for {fname}: {result}"


# ============================================================================
# outlier command
# ============================================================================

class TestOutlier:
    """Test outlier command with Excel file input."""

    @pytest.mark.parametrize("fname,method", [
        ("021_dist_normal.xlsx", "grubbs"),
        ("088_stat_outliers.xlsx", "grubbs"),
        ("088_stat_outliers.xlsx", "iqr"),
        ("088_stat_outliers.xlsx", "zscore"),
        ("065_edge_low_cardinality.xlsx", "grubbs"),
        ("031_dist_constant.xlsx", "grubbs"),
    ])
    def test_outlier(self, fname, method):
        result = handler({"command": "outlier", "params": {
            "file": xlsx(fname), "method": method
        }})
        assert succeed(result), f"outlier({method}) failed for {fname}: {result}"


# ============================================================================
# control_chart command
# ============================================================================

class TestControlChart:
    """Test control_chart command with Excel file input."""

    @pytest.mark.parametrize("fname,chart_type", [
        ("003_singlecol_500rows.xlsx", "imr"),
        ("097_stat_control_in.xlsx", "imr"),
        ("098_stat_control_ooc.xlsx", "imr"),
        ("002_singlecol_50rows.xlsx", "xbar"),
        ("021_dist_normal.xlsx", "ewma"),
    ])
    def test_control_chart(self, fname, chart_type):
        result = handler({"command": "control_chart", "params": {
            "file": xlsx(fname), "chart_type": chart_type
        }})
        assert succeed(result), f"control_chart({chart_type}) failed for {fname}: {result}"


# ============================================================================
# capability command
# ============================================================================

class TestCapability:
    """Test capability command with Excel file input."""

    @pytest.mark.parametrize("fname", [
        "087_stat_capability.xlsx",
        "021_dist_normal.xlsx",
        "003_singlecol_500rows.xlsx",
    ])
    def test_capability(self, fname):
        result = handler({"command": "capability", "params": {
            "file": xlsx(fname), "usl": 11.0, "lsl": 9.0
        }})
        assert succeed(result), f"capability failed for {fname}: {result}"


# ============================================================================
# trend command
# ============================================================================

class TestTrend:
    """Test trend command with Excel file input."""

    @pytest.mark.parametrize("fname,test_type", [
        ("085_stat_timeseries.xlsx", "cusum"),
        ("085_stat_timeseries.xlsx", "ewma"),
        ("085_stat_timeseries.xlsx", "runs"),
        ("086_stat_process_shift.xlsx", "cusum"),
    ])
    def test_trend(self, fname, test_type):
        result = handler({"command": "trend", "params": {
            "file": xlsx(fname), "test_type": test_type
        }})
        assert succeed(result), f"trend({test_type}) failed for {fname}: {result}"


# ============================================================================
# clean command
# ============================================================================

class TestClean:
    """Test clean command with Excel file input."""

    @pytest.mark.parametrize("fname,method", [
        ("037_missing_random_5pct.xlsx", "drop"),
        ("038_missing_random_20pct.xlsx", "impute_mean"),
        ("038_missing_random_20pct.xlsx", "impute_median"),
        ("088_stat_outliers.xlsx", "winsorize"),
        ("088_stat_outliers.xlsx", "clip"),
    ])
    def test_clean(self, fname, method):
        result = handler({"command": "clean", "params": {
            "file": xlsx(fname), "method": method
        }})
        assert succeed(result), f"clean({method}) failed for {fname}: {result}"


# ============================================================================
# transform command
# ============================================================================

class TestTransform:
    """Test transform command with Excel file input."""

    @pytest.mark.parametrize("fname,method", [
        ("023_dist_exponential.xlsx", "log"),
        ("021_dist_normal.xlsx", "sqrt"),
        ("024_dist_lognormal.xlsx", "boxcox"),
        ("022_dist_uniform.xlsx", "standardize"),
        ("021_dist_normal.xlsx", "rank"),
    ])
    def test_transform(self, fname, method):
        result = handler({"command": "transform", "params": {
            "file": xlsx(fname), "method": method
        }})
        assert succeed(result), f"transform({method}) failed for {fname}: {result}"


# ============================================================================
# ttest command (one-sample via generic loader)
# ============================================================================

class TestTtest:
    """Test ttest command with Excel file input."""

    @pytest.mark.parametrize("fname", [
        "021_dist_normal.xlsx",
        "002_singlecol_50rows.xlsx",
        "061_edge_all_negative.xlsx",
    ])
    def test_one_sample_ttest(self, fname):
        result = handler({"command": "ttest", "params": {
            "file": xlsx(fname), "test_type": "one_sample", "mu": 0
        }})
        assert succeed(result), f"ttest failed for {fname}: {result}"


# ============================================================================
# timeseries command
# ============================================================================

class TestTimeseries:
    """Test timeseries command with Excel file input."""

    @pytest.mark.parametrize("fname,analysis_type", [
        ("085_stat_timeseries.xlsx", "exp_smoothing"),
        ("085_stat_timeseries.xlsx", "decomposition"),
        ("085_stat_timeseries.xlsx", "acf"),
        ("003_singlecol_500rows.xlsx", "exp_smoothing"),
    ])
    def test_timeseries(self, fname, analysis_type):
        result = handler({"command": "timeseries", "params": {
            "file": xlsx(fname), "analysis_type": analysis_type
        }})
        assert succeed(result), f"timeseries({analysis_type}) failed for {fname}: {result}"


# ============================================================================
# regression (multiple) - direct file handling
# ============================================================================

class TestRegressionFile:
    """Test regression command with Excel file input (multiple/stepwise)."""

    def test_multiple_regression(self):
        result = handler({"command": "regression", "params": {
            "file": xlsx("099_stat_multi_regression.xlsx"),
            "x_columns": ["x1", "x2", "x3"],
            "y_column": "y",
            "reg_type": "multiple",
        }})
        assert succeed(result), f"multiple regression failed: {result}"

    def test_stepwise_regression(self):
        result = handler({"command": "regression", "params": {
            "file": xlsx("099_stat_multi_regression.xlsx"),
            "x_columns": ["x1", "x2", "x3"],
            "y_column": "y",
            "reg_type": "stepwise",
        }})
        assert succeed(result), f"stepwise regression failed: {result}"

    def test_regression_with_unicode_cols(self):
        result = handler({"command": "regression", "params": {
            "file": xlsx("055_mixed_unicode_cols.xlsx"),
            "x_columns": ["温度", "压力"],
            "y_column": "测量值",
            "reg_type": "multiple",
        }})
        assert succeed(result)


# ============================================================================
# multivariate (pca) - direct file handling
# ============================================================================

class TestMultivariateFile:
    """Test multivariate command with Excel file input."""

    def test_pca(self):
        result = handler({"command": "multivariate", "params": {
            "analysis_type": "pca",
            "file": xlsx("090_stat_pca.xlsx"),
            "columns": ["x1", "x2", "x3"],
        }})
        assert succeed(result), f"PCA failed: {result}"

    def test_pca_many_columns(self):
        result = handler({"command": "multivariate", "params": {
            "analysis_type": "pca",
            "file": xlsx("020_wide_100cols.xlsx"),
            "columns": [f"feature_{i}" for i in range(50)],
        }})
        assert succeed(result)

    def test_correlation_matrix(self):
        result = handler({"command": "multivariate", "params": {
            "analysis_type": "correlation_matrix",
            "file": xlsx("089_stat_correlated.xlsx"),
            "columns": ["x1", "x2", "x3"],
        }})
        assert succeed(result)

    def test_cluster(self):
        result = handler({"command": "multivariate", "params": {
            "analysis_type": "cluster",
            "file": xlsx("090_stat_pca.xlsx"),
            "columns": ["x1", "x2", "x3"],
            "n_clusters": 3,
        }})
        assert succeed(result)


# ============================================================================
# report command
# ============================================================================

class TestReport:
    """Test report command with Excel file input."""

    def test_report(self):
        result = handler({"command": "report", "params": {
            "file": xlsx("087_stat_capability.xlsx"), "usl": 11.0, "lsl": 9.0
        }})
        assert succeed(result)


# ============================================================================
# Batch coverage: every file tested with at least one command
# ============================================================================

class TestBatchCoverage:
    """Ensure every Excel fixture file is tested with at least one command."""

    @pytest.mark.parametrize("fname", [os.path.basename(f) for f in all_fixtures()])
    def test_file_loadable(self, fname):
        """Every file should be loadable via explore without crashing."""
        result = handler({"command": "explore", "params": {"file": xlsx(fname)}})
        assert result["status"] == "success", f"explore crashed for {fname}: {result}"


# ============================================================================
# Edge cases and error recovery
# ============================================================================

class TestEdgeCases:
    """Test error handling with problematic Excel files."""

    def test_all_text_file(self):
        """All-text file should return error, not crash."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("048_mixed_all_text.xlsx")
        }})
        assert result["status"] == "error"
        assert result["error_type"] == "DATA_ERROR"

    def test_mixed_messy_column(self):
        """Messy mixed-type column should handle gracefully."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("050_mixed_messy_column.xlsx")
        }})
        # Should succeed (numeric values extracted) or fail gracefully
        assert succeed(result)

    def test_inf_values(self):
        """Inf values should be filtered out."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("053_mixed_with_inf.xlsx")
        }})
        assert succeed(result)

    def test_empty_sheet(self):
        """Empty sheet should fail gracefully."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("073_sheet_empty_plus_data.xlsx"), "sheet": "Empty"
        }})
        assert result["status"] == "error"

    def test_wrong_sheet_name(self):
        """Non-existent sheet should fail gracefully."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("066_sheet_single.xlsx"), "sheet": "DoesNotExist"
        }})
        assert result["status"] == "error"

    def test_nonexistent_file(self):
        """Non-existent file should fail gracefully."""
        result = handler({"command": "descriptive", "params": {
            "file": "/tmp/nonexistent_file.xlsx"
        }})
        assert result["status"] == "error"

    def test_constant_values(self):
        """Constant values should not crash."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("031_dist_constant.xlsx")
        }})
        assert succeed(result)
        if result["status"] == "success":
            assert result["data"]["std"] == 0.0

    def test_single_row(self):
        """Single row should return partial result."""
        result = handler({"command": "descriptive", "params": {
            "file": xlsx("056_edge_single_row.xlsx")
        }})
        assert succeed(result)

    def test_duplicate_column_names(self):
        """Duplicate column names should be handled."""
        result = handler({"command": "explore", "params": {
            "file": xlsx("078_header_duplicate_names.xlsx")
        }})
        assert succeed(result)

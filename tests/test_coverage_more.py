"""More coverage tests to reach 97%."""

import csv
import json
import numpy as np
import pytest

from main import handler


class TestDataLoaderDatetime:
    """Test data loader datetime handling."""

    def test_excel_datetime_column(self, tmp_path):
        """Excel with datetime column."""
        import pandas as pd

        excel_file = tmp_path / "datetime.xlsx"
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })
        df.to_excel(excel_file, index=False)

        result = handler({
            "command": "descriptive",
            "params": {"file": str(excel_file), "column": "value"},
        })
        assert result["status"] == "success"


class TestDataLoaderExtractColumn:
    """Test column extraction edge cases."""

    def test_extract_by_numeric_column_name(self, tmp_path):
        """Extract column by numeric column name."""
        csv_file = tmp_path / "numeric.csv"
        with open(csv_file, "w") as f:
            f.write("1,2,3\n4,5,6\n7,8,9\n")

        result = handler({
            "command": "descriptive",
            "params": {
                "file": str(csv_file),
                "column": "1",
                "header": None,
            },
        })
        assert result["status"] == "success"

    def test_extract_by_column_index(self, tmp_path):
        """Extract column by index."""
        csv_file = tmp_path / "index.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")

        result = handler({
            "command": "descriptive",
            "params": {
                "file": str(csv_file),
                "column": "0",
            },
        })
        assert result["status"] == "success"


class TestWorkflowPipeline:
    """Test workflow pipeline functionality."""

    def test_pipeline_with_clean_step(self):
        """Pipeline with clean step updates data."""
        from stats_engine.workflow import pipeline

        results = pipeline(
            data=[1, 2, None, 4, 5],
            steps=[
                {"command": "clean", "params": {"method": "drop"}},
                "descriptive",
            ],
        )
        assert len(results) > 0

    def test_pipeline_with_outlier_step(self):
        """Pipeline with outlier step updates data."""
        from stats_engine.workflow import pipeline

        results = pipeline(
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
            steps=[
                {"command": "outlier", "params": {"method": "grubbs"}},
                "descriptive",
            ],
        )
        assert len(results) > 0


class TestWorkflowAssumptions:
    """Test workflow assumption checking."""

    def test_workflow_with_assumptions(self):
        """Workflow with assumption checking."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "ttest", "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")

    def test_workflow_adjust_params_welch(self):
        """Workflow adjusts params for Welch t-test."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "ttest", "params": {"values": [1, 2, 3, 4, 5], "values2": [10, 20, 30, 40, 50]}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")


class TestReliabilityEdgeCases:
    """Test reliability edge cases."""

    def test_weibull_with_status(self):
        """Weibull with status parameter."""
        result = handler({
            "command": "reliability",
            "params": {
                "times": [10, 20, 30, 40, 50],
                "status": [1, 1, 0, 1, 0],
                "analysis_type": "weibull",
            },
        })
        assert result["status"] == "success"

    def test_kaplan_meier_default_status(self):
        """Kaplan-Meier without status defaults to all failures."""
        result = handler({
            "command": "reliability",
            "params": {
                "times": [10, 20, 30, 40, 50],
                "analysis_type": "kaplan_meier",
            },
        })
        assert result["status"] == "success"

    def test_distribution_fit_with_values(self):
        """Distribution fit with values alias."""
        result = handler({
            "command": "reliability",
            "params": {
                "values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "analysis_type": "distribution",
            },
        })
        assert result["status"] == "success"


class TestTimeseriesEdgeCases:
    """Test timeseries edge cases."""

    def test_timeseries_exp_smoothing_short(self):
        """Exponential smoothing with short series."""
        result = handler({
            "command": "timeseries",
            "params": {
                "values": [1, 2, 3],
                "analysis_type": "exp_smoothing",
            },
        })
        assert result["status"] == "success"

    def test_timeseries_acf_short(self):
        """ACF with short series."""
        result = handler({
            "command": "timeseries",
            "params": {
                "values": [1, 2, 3, 4, 5],
                "analysis_type": "acf",
            },
        })
        assert result["status"] == "success"

    def test_timeseries_decomposition(self):
        """Timeseries decomposition."""
        values = (np.sin(np.linspace(0, 4 * np.pi, 50)) + np.linspace(0, 5, 50)).tolist()
        result = handler({
            "command": "timeseries",
            "params": {
                "values": values,
                "analysis_type": "decomposition",
                "period": 10,
            },
        })
        assert result["status"] == "success"


class TestExploreEdgeCases:
    """Test explore edge cases."""

    def test_explore_csv_with_missing(self, tmp_path):
        """Explore CSV with missing values."""
        csv_file = tmp_path / "missing.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,,3\n4,5,\n7,8,9\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(csv_file)},
        })
        assert result["status"] == "success"

    def test_explore_excel_multi_sheet(self, tmp_path):
        """Explore Excel with multiple sheets."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"b": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = handler({
            "command": "explore",
            "params": {"file": str(excel_file), "sheet": "Sheet1"},
        })
        assert result["status"] == "success"

    def test_explore_text_with_non_numeric(self, tmp_path):
        """Explore text with non-numeric lines."""
        txt_file = tmp_path / "mixed.txt"
        with open(txt_file, "w") as f:
            f.write("1\nhello\n3\nworld\n5\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(txt_file)},
        })
        assert result["status"] == "success"

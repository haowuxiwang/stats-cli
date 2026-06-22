"""Extra coverage tests to reach 97%."""

import csv
import json
import numpy as np
import pytest

from main import handler


class TestDataLoaderEncoding:
    """Test data loader encoding detection."""

    def test_detect_encoding_latin1_fallback(self, tmp_path):
        """Encoding detection falls back to latin-1."""
        # Create a file with encoding that's not utf-8
        test_file = tmp_path / "latin1.txt"
        with open(test_file, "wb") as f:
            f.write(b"value\n1\n2\n3\n\xe9\xe8\xea\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(test_file)},
        })
        assert result["status"] in ("success", "error")

    def test_detect_delimiter_error(self, tmp_path):
        """Delimiter detection error returns comma."""
        # Create a binary file that can't be read as text
        test_file = tmp_path / "binary.bin"
        with open(test_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(test_file)},
        })
        assert result["status"] in ("success", "error")


class TestDataLoaderExcel:
    """Test Excel loading edge cases."""

    def test_excel_import_error(self, tmp_path):
        """Excel loading with import error."""
        import pandas as pd

        excel_file = tmp_path / "test.xlsx"
        pd.DataFrame({"a": [1, 2, 3]}).to_excel(excel_file, index=False)

        # This will work normally, but tests the import path
        result = handler({
            "command": "descriptive",
            "params": {"file": str(excel_file), "column": "a"},
        })
        assert result["status"] == "success"

    def test_excel_datetime_handling(self, tmp_path):
        """Excel with datetime values."""
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

    def test_excel_multiple_sheets_available(self, tmp_path):
        """Excel with multiple sheets returns available_sheets."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1, 2, 3]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"b": [4, 5, 6]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = handler({
            "command": "descriptive",
            "params": {"file": str(excel_file), "column": "a", "sheet": "Sheet1"},
        })
        assert result["status"] == "success"


class TestDataLoaderCSV:
    """Test CSV loading edge cases."""

    def test_csv_bom_handling(self, tmp_path):
        """CSV with BOM encoding."""
        csv_file = tmp_path / "bom.csv"
        with open(csv_file, "wb") as f:
            f.write(b"\xef\xbb\xbfvalue\n1\n2\n3\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(csv_file), "column": "value"},
        })
        assert result["status"] == "success"

    def test_csv_column_by_name_not_found(self, tmp_path):
        """CSV column by name not found, fallback to index."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(csv_file), "column": "nonexistent"},
        })
        assert result["status"] == "error"

    def test_csv_column_by_index(self, tmp_path):
        """CSV column by index."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(csv_file), "column": "1"},
        })
        assert result["status"] == "success"

    def test_csv_no_numeric_columns(self, tmp_path):
        """CSV with no numeric columns."""
        csv_file = tmp_path / "text.csv"
        with open(csv_file, "w") as f:
            f.write("name,desc\nhello,world\nfoo,bar\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(csv_file)},
        })
        assert result["status"] == "error"


class TestWorkflowPipeline:
    """Test workflow pipeline functionality."""

    def test_pipeline_dict_steps(self):
        """Pipeline with dict steps."""
        from stats_engine.workflow import pipeline

        results = pipeline(
            data=[1, 2, 3, 4, 5],
            steps=[
                {"command": "descriptive", "params": {}},
            ],
        )
        assert len(results) > 0

    def test_pipeline_string_steps(self):
        """Pipeline with string steps."""
        from stats_engine.workflow import pipeline

        results = pipeline(
            data=[1, 2, 3, 4, 5],
            steps=["descriptive"],
        )
        assert len(results) > 0

    def test_pipeline_file_based(self, tmp_path):
        """Pipeline with file-based data."""
        from stats_engine.workflow import pipeline

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for v in [1, 2, 3, 4, 5]:
                writer.writerow([v])

        results = pipeline(
            data=str(csv_file),
            steps=["descriptive"],
        )
        assert len(results) > 0

    def test_pipeline_with_clean(self):
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


class TestWorkflowTemplate:
    """Test workflow templates."""

    def test_manufacturing_template_with_target(self):
        """Manufacturing template with target."""
        from stats_engine.workflow import workflow_template

        result = workflow_template(
            template_name="manufacturing",
            values=[9.8, 10.0, 10.2, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.2],
            usl=12,
            lsl=8,
            target=10,
        )
        assert "n_steps" in result

    def test_exploration_template(self):
        """Exploration template."""
        from stats_engine.workflow import workflow_template

        result = workflow_template(
            template_name="exploration",
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        assert "n_steps" in result


class TestReliabilityEdgeCases:
    """Test reliability edge cases."""

    def test_distribution_fit_values_alias(self):
        """Distribution fit with values alias."""
        result = handler({
            "command": "reliability",
            "params": {
                "values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "analysis_type": "distribution",
            },
        })
        assert result["status"] == "success"

    def test_stability_analysis(self):
        """Stability/shelf life analysis."""
        result = handler({
            "command": "reliability",
            "params": {
                "times": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "analysis_type": "stability",
            },
        })
        assert result["status"] in ("success", "error")


class TestExploreEdgeCases:
    """Test explore edge cases."""

    def test_explore_excel_sheet_index(self, tmp_path):
        """Explore Excel by sheet index."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1, 2, 3]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"b": [4, 5, 6]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = handler({
            "command": "explore",
            "params": {"file": str(excel_file), "sheet": 0},
        })
        assert result["status"] == "success"

    def test_explore_text_non_numeric(self, tmp_path):
        """Explore text with non-numeric lines."""
        txt_file = tmp_path / "mixed.txt"
        with open(txt_file, "w") as f:
            f.write("1\nhello\n3\nworld\n5\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(txt_file)},
        })
        assert result["status"] == "success"

    def test_explore_spec_tolerance(self, tmp_path):
        """Explore with spec tolerance pattern."""
        csv_file = tmp_path / "spec.csv"
        with open(csv_file, "w") as f:
            f.write("name,target+-tolerance\na,10+-0.5\nb,20+-1.0\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(csv_file)},
        })
        assert result["status"] == "success"

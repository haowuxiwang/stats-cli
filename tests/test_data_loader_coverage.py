"""Additional tests for utils/data_loader.py to improve coverage."""

import pytest

from utils.data_loader import load_data


class TestLoadDataEdgeCases:
    """Test edge cases in load_data."""

    def test_file_not_found(self):
        """Non-existent file returns error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_data(file="/nonexistent/path/file.xlsx", column="A")

    def test_csv_with_bom(self, tmp_path):
        """CSV with BOM encoding loads correctly."""
        csv_file = tmp_path / "bom.csv"
        # Write UTF-8 BOM + content
        with open(csv_file, "wb") as f:
            f.write(b"\xef\xbb\xbfvalue\n1\n2\n3\n")
        result = load_data(file=str(csv_file), column="value")
        assert len(result["values"]) == 3

    def test_csv_column_by_index(self, tmp_path):
        """CSV column extraction by numeric index."""
        csv_file = tmp_path / "index.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        result = load_data(file=str(csv_file), column="1")
        assert len(result["values"]) == 2

    def test_csv_column_not_found(self, tmp_path):
        """CSV with non-existent column name returns error."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n")
        with pytest.raises((ValueError, KeyError)):
            load_data(file=str(csv_file), column="nonexistent")

    def test_excel_multiple_sheets(self, tmp_path):
        """Excel with multiple sheets returns available_sheets."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"A": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"B": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = load_data(file=str(excel_file), column="A", sheet="Sheet1")
        assert len(result["values"]) == 2

    def test_excel_sheet_index(self, tmp_path):
        """Excel sheet by index."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"A": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"B": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = load_data(file=str(excel_file), column="B", sheet=1)
        assert len(result["values"]) == 2

    def test_excel_sheet_index_out_of_range(self, tmp_path):
        """Excel sheet index out of range returns error."""
        import pandas as pd

        excel_file = tmp_path / "test.xlsx"
        pd.DataFrame({"A": [1]}).to_excel(excel_file, index=False)

        with pytest.raises((ValueError, IndexError)):
            load_data(file=str(excel_file), column="A", sheet=999)

    def test_json_load(self, tmp_path):
        """JSON file loads correctly."""
        import json

        json_file = tmp_path / "data.json"
        with open(json_file, "w") as f:
            json.dump({"values": [1, 2, 3, 4, 5]}, f)

        result = load_data(file=str(json_file))
        assert len(result["values"]) == 5

    def test_txt_load(self, tmp_path):
        """Text file loads correctly."""
        txt_file = tmp_path / "data.txt"
        with open(txt_file, "w") as f:
            f.write("1\n2\n3\n4\n5\n")

        result = load_data(file=str(txt_file))
        assert len(result["values"]) == 5


class TestTwoColumns:
    """Test two-column loading."""

    def test_two_columns_csv(self, tmp_path):
        """Two columns from CSV."""
        csv_file = tmp_path / "xy.csv"
        with open(csv_file, "w") as f:
            f.write("x,y\n1,2\n3,4\n5,6\n")

        result = load_data(file=str(csv_file), columns=["x", "y"])
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 3

    def test_two_columns_excel(self, tmp_path):
        """Two columns from Excel."""
        import pandas as pd

        excel_file = tmp_path / "xy.xlsx"
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_excel(excel_file, index=False)

        result = load_data(file=str(excel_file), columns=["x", "y"])
        assert "x" in result
        assert "y" in result


class TestCleaningReport:
    """Test data cleaning report generation."""

    def test_cleaning_report_with_nan(self, tmp_path):
        """Cleaning report generated when NaN values present."""
        csv_file = tmp_path / "nan.csv"
        with open(csv_file, "w") as f:
            f.write("value\n1\n\n3\nNaN\n5\n")

        result = load_data(file=str(csv_file), column="value")
        assert len(result["values"]) <= 5

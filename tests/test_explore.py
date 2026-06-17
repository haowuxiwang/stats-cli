"""Tests for stats_engine/explore.py."""

import json

import pandas as pd
import pytest

from stats_engine.explore import explore


class TestExploreCSV:
    @pytest.fixture
    def csv_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_basic_csv(self, csv_file):
        result = explore(file=csv_file)
        assert result["n_rows"] == 3
        assert result["n_columns"] == 2
        assert "columns" in result
        assert "sample_data" in result

    def test_csv_column_info(self, csv_file):
        result = explore(file=csv_file)
        col_names = [c["name"] for c in result["columns"]]
        assert "a" in col_names
        assert "b" in col_names

    def test_csv_numeric_columns(self, csv_file):
        result = explore(file=csv_file)
        assert "a" in result["numeric_columns"]
        assert "b" in result["numeric_columns"]


class TestExploreExcel:
    @pytest.fixture
    def xlsx_file(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        p = tmp_path / "test.xlsx"
        df.to_excel(p, index=False)
        return str(p)

    def test_basic_xlsx(self, xlsx_file):
        result = explore(file=xlsx_file)
        assert result["n_rows"] == 3
        assert result["n_columns"] == 2
        assert result["sheet"] == "Sheet1"
        assert result["n_sheets"] == 1

    def test_xlsx_sample_data(self, xlsx_file):
        result = explore(file=xlsx_file)
        assert len(result["sample_data"]) == 3


class TestExploreJSON:
    @pytest.fixture
    def json_dict_file(self, tmp_path):
        data = {"values": [1, 2, 3, 4, 5], "name": "test"}
        p = tmp_path / "test.json"
        with open(p, "w") as f:
            json.dump(data, f)
        return str(p)

    @pytest.fixture
    def json_list_file(self, tmp_path):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        p = tmp_path / "test_list.json"
        with open(p, "w") as f:
            json.dump(data, f)
        return str(p)

    def test_json_dict(self, json_dict_file):
        result = explore(file=json_dict_file)
        assert result["format"] == "json"
        assert result["type"] == "dict"
        assert "values" in result["keys"]

    def test_json_list(self, json_list_file):
        result = explore(file=json_list_file)
        assert result["format"] == "json"
        assert result["type"] == "list"
        assert result["n_items"] == 3


class TestExploreText:
    @pytest.fixture
    def text_file(self, tmp_path):
        p = tmp_path / "test.txt"
        with open(p, "w") as f:
            f.write("1.0\n2.0\n3.0\n4.0\n5.0\n")
        return str(p)

    def test_text_file(self, text_file):
        result = explore(file=text_file)
        assert result["format"] == "text"
        assert result["n_numeric"] == 5
        assert result["n_non_numeric"] == 0

    def test_text_basic_stats(self, text_file):
        result = explore(file=text_file)
        assert "basic_stats" in result
        assert result["basic_stats"]["mean"] == 3.0


class TestExploreErrors:
    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            explore(file="/tmp/nonexistent_xyz_12345.xlsx")

    def test_invalid_csv(self, tmp_path):
        p = tmp_path / "bad.csv"
        with open(p, "w") as f:
            f.write(",,,")
        result = explore(file=str(p))
        # Should still return something without crashing
        assert "n_rows" in result or "error" in result


class TestExploreOptions:
    @pytest.fixture
    def csv_file(self, tmp_path):
        df = pd.DataFrame({"x": range(100), "y": [i * 2 for i in range(100)], "z": ["cat"] * 50 + ["dog"] * 50})
        p = tmp_path / "large.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_custom_rows(self, csv_file):
        result = explore(file=csv_file, rows=3)
        assert len(result["sample_data"]) == 3

    def test_default_rows(self, csv_file):
        result = explore(file=csv_file)
        assert len(result["sample_data"]) == 5


def test_explore_spec_detection():
    """Explore detects USL/LSL columns."""
    import os
    import tempfile

    # Create a CSV with spec limit columns
    csv_content = "value,USL,LSL\n10,12,8\n11,12,8\n10.5,12,8\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmpfile = f.name
    try:
        from stats_engine.explore import explore

        result = explore(file=tmpfile)
        assert "detected_specs" in result
        assert "usl" in result["detected_specs"]
        assert result["detected_specs"]["usl"] == 12
        assert result["detected_specs"]["lsl"] == 8
    finally:
        os.unlink(tmpfile)


def test_explore_no_spec_columns():
    """Explore with no spec columns returns no detected_specs."""
    import os
    import tempfile

    csv_content = "x,y\n1,2\n3,4\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmpfile = f.name
    try:
        from stats_engine.explore import explore

        result = explore(file=tmpfile)
        assert "detected_specs" not in result
    finally:
        os.unlink(tmpfile)


def test_explore_spec_target_column():
    """Explore detects Target column."""
    import os
    import tempfile

    csv_content = "value,Target\n10,10\n11,10\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmpfile = f.name
    try:
        from stats_engine.explore import explore

        result = explore(file=tmpfile)
        assert "detected_specs" in result
        assert "target" in result["detected_specs"]
    finally:
        os.unlink(tmpfile)

"""Additional tests for stats_engine/explore.py to improve coverage."""

import pytest

from main import handler


class TestExploreEdgeCases:
    """Test explore edge cases."""

    def test_explore_csv(self, tmp_path):
        """Explore CSV file."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(csv_file)},
        })
        assert result["status"] == "success"
        assert "columns" in result["data"] or "n_rows" in result["data"]

    def test_explore_excel(self, tmp_path):
        """Explore Excel file."""
        import pandas as pd

        excel_file = tmp_path / "data.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(excel_file, index=False)

        result = handler({
            "command": "explore",
            "params": {"file": str(excel_file)},
        })
        assert result["status"] == "success"

    def test_explore_json(self, tmp_path):
        """Explore JSON file."""
        import json

        json_file = tmp_path / "data.json"
        with open(json_file, "w") as f:
            json.dump({"values": [1, 2, 3, 4, 5]}, f)

        result = handler({
            "command": "explore",
            "params": {"file": str(json_file)},
        })
        assert result["status"] == "success"

    def test_explore_text(self, tmp_path):
        """Explore text file."""
        txt_file = tmp_path / "data.txt"
        with open(txt_file, "w") as f:
            f.write("1\n2\n3\n4\n5\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(txt_file)},
        })
        assert result["status"] == "success"

    def test_explore_text_with_non_numeric(self, tmp_path):
        """Explore text file with non-numeric lines."""
        txt_file = tmp_path / "mixed.txt"
        with open(txt_file, "w") as f:
            f.write("1\nhello\n3\nworld\n5\n")

        result = handler({
            "command": "explore",
            "params": {"file": str(txt_file)},
        })
        assert result["status"] == "success"

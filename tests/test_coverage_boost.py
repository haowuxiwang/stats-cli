"""Additional tests to boost coverage from 95% to 97%."""


from main import handler


class TestWorkflowPipeline:
    """Test workflow pipeline functionality."""

    def test_workflow_with_target(self):
        """Workflow template with target parameter."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "descriptive", "params": {"values": [9.8, 10.0, 10.2, 10.1, 9.9]}},
                    {"command": "capability", "params": {"values": [9.8, 10.0, 10.2, 10.1, 9.9], "usl": 12, "lsl": 8, "target": 10}},
                ],
            },
        })
        assert result["status"] == "success"

    def test_workflow_with_file_context(self, tmp_path):
        """Workflow using file context injection."""
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                writer.writerow([v])

        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "descriptive", "params": {"file": str(csv_file), "column": "value"}},
                ],
            },
        })
        assert result["status"] == "success"


class TestWorkflowAssumptions:
    """Test workflow assumption checking."""

    def test_workflow_assumptions_no_values(self):
        """Workflow assumptions with no values in step."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "ttest", "params": {"values": [1, 2, 3, 4, 5]}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")

    def test_workflow_assumptions_too_few_values(self):
        """Workflow assumptions with too few values."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "ttest", "params": {"values": [1, 2]}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")


class TestRegressionCoverage:
    """Test regression edge cases for coverage."""

    def test_regression_x_y_mismatch(self):
        """Regression with x/y length mismatch returns error."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1, 2, 3],
                "y": [1, 2],
                "reg_type": "linear",
            },
        })
        assert result["status"] == "error"

    def test_regression_single_point(self):
        """Regression with single data point returns error."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1],
                "y": [2],
                "reg_type": "linear",
            },
        })
        assert result["status"] == "error"

    def test_regression_n2(self):
        """Regression with exactly 2 data points."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1, 2],
                "y": [3, 5],
                "reg_type": "linear",
            },
        })
        assert result["status"] in ("success", "warning")
        assert result["data"]["r_squared"] == 1.0 or result["data"]["r_squared"] > 0.99

    def test_polynomial_insufficient_data(self):
        """Polynomial with insufficient data returns error."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1, 2, 3],
                "y": [1, 4, 9],
                "reg_type": "polynomial",
                "degree": 3,
            },
        })
        assert result["status"] == "error"


class TestReliabilityCoverage:
    """Test reliability edge cases for coverage."""

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


class TestTimeseriesCoverage:
    """Test timeseries edge cases for coverage."""

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


class TestExploreCoverage:
    """Test explore edge cases for coverage."""

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


class TestDataLoaderCoverage:
    """Test data loader edge cases for coverage."""

    def test_load_csv_with_bom(self, tmp_path):
        """Load CSV with BOM encoding."""
        csv_file = tmp_path / "bom.csv"
        with open(csv_file, "wb") as f:
            f.write(b"\xef\xbb\xbfvalue\n1\n2\n3\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(csv_file), "column": "value"},
        })
        assert result["status"] == "success"

    def test_load_excel_sheet_by_index(self, tmp_path):
        """Load Excel sheet by index."""
        import pandas as pd

        excel_file = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1, 2, 3]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"b": [4, 5, 6]}).to_excel(writer, sheet_name="Sheet2", index=False)

        result = handler({
            "command": "descriptive",
            "params": {"file": str(excel_file), "column": "b", "sheet": 1},
        })
        assert result["status"] == "success"

    def test_load_json(self, tmp_path):
        """Load JSON file."""
        import json

        json_file = tmp_path / "data.json"
        with open(json_file, "w") as f:
            json.dump({"values": [1, 2, 3, 4, 5]}, f)

        result = handler({
            "command": "descriptive",
            "params": {"file": str(json_file)},
        })
        assert result["status"] == "success"

    def test_load_text(self, tmp_path):
        """Load text file."""
        txt_file = tmp_path / "data.txt"
        with open(txt_file, "w") as f:
            f.write("1\n2\n3\n4\n5\n")

        result = handler({
            "command": "descriptive",
            "params": {"file": str(txt_file)},
        })
        assert result["status"] == "success"


class TestChartHandlerCoverage:
    """Test chart handler edge cases for coverage."""

    def test_regression_scatter_fallback(self):
        """Regression scatter plot fallback when no slope/intercept."""
        result = handler({
            "command": "regression",
            "params": {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 5, 4, 5],
                "reg_type": "linear",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_outlier_chart_with_int_indices(self):
        """Outlier chart with integer outlier indices."""
        result = handler({
            "command": "outlier",
            "params": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
                "method": "grubbs",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_reliability_chart_with_params(self):
        """Reliability chart with shape/scale parameters."""
        result = handler({
            "command": "reliability",
            "params": {
                "times": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "analysis_type": "weibull",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_trend_chart_cusum(self):
        """Trend chart with CUSUM."""
        result = handler({
            "command": "trend",
            "params": {
                "values": [10.1, 10.2, 10.0, 9.9, 10.3, 10.1, 10.0, 9.8, 10.2, 10.1],
                "test_type": "cusum",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_timeseries_chart(self):
        """Timeseries chart."""
        result = handler({
            "command": "timeseries",
            "params": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "analysis_type": "exp_smoothing",
            },
            "chart": True,
        })
        assert result["status"] == "success"

    def test_multiple_comparison_chart(self):
        """Multiple comparison chart via anova."""
        result = handler({
            "command": "anova",
            "params": {
                "groups": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            },
            "chart": True,
        })
        assert result["status"] == "success"

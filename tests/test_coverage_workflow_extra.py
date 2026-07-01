"""Coverage-boosting tests for explore.py, workflow.py, control_chart.py edge cases."""

from main import handler


class TestExploreEdgeCases:
    """Target explore.py missed lines: 187-195 (tolerance detection)."""

    def test_explore_with_tolerance_column(self):
        """Lines 187-195: tolerance pattern like '10.0 +/- 0.5'."""
        import csv
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Batch", "10.0 +/- 0.5"])
            for i in range(10):
                writer.writerow([f"B{i}", f"{10 + i * 0.1:.2f}"])
            tmpfile = f.name

        try:
            result = handler({"command": "explore", "params": {"file": tmpfile}})
            assert result["status"] == "success"
        finally:
            os.unlink(tmpfile)


class TestWorkflowEdgeCases:
    """Target workflow.py missed lines: 42,86-87,114,120,138-140,145,154."""

    def test_workflow_with_auto_check(self):
        """Line 45: auto_check assumptions."""
        result = handler(
            {
                "command": "workflow",
                "params": {
                    "steps": [
                        {"command": "descriptive", "params": {}},
                        {"command": "ttest", "params": {"test_type": "one_sample", "population_mean": 10}},
                    ],
                    "values": [10.1, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.1],
                    "auto_check": True,
                },
            }
        )
        assert result["status"] == "success"
        assert "assumptions" in result["data"]

    def test_workflow_step_error(self):
        """Lines 86-87: step execution error."""
        result = handler(
            {
                "command": "workflow",
                "params": {
                    "steps": [
                        {"command": "descriptive", "params": {"values": [1, 2, 3]}},
                        {"command": "ttest", "params": {"test_type": "two_sample"}},  # Missing values2
                    ],
                },
            }
        )
        assert result["status"] == "success"
        # Second step should be error
        steps = result["data"]["steps_results"]
        assert steps[1]["status"] == "error"

    def test_workflow_template(self):
        """Line 114: workflow_template."""
        result = handler(
            {
                "command": "workflow_template",
                "params": {"template_name": "capability", "values": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10], "usl": 15, "lsl": 5},
            }
        )
        assert result["status"] == "success"

    def test_workflow_with_data_chaining(self):
        """Lines 138-140, 145, 154: context update after clean/transform."""
        result = handler(
            {
                "command": "workflow",
                "params": {
                    "steps": [
                        {"command": "clean", "params": {}},
                        {"command": "descriptive", "params": {}},
                    ],
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
            }
        )
        assert result["status"] == "success"


class TestControlChartEdgeCases:
    """Target control_chart.py missed lines. Uses correct chart_type values."""

    def test_cusum_chart(self):
        """CUSUM chart."""
        result = handler(
            {
                "command": "control_chart",
                "params": {
                    "chart_type": "cusum",
                    "values": [10, 12, 11, 13, 10, 14, 12, 11, 13, 10, 12, 14, 11, 13, 10],
                },
            }
        )
        assert result["status"] == "success"

    def test_ewma_chart(self):
        """EWMA chart."""
        result = handler(
            {
                "command": "control_chart",
                "params": {
                    "chart_type": "ewma",
                    "values": [10, 12, 11, 13, 10, 14, 12, 11, 13, 10, 12, 14, 11, 13, 10],
                },
            }
        )
        assert result["status"] == "success"

    def test_hotelling_t2_chart(self):
        """Hotelling T2 (multivariate) chart."""
        result = handler(
            {
                "command": "control_chart",
                "params": {
                    "chart_type": "hotelling_t2",
                    "values": [[1, 2], [2, 3], [1, 3], [2, 2], [3, 4], [2, 3], [1, 2], [3, 3], [2, 4], [1, 3]],
                },
            }
        )
        # May warn on small subgroups
        assert result["status"] in ("success", "warning")

    def test_p_chart(self):
        """p-chart (attribute) path."""
        result = handler(
            {
                "command": "control_chart",
                "params": {
                    "chart_type": "p",
                    "values": [5, 3, 7, 4, 6, 2, 5, 4, 3, 6],
                    "subgroup_size": 100,
                },
            }
        )
        assert result["status"] == "success"

    def test_xbar_chart(self):
        """X-bar/R chart."""
        result = handler(
            {
                "command": "control_chart",
                "params": {
                    "chart_type": "xbar",
                    "values": [10, 12, 11, 13, 10, 14, 12, 11, 13, 10, 12, 14],
                    "subgroup_size": 3,
                },
            }
        )
        assert result["status"] == "success"

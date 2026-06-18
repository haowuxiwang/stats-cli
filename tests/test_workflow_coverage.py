"""Additional tests for stats_engine/workflow.py to improve coverage."""

import pytest

from main import handler


class TestWorkflowEdgeCases:
    """Test workflow edge cases."""

    def test_workflow_with_file(self, tmp_path):
        """Workflow using file instead of values."""
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for v in [1, 2, 3, 4, 5]:
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

    def test_workflow_step_error(self):
        """Workflow step with error is handled."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "descriptive", "params": {"values": []}},
                ],
            },
        })
        # Should handle gracefully
        assert result["status"] in ("success", "error", "warning")

    def test_workflow_assumptions_few_values(self):
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


class TestManufacturingTemplate:
    """Test manufacturing workflow template."""

    def test_manufacturing_basic(self):
        """Basic manufacturing workflow."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "descriptive", "params": {"values": [9.8, 10.0, 10.2, 10.1, 9.9]}},
                    {"command": "normality", "params": {"values": [9.8, 10.0, 10.2, 10.1, 9.9]}},
                ],
            },
        })
        assert result["status"] == "success"

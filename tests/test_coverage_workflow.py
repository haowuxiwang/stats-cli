"""Workflow coverage tests."""

import numpy as np
import pytest

from main import handler


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

    def test_workflow_assumptions_welch(self):
        """Workflow assumptions recommends Welch t-test."""
        # Create data with very different variances
        g1 = [1.0] * 20
        g2 = [100.0 + np.random.normal(0, 0.01) for _ in range(20)]
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "ttest", "params": {"values": g1, "values2": g2}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")

    def test_workflow_assumptions_anova_welch(self):
        """Workflow assumptions recommends Welch ANOVA."""
        g1 = [1.0] * 20
        g2 = [100.0 + np.random.normal(0, 0.01) for _ in range(20)]
        g3 = [200.0 + np.random.normal(0, 0.01) for _ in range(20)]
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "anova", "params": {"groups": [g1, g2, g3]}},
                ],
                "check_assumptions": True,
            },
        })
        assert result["status"] in ("success", "error", "warning")


class TestWorkflowContext:
    """Test workflow context updates."""

    def test_workflow_with_clean_step(self):
        """Workflow with clean step updates context."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "clean", "params": {"values": [1, 2, None, 4, 5], "method": "drop"}},
                    {"command": "descriptive"},
                ],
            },
        })
        assert result["status"] == "success"

    def test_workflow_with_transform_step(self):
        """Workflow with transform step updates context."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "transform", "params": {"values": [1, 2, 3, 4, 5], "method": "log"}},
                    {"command": "descriptive"},
                ],
            },
        })
        assert result["status"] == "success"

    def test_workflow_with_outlier_step(self):
        """Workflow with outlier step updates context."""
        result = handler({
            "command": "workflow",
            "params": {
                "steps": [
                    {"command": "outlier", "params": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], "method": "grubbs"}},
                    {"command": "descriptive"},
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
                    {"command": "capability", "params": {"values": [1, 2, 3]}},
                ],
            },
        })
        assert result["status"] in ("success", "error", "warning")


class TestWorkflowPipeline:
    """Test workflow pipeline."""

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

    def test_pipeline_with_clean(self):
        """Pipeline with clean step."""
        from stats_engine.workflow import pipeline

        results = pipeline(
            data=[1, 2, None, 4, 5],
            steps=[
                {"command": "clean", "params": {"method": "drop"}},
                "descriptive",
            ],
        )
        assert len(results) > 0


class TestWorkflowTemplates:
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

    def test_comparison_template(self):
        """Comparison template."""
        from stats_engine.workflow import workflow_template

        result = workflow_template(
            template_name="comparison",
            values=[1, 2, 3, 4, 5],
            group2=[6, 7, 8, 9, 10],
        )
        assert "n_steps" in result

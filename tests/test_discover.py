"""Tests for stats_engine/discover.py."""
import json

from stats_engine.discover import COMMANDS


class TestCommandsStructure:
    def test_commands_is_dict(self):
        assert isinstance(COMMANDS, dict)
        assert len(COMMANDS) >= 26

    def test_all_commands_have_required_fields(self):
        required_fields = {"description", "category", "input", "output_fields", "example"}
        for name, cmd in COMMANDS.items():
            missing = required_fields - set(cmd.keys())
            assert not missing, f"{name}: missing fields {missing}"

    def test_all_commands_have_valid_category(self):
        valid_categories = {
            "basic", "hypothesis", "spc", "regression",
            "doe", "msa", "reliability", "multivariate",
            "advanced", "data", "report", "workflow"
        }
        for name, cmd in COMMANDS.items():
            assert cmd["category"] in valid_categories, f"{name}: invalid category '{cmd['category']}'"

    def test_input_is_list(self):
        for name, cmd in COMMANDS.items():
            assert isinstance(cmd["input"], list), f"{name}: input should be list"

    def test_output_fields_is_list(self):
        for name, cmd in COMMANDS.items():
            assert isinstance(cmd["output_fields"], list), f"{name}: output_fields should be list"


class TestRequiredCommands:
    REQUIRED_COMMANDS = [
        "descriptive", "normality", "outlier",
        "ttest", "anova", "nonparametric", "homogeneity",
        "equivalence", "multiple_comparison", "power",
        "control_chart", "capability", "trend",
        "regression", "correlation",
        "doe", "gage_rr",
        "reliability", "multivariate", "timeseries",
        "explore", "clean", "transform", "report",
        "discover", "advanced", "run",
    ]

    def test_all_required_commands_exist(self):
        for cmd in self.REQUIRED_COMMANDS:
            assert cmd in COMMANDS, f"Missing required command: {cmd}"


class TestExamples:
    # Some examples use [...] as placeholders which is not valid JSON
    SKIP_PARSE = {
        "capability", "outlier", "control_chart", "trend", "doe", "gage_rr",
        "reliability", "multivariate", "clean", "transform", "report", "run"
    }

    def test_all_examples_have_valid_json_or_placeholder(self):
        for name, cmd in COMMANDS.items():
            example = cmd["example"]
            if name in self.SKIP_PARSE:
                # These use [...] placeholder; just verify structure
                assert '"command"' in example, f"{name}: example missing 'command'"
                assert f'"command": "{name}"' in example or f'"command":"{name}"' in example, \
                    f"{name}: example command doesn't match key"
            else:
                parsed = json.loads(example)
                assert "command" in parsed, f"{name}: example missing 'command' field"
                assert parsed["command"] == name, f"{name}: example command '{parsed['command']}' doesn't match key"

    def test_all_examples_have_command_field(self):
        for name, cmd in COMMANDS.items():
            assert '"command"' in cmd["example"], f"{name}: example missing 'command'"

    def test_example_command_matches_key(self):
        for name, cmd in COMMANDS.items():
            assert f'"command": "{name}"' in cmd["example"], f"{name}: example command doesn't match key"


class TestCategoryCommands:
    def test_basic_commands(self):
        basic = [k for k, v in COMMANDS.items() if v["category"] == "basic"]
        assert "descriptive" in basic
        assert "normality" in basic
        assert "outlier" in basic

    def test_hypothesis_commands(self):
        hyp = [k for k, v in COMMANDS.items() if v["category"] == "hypothesis"]
        assert "ttest" in hyp
        assert "anova" in hyp
        assert "power" in hyp

    def test_spc_commands(self):
        spc = [k for k, v in COMMANDS.items() if v["category"] == "spc"]
        assert "control_chart" in spc
        assert "capability" in spc
        assert "trend" in spc

    def test_data_commands(self):
        data = [k for k, v in COMMANDS.items() if v["category"] == "data"]
        assert "explore" in data
        assert "clean" in data
        assert "transform" in data
        assert "discover" in data

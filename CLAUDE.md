# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

stats-cli-py is a Python statistical analysis CLI/library for manufacturing and quality engineering. It provides 26 statistical commands (descriptive stats, hypothesis testing, SPC, capability, DOE, MSA, reliability, etc.) designed to be called as an AI-agent skill. All I/O is JSON.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with JSON from stdin
echo '{"command":"descriptive","params":{"values":[1,2,3]}}' | python main.py

# Run with JSON from file
python main.py input.json

# Run tests (pytest-based, tests/ directory)
pytest
pytest tests/test_specific.py          # single test file
pytest tests/test_specific.py::test_fn  # single test
```

## Architecture

**Entry point:** `main.py` — `handler(input_data)` parses JSON, routes via `_route(command, params)`, wraps output in standard envelope.

**Command router:** `_route()` in `main.py` uses lazy imports (imports inside each `if/elif` branch) to avoid loading heavy scientific libraries unnecessarily. Each command maps 1:1 to a module function in `stats_engine/`.

**Standard output envelope** (defined in `utils/output.py`):
- Success: `{"status": "success", "version": "1.0.0", "timestamp": "...", "data": {...}}`
- Error: `{"status": "error", "error_type": "...", "message": "...", "suggestion": "..."}`

**Data flow:** Input provides either `values` (direct array) or `file` (path). `utils/data_loader.py` handles file loading (Excel/CSV/JSON/text) with auto encoding detection (utf-8-sig, utf-8, latin-1, gbk, gb2312) and CSV delimiter detection. `utils/data_cleaner.py` strips NaN/Inf/non-numeric values. `utils/validators.py` validates inputs (values, spec limits, groups, alpha, subgroup sizes).

**Statistical modules** (`stats_engine/`): Each module is self-contained with a single public function that accepts `**params` and returns a dict. All computation uses `scipy.stats`, `numpy`, and `statsmodels` — no custom statistical implementations. All numeric output is rounded to 4 decimal places. Every analysis returns an `interpretation` string.

**Command discovery:** `stats_engine/discover.py` registers all commands with metadata (description, category, input/output fields, examples). The `discover` command exposes this for AI-agent introspection.

**Key modules by category:**
- **Basic:** `descriptive`, `normality`, `outlier`
- **Hypothesis:** `ttest`, `anova`, `nonparametric`, `homogeneity`, `equivalence`, `multiple_comparison`, `power`
- **SPC:** `control_chart`, `capability`, `trend`
- **Regression:** `regression`, `correlation`
- **DOE/MSA:** `doe`, `gage_rr`
- **Advanced:** `reliability`, `multivariate`, `timeseries`
- **Data:** `explore`, `clean`, `transform`, `report`, `discover`, `run`

## Key Conventions

- Adding a new command requires: (1) create module in `stats_engine/`, (2) add route in `_route()` in `main.py`, (3) register in `COMMANDS` dict in `stats_engine/discover.py`.
- The `run` command executes arbitrary Python via `exec()` — used for custom analysis scripts.
- Input validation uses helpers from `utils/validators.py` — don't inline validation logic.
- File-based commands support Excel (`.xlsx`, `.xls`), CSV (auto-delimiter), JSON, and plain text (one value per line).

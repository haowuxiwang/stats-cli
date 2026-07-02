# Contributing to stats-cli-py

Thank you for your interest in contributing! This document covers how to add commands, run tests, and submit changes.

## How to Add a New Command

### 1. Create or extend the module

Add your function to the appropriate module under `stats_engine/`, or create a new one. Example:

```python
# stats_engine/my_new_module.py

def my_new_command(values, alpha=0.05):
    """Short description of what this command does.

    Args:
        values: list of numbers
        alpha: significance level (default 0.05)

    Returns:
        dict with results (no underscored keys — those are reserved for metadata)
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values")

    # ... computation ...

    return {
        "statistic": ...,
        "p_value": ...,
    }
```

### 2. Register the command in `main.py`

Add an entry to `COMMAND_REGISTRY`:

```python
COMMAND_REGISTRY = {
    ...
    "my_new_command": ("stats_engine.my_new_module", "my_new_command"),
}
```

### 3. Add a schema to `stats_engine/discover.py`

Add a `CommandSchema` entry so that `discover` shows your command's description, parameters, and example.

### 4. Write tests

Create `tests/test_my_new_command.py` covering:

- Normal input → expected output
- Edge cases (empty input, single value, all identical)
- Invalid input → proper error raised

### 5. Verify

```bash
python -m pytest tests/test_my_new_command.py -v
python -m pytest tests/ --cov=stats_engine --cov-fail-under=95
ruff check .
```

## How to Run Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run full suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=stats_engine --cov-report=term-missing --cov-fail-under=95

# Run a single file
python -m pytest tests/test_descriptive.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "ttest" -v
```

## Code Style

We use **ruff** for linting and formatting.

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Style rules

- Python 3.9+ syntax only (`target-version = "py39"`)
- Line length: 120 characters
- Imports: sorted by ruff isort
- Prefer explicit error messages with both English and Chinese ("Error message / 错误信息")
- Use type hints on public functions
- No bare `except:` clauses

## PR Process

1. Fork or branch from `main`
2. Create feature branch (`feature/my-command` or `fix/edge-case`)
3. Make your change + add tests
4. Run full test suite locally (must pass ≥95% coverage)
5. Run `ruff check .` and `ruff format --check .` (both must pass)
6. Push and open PR against `main`
7. PR description should include:
   - What changed and why
   - Test results (coverage %)
   - Any breaking changes
8. CI must pass before merge (tests + lint run automatically)
9. Squash merge to main

## Error Handling Conventions

All errors returned to the caller follow this envelope:

```json
{
  "status": "error",
  "error_type": "DATA_ERROR",
  "message": "Description of what went wrong",
  "suggestion": "How to fix it"
}
```

Known error types are defined in `utils/output.py`: `INVALID_INPUT`, `MISSING_COMMAND`, `UNKNOWN_COMMAND`, `PARAM_ERROR`, `DATA_ERROR`, `COMPUTATION_ERROR`, `FILE_NOT_FOUND`, `MEMORY_ERROR`, `MISSING_DEPENDENCY`, `INTERNAL_ERROR`.

Raise `ValueError` with a descriptive message from your handler — `main.py`'s `handler()` function will classify and wrap it.

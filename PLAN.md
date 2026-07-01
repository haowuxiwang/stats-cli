# stats-cli-py: Short-Term Stabilization Plan

**Date:** 2026-07-01
**Target:** Stabilize baseline, prepare for mid-term feature work
**Philosophy:** Fix critical gaps, improve robustness, no speculative features.

---

## Phase 0: Pre-Flight Checklist (Today — Before Tomorrow)

| # | Task | Verify | Status |
|---|------|--------|--------|
| 0.1 | CI fully green (lint + 6 test jobs) | `gh` or GitHub UI shows all green | ⬜ Pending |
| 0.2 | All 1825 tests pass locally | `pytest tests/ -q` | ✅ Done |
| 0.3 | Coverage ≥ 96% | `--cov-fail-under=96` | ✅ 96.24% |
| 0.4 | Both ZIP packages build | `package.py` + `package_aily.py` | ✅ Done |
| 0.5 | Git working tree clean | `git status` shows clean | ⬜ Verify |

---

## Phase 1: Critical Fixes (Today — Short-Term)

### 1.1 Security: Harden `run` Command

**Problem:** Blacklist bypass possible, `print` in safe_builtins leaks to stdout.

**Fix:**
- Remove `print` from `safe_builtins` in `main.py`
- Replace string blacklist with `ast.parse` + node visitor (block all imports, attribute access, function calls except whitelisted)
- Add resource limit: max 10MB memory per script (use `resource.setrlimit` on Unix, `ctypes` on Windows)
- Return structured error for blocked patterns (not raw exception)

**Files:** `main.py` (lines 277-391)
**Tests:** Add `tests/test_run_security.py` — verify blocked patterns, memory limit, no stdout leak
**Effort:** 2-3 hours

### 1.2 Input Validation: Size Limits

**Problem:** No input size limit — agent can send 10M points → OOM.

**Fix:**
- Add `MAX_VALUES = 100_000` constant in `main.py`
- In `handler()`, check `len(params.get("values", []))` against limit
- Return `DATA_ERROR` with clear message: "values exceeds max size (100000). Use file input or sample."
- Same for `groups` (total elements), `x`/`y` arrays

**Files:** `main.py` (handler function)
**Tests:** Add to `tests/test_coverage_workflow_extra.py` — verify limit enforced
**Effort:** 30 minutes

### 1.3 Discover Schema Completeness

**Problem:** `discover` schema missing `optional` params, enum constraints, `chart` param.

**Fix:**
- Update `COMMANDS` dict in `stats_engine/discover.py` to include:
  - `optional: true/false` for each param
  - `enum: [...]` for params with fixed choices (test_type, chart_type, etc.)
  - `default: value` where applicable
- Add `chart: {type: "boolean", desc: "Generate chart (base64 PNG)", default: false}` to all chart-supporting commands
- Add `discover` and `run` to COMMANDS dict (currently missing from discover output)

**Files:** `stats_engine/discover.py`
**Tests:** Add `tests/test_discover_schema.py` — verify all commands have complete schema
**Effort:** 2 hours

### 1.4 Error Type for Unknown Commands

**Problem:** Unknown command returns `PARAM_ERROR` instead of dedicated type.

**Fix:**
- In `_route()`, catch unknown command before ValueError
- Return `error("Unknown command: {command}", ErrorType.UNKNOWN_COMMAND, suggestion="...")`
- Add `UNKNOWN_COMMAND` to `ErrorType` class

**Files:** `main.py`, `utils/output.py`
**Tests:** Add test for unknown command error type
**Effort:** 30 minutes

### 1.5 Chart Generation Failure Feedback

**Problem:** `_generate_chart` silently swallows exceptions — agent doesn't know why chart missing.

**Fix:**
- Return `chart_error: "Chart generation failed: {reason}"` in response when chart requested but failed
- Don't fail the whole request — return data with warning + chart_error field

**Files:** `main.py` (handler function, lines 43-57)
**Tests:** Add test that triggers chart failure (e.g., empty data with chart=true)
**Effort:** 1 hour

---

## Phase 2: Robustness Improvements (Today — Short-Term)

### 2.1 Defensive Copy for Nested Params

**Problem:** `handler()` does `params = dict(params or {})` — shallow copy only. Nested dicts (like `groups`) can still be mutated.

**Fix:**
- Use `copy.deepcopy(params)` instead of `dict(params)`
- Or document that handler may mutate nested structures

**Files:** `main.py`
**Effort:** 15 minutes

### 2.2 Timeout for Long-Running Commands

**Problem:** Some commands (distribution fitting with many distributions, MCMC bayesian) can hang.

**Fix:**
- Add `timeout` parameter to handler (default 30s)
- Use `signal.alarm` (Unix) / threading timer (Windows) to enforce
- Return `COMPUTATION_ERROR` with "Command timed out after {N}s"

**Files:** `main.py`
**Effort:** 1 hour

### 2.3 Standardize Missing Value Handling

**Problem:** Different modules handle NaN/Inf differently — some filter, some crash.

**Fix:**
- Add `_filter_nan_inf(values)` utility in `utils/validators.py`
- Call it at the start of every command that accepts `values`
- Document: "NaN and Inf are automatically removed before analysis"

**Files:** `utils/validators.py`, all stats_engine modules
**Effort:** 2 hours

### 2.4 Improve `gage_rr` Input Flexibility

**Problem:** Gage RR requires specific flat array format — agents struggle to construct it.

**Fix:**
- Accept both flat and 2D array format for `measurements`
- Auto-detect format and convert internally
- Accept `tolerance` as percentage (e.g., `tolerance_percent: 30`) in addition to absolute

**Files:** `stats_engine/gage_rr.py`
**Effort:** 1 hour

---

## Phase 3: Documentation & Developer Experience (Today — Short-Term)

### 3.1 Add `CONTRIBUTING.md`

**Content:**
- How to add a new command (step-by-step)
- How to add tests (coverage requirements)
- How to run CI locally (`act` or `ruff` + `pytest`)
- Code style (ruff config, naming conventions)

**Effort:** 1 hour

### 3.2 Add `CHANGELOG.md` Entry Template

**Content:**
- Format: `## [X.Y.Z] — YYYY-MM-DD`
- Sections: Added, Fixed, Changed, Security
- Link to comparison: `[X.Y.Z]: https://github.com/.../compare/vX.Y.Z...vX.Y.Z+1`

**Effort:** 30 minutes

### 3.3 Add Pre-Commit Hooks

**Content:**
- `.pre-commit-config.yaml` with ruff check + ruff format
- Document setup: `pip install pre-commit && pre-commit install`

**Files:** `.pre-commit-config.yaml`
**Effort:** 30 minutes

---

## Phase 4: Mid-Term Preparation (Today — Design Only)

These are for **tomorrow's execution**. Today we only design the approach.

### 4.1 Cox Proportional Hazards

**Why:** Pharma stability studies, survival analysis gap.
**Approach:** Use `lifelines.CoxPHFitter` or `scipy` implementation.
**API:** `{"command": "reliability", "params": {"analysis_type": "cox_ph", "time": [...], "event": [...], "covariates": [...]}}`
**Dependencies:** Add `lifelines>=0.28` to requirements.txt

### 4.2 Repeated Measures ANOVA

**Why:** Stability studies with multiple timepoints per batch.
**Approach:** Use `statsmodels.MixedLM` or `AnovaRM`.
**API:** `{"command": "anova", "params": {"anova_type": "repeated", "data": [...], "subject": "batch", "within": "time"}}`

### 4.3 N-Way ANOVA (3+ Factors)

**Why:** DOE with 3+ factors needs interaction analysis.
**Approach:** Extend existing `anova` command with `anova_type: "n_way"`.
**API:** `{"command": "anova", "params": {"anova_type": "n_way", "factors": {"A": [...], "B": [...], "C": [...]}, "response": [...]}}`

### 4.4 Measurement Uncertainty (GUM)

**Why:** ISO 17025 accredited labs require uncertainty budgets.
**Approach:** Implement GUM framework (Type A + Type B evaluation).
**API:** `{"command": "measurement_uncertainty", "params": {"measurements": [...], "resolution": 0.01, "calibration_uncertainty": 0.005}}`

---

## Phase 5: Maintenance Mode Entry Criteria

**We enter maintenance mode when:**

- [ ] CI fully green for 3 consecutive days
- [ ] All Phase 1 + 2 fixes merged and tested
- [ ] No CRITICAL or HIGH issues from adversarial review remain
- [ ] Documentation complete (CONTRIBUTING, CHANGELOG, pre-commit)
- [ ] Phase 4 designs reviewed and approved

**Maintenance mode rules:**
- No new features unless critical bug fix
- Dependency updates: monthly, only patch versions
- Security patches: immediate
- Documentation updates: as needed
- Test coverage: never drop below 96%

---

## Today's Execution Order

```
1. Verify CI green (0.1) — 5 min
2. Phase 1.2: Input size limits — 30 min
3. Phase 1.4: Unknown command error type — 30 min
4. Phase 1.1: Run command security — 2 hours
5. Phase 1.3: Discover schema — 2 hours
6. Phase 1.5: Chart error feedback — 1 hour
7. Phase 2.1: Deep copy params — 15 min
8. Phase 2.3: NaN standardization — 2 hours
9. Phase 3.1-3.3: Docs + pre-commit — 2 hours
10. Commit, push, verify CI — 30 min
```

**Total estimated effort: ~11 hours (1.5 work days)**

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| CI status | All 7 jobs green | 6/7 (lint fix pending) |
| Test count | ≥ 1870 (1825 + 45 new) | 1825 |
| Coverage | ≥ 96% | 96.24% |
| Security issues | 0 CRITICAL | 1 (run command) |
| Discover schema | 100% complete | ~70% |
| Input validation | All entry points | None |

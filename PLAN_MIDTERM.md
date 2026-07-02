# Mid-Term Execution Plan (Phase 4)

**Date:** 2026-07-02 (Tomorrow)
**Goal:** Add 4 new analysis capabilities identified as critical gaps
**Constraint:** Maintain ≥96% coverage, no regressions, match existing code style

---

## Pre-Flight (Start of Session)

1. Verify CI green on latest main
2. Run full test suite locally: `pytest tests/ -q --cov --cov-fail-under=96`
3. Create feature branch: `git checkout -b feat/mid-term-analysis`
4. Read existing patterns: `stats_engine/anova.py`, `stats_engine/reliability.py`

---

## Task 1: N-Way ANOVA (Extend Existing)

**Priority:** LOW (extends existing anova command)
**Effort:** 2 hours
**API:** `{"command": "anova", "params": {"anova_type": "n_way", "factors": {"A": [...], "B": [...], "C": [...]}, "response": [...]}}`

### Approach
- Extend `stats_engine/anova.py` with `_n_way_anova()` function
- Use `statsmodels.formula.api.ols` + `patsy` formula for multi-factor design
- Support 2-way (existing) and 3+ way (new) with interaction terms
- Return same envelope as one_way anova: f_statistic, p_value, significant, anova_table, interaction_effects

### Implementation Steps
1. Add `_n_way_anova(factors, response, alpha=0.05)` function
2. Build formula dynamically: `response ~ A + B + C + A:B + A:C + B:C + A:B:C`
3. Use `sm.stats.anova_lm(model, typ=2)` for Type II SS
4. Return pandas-style ANOVA table as dict
5. Wire into `anova()` dispatch via `anova_type == "n_way"`
6. Update `discover.py` schema: add `anova_type: "n_way"` to enum
7. Add tests in `tests/test_anova.py`

### Test Plan
```python
def test_n_way_anova_3_factors():
    factors = {"A": [1,1,2,2]*4, "B": [1,2]*8, "C": [1,1,1,1,2,2,2,2]*2}
    response = [10,12,14,16,11,13,15,17,12,14,16,18,13,15,17,19]
    result = handler({"command": "anova", "params": {
        "anova_type": "n_way", "factors": factors, "response": response
    }})
    assert result["status"] == "success"
    assert "f_statistic" in result["data"]
    assert len(result["data"]["anova_table"]) > 0
```

### Success Criteria
- [ ] 3-factor ANOVA works with interaction terms
- [ ] Returns consistent output format with one_way anova
- [ ] Tests added and passing
- [ ] discover.py updated

---

## Task 2: Repeated Measures ANOVA

**Priority:** HIGH (pharma stability studies)
**Effort:** 3 hours
**API:** `{"command": "anova", "params": {"anova_type": "repeated", "data": [...], "subject": "col", "within": "col"}}`

### Approach
- Use `statsmodels.stats.anova.AnovaRM` for repeated measures
- Support: one-way repeated, two-way repeated (subject × within × within)
- Return: f_statistic, p_value, anova_table, sphericity_test (Mauchly), corrections (Greenhouse-Geisser, Huynh-Feldt)

### Implementation Steps
1. Add `_repeated_anova(data, subject, within, alpha=0.05)` function
2. Accept data as DataFrame (from file) or dict of arrays
3. Use `AnovaRM(data, depvar, subject, within=[...]).fit()`
4. Add Mauchly sphericity test via `scipy.stats`
5. If p_sphericity < 0.05, apply Greenhouse-Geisser correction
6. Return results dict with raw + corrected p-values
7. Wire into `anova()` dispatch
8. Update discover.py schema

### Test Plan
```python
def test_repeated_measures_anova():
    # 6 subjects, 3 timepoints (within-subject)
    df = pd.DataFrame({
        "subject": [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6],
        "time": ["T0","T1","T2"] * 6,
        "value": [10,12,14,11,13,15,10,11,13,12,14,16,11,12,14,10,13,15]
    })
    # Use file-based input
    result = handler({"command": "anova", "params": {
        "anova_type": "repeated",
        "file": "test_data.csv",  # saved df
        "within": "time",
        "subject": "subject",
        "response": "value"
    }})
    assert result["status"] == "success"
    assert "f_statistic" in result["data"]
```

### Success Criteria
- [ ] One-way repeated measures works
- [ ] Sphericity test included
- [ ] GG/HF corrections applied when needed
- [ ] Consistent output with other anova types
- [ ] Tests passing

---

## Task 3: Cox Proportional Hazards

**Priority:** HIGH (pharma survival analysis gap)
**Effort:** 3 hours
**API:** `{"command": "reliability", "params": {"analysis_type": "cox_ph", "time": [...], "event": [...], "covariates": [[...], [...]}}`

### Approach
- Use `lifelines.CoxPHFitter` (add `lifelines>=0.28` to requirements.txt)
- Support: single covariates, multiple covariates, hazard ratios, confidence intervals
- Return: coefficients, hazard_ratios, confidence_interval, concordance_index, log_likelihood

### Implementation Steps
1. Add `_cox_ph(time, event, covariates, alpha=0.05)` function
2. Accept covariates as 2D array (samples × features)
3. Fit `CoxPHFitter().fit(duration_col="time", event_col="event")`
4. Extract: summary table, hazard_ratios, ph_assumption_test (Schoenfeld residuals)
5. If PH assumption violated (p < 0.05), return warning with suggestion to use time-varying model
6. Wire into `reliability()` command dispatch as `analysis_type: "cox_ph"`
7. Update discover.py schema
8. Add to chart support (survival curve chart)

### Dependencies
- Add `lifelines>=0.28,<1` to requirements.txt

### Test Plan
```python
def test_cox_ph_basic():
    np.random.seed(42)
    n = 50
    time = np.random.exponential(100, n)
    event = np.random.binomial(1, 0.8, n)
    age = np.random.normal(60, 10, n)
    treatment = np.random.binomial(1, 0.5, n)
    covariates = np.column_stack([age, treatment]).tolist()

    result = handler({"command": "reliability", "params": {
        "analysis_type": "cox_ph",
        "time": time.tolist(),
        "event": event.tolist(),
        "covariates": covariates
    }})
    assert result["status"] == "success"
    assert "hazard_ratios" in result["data"]
    assert len(result["data"]["hazard_ratios"]) == 2  # 2 covariates
    assert "concordance_index" in result["data"]
```

### Success Criteria
- [ ] Cox PH fitting works with 1+ covariates
- [ ] Hazard ratios with CI returned
- [ ] PH assumption test included
- [ ] Graceful handling when lifelines not installed
- [ ] Tests passing

---

## Task 4: Measurement Uncertainty (GUM)

**Priority:** MEDIUM (ISO 17025 requirement)
**Effort:** 4 hours
**API:** `{"command": "measurement_uncertainty", "params": {"measurements": [...], "resolution": 0.01, "calibration_uncertainty": 0.005, "coverage_factor": 2}}`

### Approach
- Implement GUM (Guide to Expression of Uncertainty in Measurement) framework
- Type A: statistical evaluation (from repeated measurements)
- Type B: systematic evaluation (calibration, resolution, environment)
- Combine: expanded uncertainty = coverage_factor × combined_standard_uncertainty

### Implementation Steps
1. New file: `stats_engine/uncertainty.py`
2. Implement class `MeasurementUncertainty`:
   - `add_type_a(values, name)` — from repeated measurements
   - `add_type_b_rectangular(name, half_width)` — resolution, quantization
   - `add_type_b_normal(name, std)` — calibration certificates
   - `add_type_b_triangular(name, half_width)` — environmental factors
   - `calculate()` — returns combined and expanded uncertainty
3. Wire into `measurement_uncertainty` command
4. Return: uncertainty_budget (table), combined_uncertainty, expanded_uncertainty, effective_dof, sensitivity_coefficients
5. Update discover.py

### Test Plan
```python
def test_measurement_uncertainty_basic():
    result = handler({"command": "measurement_uncertainty", "params": {
        "type_a_sources": [{"name": "repeatability", "values": [10.1, 10.2, 10.0, 10.1, 10.2]}],
        "type_b_sources": [
            {"name": "resolution", "distribution": "rectangular", "half_width": 0.005},
            {"name": "calibration", "distribution": "normal", "std": 0.002}
        ],
        "coverage_factor": 2,
        "sensitivity_coefficients": [1, 1, 1]
    }})
    assert result["status"] == "success"
    assert result["data"]["expanded_uncertainty"] > result["data"]["combined_uncertainty"]
    assert len(result["data"]["uncertainty_budget"]) == 3  # 1 type A + 2 type B
    assert result["data"]["coverage_factor"] == 2
    assert "effective_degrees_of_freedom" in result["data"]
```

### Success Criteria
- [ ] Type A evaluation from repeated measurements
- [ ] Type B with rectangular, normal, triangular distributions
- [ ] Combined and expanded uncertainty calculated
- [ ] Welch-Satterthwaite effective degrees of freedom
- [ ] Uncertainty budget table returned
- [ ] Tests passing
- [ ] New command registered in discover.py

---

## Execution Order (Tomorrow)

```
1. Pre-flight: verify CI, create branch, read existing code
2. Task 1: N-Way ANOVA (2h) — extends existing, lowest risk
3. Task 2: Repeated Measures ANOVA (3h) — high value for pharma
4. Task 3: Cox PH (3h) — closes JMP parity gap
5. Task 4: Measurement Uncertainty (4h) — new command, most complex
6. Run full test suite, verify coverage
7. Update discover.py, SKILL.md, OUTPUT.md
8. Commit, push, open PR
```

**Total estimated effort: ~12 hours (1.5 work days)**

---

## Post-Execution (Enter Maintenance Mode)

After all 4 tasks:
- Run full regression test suite
- Verify CI green
- Rebuild ZIP packages
- Update CHANGELOG.md
- Enter Phase 5: Maintenance Mode

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| lifelines dependency large | Optional import, graceful error if missing |
| Repeated measures formula parsing | Require file-based input, reject dict format |
| N-way ANOVA factorial explosion | Limit to 5 factors max, warn if design > 100 rows |
| GUM complexity | Start with basic Type A+B, skip Monte Carlo |
| Coverage drop | Each task must include tests before merge |

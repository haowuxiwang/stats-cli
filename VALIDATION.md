# Validation Report

## stats-cli-py v1.4.0

This document describes the validation methodology and results for stats-cli-py.

## Validation Methodology

### 1. NIST Reference Datasets (National Institute of Standards and Technology)
- **Source**: NIST Statistical Reference Datasets (StRD)
- **Coverage**: 9 univariate datasets + 26 nonlinear regression datasets
- **Tolerance**: 1e-4 to 0.1 relative tolerance depending on dataset
- **Result**: 35/35 PASS

### 2. R Cross-Validation
- **Source**: R statistical computing environment
- **Coverage**: Core statistical methods (descriptive, t-test, ANOVA, regression, correlation, nonparametric, homogeneity, capability, Bayesian, PLS, GLM, Bootstrap, factor analysis, MANOVA)
- **Tolerance**: 1e-10 to 1e-2 absolute/relative tolerance
- **Result**: 23/23 PASS

### 3. Benchmark Tests
- **Type**: Numerical accuracy + performance benchmarks
- **Coverage**: 35/39 commands
- **Tests**: 162 benchmark tests
- **Result**: 162/162 PASS

### 4. Stress Tests
- **Type**: Edge cases, boundary conditions, extreme values
- **Tests**: 64 stress tests
- **Result**: 64/64 PASS

### 5. Real-World Data
- **Source**: Manufacturing environment monitoring data (Chinese pharmaceutical factory)
- **Files**: 10 Excel files with environmental monitoring data
- **Analyses**: 29 analysis calls across multiple command types
- **Result**: 28/29 SUCCESS (1 encrypted file - expected failure)

### 6. Reproducibility
- **Method**: Same input + same seed -> identical output
- **Coverage**: All random methods (Monte Carlo, Sobol, Bootstrap, classification, DOE)
- **Result**: All PASS

## Accuracy Guarantees

| Method | Accuracy Level | Reference |
|--------|---------------|-----------|
| Descriptive statistics | 10+ significant digits | NIST |
| Linear regression | 8+ significant digits | NIST, R |
| Nonlinear regression | 1-20% relative error | NIST |
| Hypothesis tests (t, ANOVA) | p-value to 1e-20 | R |
| Correlation | 8+ significant digits | R |
| Process capability | 1% relative error | R |
| Bayesian methods | Consistent with R BayesFactor | R |
| PLS regression | Consistent with R pls package | R |
| GLM | Consistent with R glm() | R |
| Bootstrap | Consistent with R boot package | R |

## Test Suite Summary

| Metric | Value |
|--------|-------|
| Total tests | 1778 |
| Passed | 1778 |
| Failed | 0 |
| Skipped | 10 |
| Coverage | 97% |
| Benchmark tests | 162 |
| Stress tests | 64 |
| R cross-validation | 23 |
| NIST validation | 35 |

## Limitations

1. Nonlinear regression convergence depends on initial parameter guesses
2. Some edge cases (encrypted files, complex Excel headers) may not be handled
3. Monte Carlo results vary with seed (by design - reproducible with same seed)

## Comparison with Commercial Software

| Feature | stats-cli | JMP | Minitab |
|---------|-----------|-----|---------|
| NIST validation | YES (35/35) | YES | YES |
| R cross-validation | YES (23/23) | NO | NO |
| Open source | YES | NO | NO |
| CLI/API | YES | Partial | NO |
| AI agent integration | YES | NO | NO |
| FDA (Functional Data) | YES | NO | NO |
| Bayesian statistics | YES | Partial | NO |

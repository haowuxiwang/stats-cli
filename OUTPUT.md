# Output Interpretation Guide

This document explains how to interpret the key statistical metrics returned by each command.

---

## Table of Contents

- [1. Descriptive Statistics](#1-descriptive-statistics)
- [2. Normality Testing](#2-normality-testing)
- [3. Hypothesis Testing (t-test, ANOVA)](#3-hypothesis-testing)
- [4. Correlation & Regression](#4-correlation--regression)
- [5. Control Charts](#5-control-charts)
- [6. Process Capability](#6-process-capability)
- [7. Gage R&R (MSA)](#7-gage-rr-msa)
- [8. Reliability](#8-reliability)
- [9. Distribution Fitting](#9-distribution-fitting)
- [10. Common Patterns](#10-common-patterns)

---

## 1. Descriptive Statistics

**Command:** `descriptive`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `n` | Sample size | More = more reliable. n < 30 is small sample. |
| `mean` | Arithmetic average | Central tendency. Sensitive to outliers. |
| `median` | 50th percentile | Robust central tendency. Use when data is skewed. |
| `std` | Standard deviation (sample) | Spread of data. 68% within mean ± 1 std. |
| `rsd_percent` | Relative standard deviation (CV) | std/mean × 100. Use for comparing variability across different scales. |
| `skewness` | Asymmetry of distribution | 0 = symmetric. > 0 = right tail. < 0 = left tail. |
| `kurtosis` | Tail heaviness (excess) | 0 = normal. > 0 = heavy tails. < 0 = light tails. |
| `ci_95_lower`, `ci_95_upper` | 95% confidence interval for mean | 95% confident true mean is within this range. |

**Rule of thumb:**
- |skewness| > 1 → strongly non-normal, consider transformation
- |kurtosis| > 3 → heavy tails, check for outliers
- RSD < 5% → good precision; RSD > 15% → high variability

---

## 2. Normality Testing

**Command:** `normality`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `shapiro_wilk.statistic` | Test statistic (0-1) | Closer to 1 = more normal |
| `shapiro_wilk.p_value` | p-value | > 0.05 → cannot reject normality |
| `is_normal` | Overall verdict | true = data is approximately normal |

**Decision logic:**
```
if is_normal:
    use parametric tests (ttest, anova, pearson)
else:
    use nonparametric tests (mann_whitney, kruskal_wallis, spearman)
```

**Important notes:**
- Shapiro-Wilk works best for n < 5000
- With large n, even tiny deviations give p < 0.05 (use QQ plot visually)
- Always check `anderson_darling` as backup

---

## 3. Hypothesis Testing

### t-test

**Command:** `ttest`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `t_statistic` | t-value | Larger absolute = stronger evidence against H₀ |
| `p_value` | Probability of observed data if H₀ true | < 0.05 → reject H₀ (significant) |
| `significant` | Verdict | true = p < alpha |
| `ci_95` | 95% CI for mean difference | If CI doesn't include 0 → significant |

**Interpretation template:**
```
if p_value < 0.05:
    "There IS a statistically significant difference (p={p_value})"
else:
    "There is NO statistically significant difference (p={p_value})"
```

### ANOVA

**Command:** `anova`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `f_statistic` | F-ratio | Larger = more between-group variance vs within |
| `p_value` | p-value | < 0.05 → at least one group differs |
| `significant` | Verdict | true = p < alpha |

**After ANOVA is significant:** Run `multiple_comparison` to find WHICH groups differ.

### Multiple Comparison

**Command:** `multiple_comparison`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `comparisons` | Pairwise results | Each has group1, group2, mean_diff, p_value, significant |
| `significant` | Any pair different? | true = at least one pair is significant |

---

## 4. Correlation & Regression

### Correlation

**Command:** `correlation`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `pearson_r` | Linear correlation (-1 to 1) | > 0.7 strong, 0.3-0.7 moderate, < 0.3 weak |
| `spearman_r` | Rank correlation | Use when non-linear but monotonic |
| `p_value` | Significance | < 0.05 → correlation is real |

### Regression

**Command:** `regression`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `r_squared` | Coefficient of determination (0-1) | Fraction of variance explained. > 0.7 = good fit |
| `adj_r_squared` | Adjusted R² | Use for multiple regression (penalizes extra predictors) |
| `slope` | Rate of change | For 1 unit increase in X, Y changes by slope |
| `intercept` | Y when X = 0 | Baseline value |
| `coefficients` | For polynomial/multiple | Ordered from highest to lowest degree |
| `residual_std` | Standard deviation of residuals | Smaller = tighter fit |
| `equation` | Human-readable formula | "y = 2.5*x + 1.3" |

**Regression quality guide:**

| R² value | Interpretation |
|----------|---------------|
| > 0.9 | Excellent fit |
| 0.7 - 0.9 | Good fit |
| 0.5 - 0.7 | Moderate fit |
| 0.3 - 0.5 | Weak fit |
| < 0.3 | Poor fit — consider other models |

---

## 5. Control Charts

**Command:** `control_chart`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `center` | Center line (process mean) | Target value |
| `ucl` | Upper control limit | mean + 3σ |
| `lcl` | Lower control limit | mean - 3σ |
| `out_of_control_points` | Points beyond limits | Indicate special cause variation |
| `summary.stable` | Process stability verdict | true = no special causes detected |

**Out-of-control rules (Western Electric):**
1. Any point outside 3σ (beyond UCL/LCL)
2. 2 of 3 consecutive points beyond 2σ
3. 4 of 5 consecutive points beyond 1σ
4. 8 consecutive points on one side of center
5. 6 consecutive points steadily increasing/decreasing

**If process is NOT stable:** Investigate and remove special causes BEFORE calculating capability.

---

## 6. Process Capability

**Command:** `capability`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `cp` | Potential capability | What process COULD do if perfectly centered |
| `cpk` | Actual capability | What process IS doing (accounts for off-centering) |
| `pp` | Performance capability | Uses overall std (long-term) |
| `ppk` | Performance capability (centered) | Long-term actual |
| `ppm` | Parts per million out of spec | Defect rate estimate |
| `rating` | Verdict | "Excellent"/"Good"/"Marginal"/"Poor" |

**Capability benchmarks:**

| Cpk value | Interpretation | PPM (approx) |
|-----------|---------------|--------------|
| ≥ 1.67 | World-class | < 0.6 |
| 1.33 - 1.67 | Excellent | < 60 |
| 1.0 - 1.33 | Capable | < 2700 |
| 0.67 - 1.0 | Marginal | < 63 |
| < 0.67 | Incapable | > 45000 |

**Key rules:**
- Always use Cpk (not Cp) for decision-making
- Cpk < 1.0 → process produces defects
- Cpk = 1.33 → minimum for new processes
- Cpk = 1.67 → target for critical characteristics
- Process MUST be stable (control chart) before capability is valid

---

## 7. Gage R&R (MSA)

**Command:** `gage_rr`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `contribution` | % of total variance from each source | Repeatability + Reproducibility should be < 30% |
| `study_variation` | % of tolerance consumed by measurement | < 10% acceptable, < 30% marginal |
| `ndc` | Number of distinct categories | ≥ 5 required for good discrimination |
| `rating` | Overall verdict | "Acceptable"/"Marginal"/"Unacceptable" |

**Gage R&R acceptance criteria:**

| Metric | Acceptable | Marginal | Unacceptable |
|--------|-----------|----------|--------------|
| % Contribution | < 1% | 1-9% | > 9% |
| % Study Variation | < 10% | 10-30% | > 30% |
| NDC | ≥ 5 | 3-4 | < 3 |

**If Gage R&R is unacceptable:**
- Repeatability high → equipment needs maintenance/replacement
- Reproducibility high → operator training needed, fixturing improved

---

## 8. Reliability

**Command:** `reliability`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `shape` (β) | Weibull shape parameter | < 1 = early failures, 1 = random, > 1 = wear-out |
| `scale` (η) | Weibull characteristic life | 63.2% fail by this time |
| `mtbf` | Mean time between failures | Average lifetime |
| `failure_rate` | Failures per unit time | Constant for exponential, varies for Weibull |

**Weibull shape interpretation:**

| β value | Failure mode | Action |
|---------|-------------|--------|
| < 1 | Infant mortality (decreasing failure rate) | Burn-in, quality screening |
| = 1 | Random failures (constant rate) | Preventive maintenance doesn't help |
| 1 < β < 4 | Wear-out (increasing rate) | Scheduled replacement |
| ≥ 4 | Rapid wear-out | Immediate design review |

---

## 9. Distribution Fitting

**Command:** `distribution`

| Field | Meaning | How to interpret |
|-------|---------|------------------|
| `params` | Fitted parameters | Varies by distribution |
| `aic` | Akaike Information Criterion | Lower = better fit (compare models) |
| `bic` | Bayesian Information Criterion | Lower = better (penalizes complexity more than AIC) |
| `ks_statistic` | Kolmogorov-Smirnov statistic | Max distance between fitted and empirical CDF |
| `ks_p_value` | KS test p-value | > 0.05 → cannot reject fit |

**Distribution selection guide:**

| Data type | Common distributions |
|-----------|---------------------|
| Lifetime/reliability | Weibull, Exponential, Lognormal |
| Natural measurements | Normal, Lognormal |
| Count data | Poisson, Negative Binomial |
| Proportions | Beta |
| Extreme values | Gumbel, Frechet |

---

## 10. Common Patterns

### Pattern: Normality → Compare → Report

```
1. descriptive(values)        → check mean, std, skewness
2. normality(values)          → is_normal?
3. if normal:
       ttest(values, values2) → p_value
   else:
       nonparametric(x=values, y=values2, test_type="mann_whitney")
4. report(values, usl, lsl)   → full summary
```

### Pattern: Process Validation

```
1. control_chart(values)      → stable?
2. if stable:
       capability(values, usl, lsl) → cpk, ppm
3. gage_rr(...)               → measurement system ok?
4. report(...)                → batch release decision
```

### Pattern: Root Cause Investigation

```
1. descriptive(values)        → basic stats
2. outlier(values)            → any anomalies?
3. normality(values)          → distribution shape
4. control_chart(values)      → when did shift happen?
5. trend(values)              → systematic drift?
```

---

## Quick Reference Card

| If you see... | It means... | Action |
|---------------|-------------|--------|
| p_value < 0.05 | Significant difference/relationship | Reject null hypothesis |
| p_value ≥ 0.05 | No significant difference | Cannot reject null (not "accept") |
| Cpk < 1.0 | Process produces defects | Center process or reduce variation |
| Cpk ≥ 1.33 | Capable process | Monitor with control charts |
| NDC < 5 | Measurement system can't discriminate | Improve equipment/training |
| % Study Variation > 30% | Measurement error too large | Fix measurement system first |
| is_normal = false | Data not normal | Use nonparametric tests |
| stable = false | Special causes present | Investigate before capability |
| r_squared < 0.5 | Model doesn't explain much | Try different model or more predictors |

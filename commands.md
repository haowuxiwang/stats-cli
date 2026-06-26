# stats-cli-py Command Reference

Complete list of all 39 commands with parameters and examples.

## Basic Statistics

### descriptive
Descriptive statistics: mean, median, SD, RSD%, range, quartiles, 95% CI, skewness, kurtosis.
```json
{"command": "descriptive", "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4]}}
```

### normality
Normality tests: Shapiro-Wilk, Anderson-Darling, Lilliefors.
```json
{"command": "normality", "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4]}}
```

### outlier
Outlier detection: Grubbs, Dixon, IQR, Z-score, MAD.
```json
{"command": "outlier", "params": {"values": [10, 11, 10, 12, 100], "method": "grubbs"}}
```

## Hypothesis Testing

### ttest
t-tests: one_sample, two_sample, paired.
```json
{"command": "ttest", "params": {"test_type": "two_sample", "values": [10.2, 10.5], "values2": [11.3, 11.5]}}
```

### anova
ANOVA: one_way, two_way.
```json
{"command": "anova", "params": {"anova_type": "one_way", "groups": [[10.2, 10.5], [11.3, 11.5]]}}
```

### nonparametric
Non-parametric tests: mann_whitney, kruskal_wallis, wilcoxon, chi_square, friedman.
```json
{"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": [10.2, 10.5], "y": [11.3, 11.5]}}
```

### homogeneity
Homogeneity of variance: levene, bartlett.
```json
{"command": "homogeneity", "params": {"test_type": "levene", "groups": [[10.2, 10.5], [11.3, 11.5]]}}
```

### equivalence
Equivalence tests: tost, one_sample_tost.
```json
{"command": "equivalence", "params": {"test_type": "tost", "values": [10.2, 10.5], "values2": [10.3, 10.6], "delta": 0.5}}
```

### multiple_comparison
Multiple comparison: tukey, bonferroni, scheffe.
```json
{"command": "multiple_comparison", "params": {"test_type": "tukey", "groups": [[10.2, 10.5], [11.3, 11.5], [12.1, 12.4]]}}
```

### power
Power analysis and sample size calculation.
```json
{"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}}
```

## Regression

### regression
Linear, polynomial, multiple, stepwise, logistic, nonlinear, LASSO, Ridge, ElasticNet, robust, PLS, GLM, cross_validate.
```json
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}}
{"command": "regression", "params": {"x": [[1,2],[3,4],[5,6]], "y": [1,2,3], "reg_type": "pls"}}
{"command": "regression", "params": {"x": [[1],[2],[3]], "y": [2,4,6], "reg_type": "poisson"}}
```

### correlation
Correlation: pearson, spearman, kendall.
```json
{"command": "correlation", "params": {"x": [1, 2, 3], "y": [2, 4, 6]}}
```

## SPC / Quality Control

### control_chart
Control charts: xbar, r, imr, p, np, c, u, ewma, cusum, hotelling_t2, ewma_mv, zmr.
```json
{"command": "control_chart", "params": {"chart_type": "imr", "values": [10.1, 10.2, 10.0, 10.3, 10.1]}}
```

### capability
Process capability: Cp, Cpk, Pp, Ppk, Cpm, Box-Cox, Johnson.
```json
{"command": "capability", "params": {"values": [10.1, 10.2, 10.0, 10.3], "usl": 11.0, "lsl": 9.0}}
```

### trend
Trend analysis: cusum, ewma, runs.
```json
{"command": "trend", "params": {"values": [10.1, 10.2, 10.0, 10.3, 10.1], "test_type": "cusum"}}
```

## MSA / Gage R&R

### gage_rr
MSA: crossed, nested, attribute, bias, linearity, stability, destructive.
```json
{"command": "gage_rr", "params": {"analysis_type": "crossed", "measurements": [10, 11, 10, 11, 12, 11], "parts": ["P1","P1","P1","P2","P2","P2"], "operators": ["A","B","A","A","B","A"]}}
```

## DOE

### doe
DOE: full_factorial, fractional_factorial, response_surface, taguchi, definitive_screening.
```json
{"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "A", "levels": 3}, {"name": "B", "levels": 2}]}}
```

## Reliability

### reliability
Reliability: weibull, kaplan_meier, distribution, stability, alt, crow.
```json
{"command": "reliability", "params": {"analysis_type": "weibull", "times": [100, 200, 300], "status": [1, 1, 1]}}
{"command": "reliability", "params": {"analysis_type": "alt", "stress_levels": [350, 400, 450], "failure_times": [1000, 500, 200], "stress_model": "arrhenius", "use_stress": 300}}
```

## Multivariate

### multivariate
Multivariate: pca, cluster, discriminant, correlation_matrix, factor_analysis, manova.
```json
{"command": "multivariate", "params": {"analysis_type": "pca", "columns": ["x1", "x2"], "file": "data.csv"}}
{"command": "multivariate", "params": {"analysis_type": "factor_analysis", "values": [[1,2],[3,4],[5,6]], "n_factors": 1}}
```

## Bayesian

### bayesian
Bayesian: estimate, ttest, proportion, anova.
```json
{"command": "bayesian", "params": {"analysis_type": "ttest", "values": [10.2, 10.5, 10.3], "values2": [11.3, 11.5, 11.1]}}
```

## Distribution Analysis

### distribution
Distribution: fit, gof, select.
```json
{"command": "distribution", "params": {"analysis_type": "select", "values": [1.2, 1.5, 1.3, 1.8, 2.1, 1.9]}}
```

## Data Mining

### mining
Mining: classify, anomaly, associate.
```json
{"command": "mining", "params": {"analysis_type": "classify", "features": [[1,2],[3,4],[5,6],[7,8]], "labels": [0,0,1,1]}}
```

## Acceptance Sampling

### acceptance_sampling
Acceptance sampling: single_plan, double_plan, oc_curve, find_plan.
```json
{"command": "acceptance_sampling", "params": {"analysis_type": "single_plan", "n": 50, "c": 2, "defect_rate": 0.05}}
```

## Sensitivity Analysis

### sensitivity
Sensitivity: monte_carlo, tornado, sobol.
```json
{"command": "sensitivity", "params": {"analysis_type": "monte_carlo", "inputs": {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}, "formula": "x * 2", "n_simulations": 10000}}
```

## Functional Data Analysis

### functional
FDA: basis, smooth, derivative, fpca, regression, fanova, cluster.
```json
{"command": "functional", "params": {"analysis_type": "fpca", "curves": [[1,2,3,4,5],[2,3,4,5,6]], "t": [0,0.25,0.5,0.75,1.0], "n_components": 2}}
```

## Advanced

### advanced
Advanced: exact_test, mcnemar, cochran_q, mixed_effects, bootstrap.
```json
{"command": "advanced", "params": {"analysis_type": "bootstrap", "values": [1,2,3,4,5,6,7,8,9,10], "statistic": "mean"}}
```

## Data Processing

### clean
Data cleaning: drop, impute_mean, impute_median, winsorize, clip.
```json
{"command": "clean", "params": {"values": [10.1, 10.2, null, 10.3], "method": "impute_mean"}}
```

### transform
Data transformation: log, sqrt, boxcox, johnson, rank, standardize, recip.
```json
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6], "method": "boxcox"}}
```

## Workflow

### workflow
Multi-step workflow with automatic assumption checking.
```json
{"command": "workflow", "params": {"steps": [{"command": "descriptive"}, {"command": "normality"}], "values": [10.2, 10.5, 10.3]}}
```

### check_assumptions
Check statistical assumptions for planned analysis.
```json
{"command": "check_assumptions", "params": {"values": [10.2, 10.5, 10.3], "test_type": "ttest"}}
```

### recommend
Get recommended statistical tests.
```json
{"command": "recommend", "params": {"data_description": {"goal": "compare means", "n_groups": 2}}}
```

## Data Exploration

### explore
Explore data file structure.
```json
{"command": "explore", "params": {"file": "data.xlsx"}}
```

## Meta Commands

### discover
List all available commands and schemas.
```json
{"command": "discover", "params": {}}
{"command": "discover", "params": {"category": "spc"}}
{"command": "discover", "params": {"command_name": "regression"}}
```

### run
Execute custom Python script.
```json
{"command": "run", "params": {"script": "result = sum(data['values'])", "data": {"values": [1, 2, 3]}}}
```

## Reporting

### report
Generate comprehensive analysis report.
```json
{"command": "report", "params": {"values": [10.1, 10.2, 10.0, 10.3], "usl": 11.0, "lsl": 9.0}}
```

### export_excel
Export to Excel.
```json
{"command": "export_excel", "params": {"report_data": {}, "output_path": "report.xlsx"}}
```

### export_pdf
Export to PDF.
```json
{"command": "export_pdf", "params": {"report_data": {}, "output_path": "report.pdf"}}
```

### workflow_template
Execute predefined workflow templates.
```json
{"command": "workflow_template", "params": {"template_name": "manufacturing", "values": [10.2, 10.5, 10.3], "usl": 11.0, "lsl": 9.0}}
```

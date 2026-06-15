# Changelog

All notable changes to stats-cli-py will be documented in this file.

## [1.2.1] - 2026-06-15

### Fixed

- Added missing `fpdf2>=2.8` to pyproject.toml dependencies
- Fixed build-backend from legacy to `setuptools.build_meta`
- Updated version to 1.2.1 across all config files
- Fixed fpdf2 deprecation warnings in report.py (`ln=True` → `new_x=XPos.LMARGIN, new_y=YPos.NEXT`)
- Fixed Matplotlib deprecation in charts.py (`labels` → `tick_labels`)
- Fixed silent exception swallowing in main.py chart generation (now logs via `logging.debug`)
- Fixed silent exception swallowing in transform.py Johnson fallback (now logs exception)

### Improved (SKILL.md - AI Agent Best Practices)

- Added Intent-to-Command quick lookup table at top of SKILL.md (19 commands mapped)
- Added `discover` command recommendation for AI agents at document top
- Added negative triggers to YAML description (NOT for: streaming, text, image, non-numeric)
- Added top 5 command output examples (descriptive, normality, ttest, capability, control_chart)
- Added error recovery workflow table (6 error types with recovery steps)
- Added `auto_check` failure semantics documentation
- Added `run` command sandbox restrictions documentation
- Added Box-Cox cross-links in comparison decision tree (non-normal → consider transform first)
- Replaced all placeholder `[...]` with runnable example data (27 occurrences)
- Excluded TODO.md and CLAUDE.md from distributable zip

### Improved (discover.py - Structured Parameters)

- Added structured `params` field to all 33 commands in discover registry
- Each parameter now includes: `name`, `required` (boolean), `type`, `desc`
- AI agents can query `discover {"command_name": "xxx"}` to get required/optional distinction

## [1.0.0] - 2026-06-09

### Added

**Core Statistical Engine (26 commands)**

- **Data Exploration**: `explore` (multi-sheet Excel, CSV, JSON, text), `discover` (command registry)
- **Basic Statistics**: `descriptive` (mean, median, std, RSD%, CI, skewness, kurtosis), `normality` (Shapiro-Wilk, Anderson-Darling, Lilliefors), `outlier` (Grubbs, Dixon, IQR, Z-score)
- **Hypothesis Testing**: `ttest` (one-sample, two-sample, paired), `anova` (one-way, two-way), `nonparametric` (Mann-Whitney, Kruskal-Wallis, Wilcoxon, Chi-square, Friedman), `homogeneity` (Levene, Bartlett), `equivalence` (TOST), `multiple_comparison` (Tukey, Bonferroni, Scheffe), `power` (t-test, ANOVA, proportion, correlation)
- **Regression**: `regression` (linear, quadratic, polynomial, multiple, stepwise, logistic), `correlation` (Pearson, Spearman, Kendall)
- **SPC**: `control_chart` (X-bar, R, I-MR, p, np, c, u, EWMA, CUSUM + Western Electric rules), `capability` (Cp, Cpk, Pp, Ppk, Cpm, Box-Cox, PPM/yield), `trend` (CUSUM, EWMA, runs test)
- **MSA**: `gage_rr` (crossed, nested, attribute, bias, linearity, stability)
- **DOE**: `doe` (full factorial, fractional factorial, response surface, Taguchi)
- **Reliability**: `reliability` (Weibull, Kaplan-Meier, distribution fitting, stability/shelf life)
- **Multivariate**: `multivariate` (PCA, cluster, discriminant, correlation matrix)
- **Time Series**: `timeseries` (exponential smoothing, ARIMA, decomposition, ACF/PACF)
- **Data Processing**: `clean` (drop, impute, winsorize, clip), `transform` (log, sqrt, Box-Cox, Johnson, rank, standardize, reciprocal)
- **Advanced**: `advanced` (mixed effects, Fisher's exact test, McNemar, Cochran's Q)
- **Reporting**: `report` (comprehensive analysis), `run` (custom Python scripts)

**Infrastructure**

- Pure Python implementation (scipy + statsmodels, no R dependency)
- Multi-sheet Excel support with automatic sheet discovery
- Smart guidance flow for vague user requests
- Standard JSON output envelope (success/error)
- Auto encoding detection (utf-8-sig, utf-8, latin-1, gbk, gb2312)
- Auto CSV delimiter detection
- SKILL.md with 4 decision trees and 5 scenario workflows
- CI/CD pipeline (GitHub Actions)
- Pre-commit hooks (ruff)
- 185 unit tests + 31 real-data integration tests

## [0.1.0] - 2026-06-01

### Added

- Initial project structure
- Core statistical modules (descriptive, normality, ttest, anova)
- Basic CLI with JSON I/O

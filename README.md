# stats-cli-py

Pure Python statistical analysis CLI/library for manufacturing and quality engineering.

**Version**: 1.0.0
**Commands**: 26
**Dependencies**: scipy, statsmodels, pandas, numpy, scikit-learn, openpyxl

---

## Features

### Data Exploration
- **explore**: Inspect Excel/CSV structure (columns, types, missing values)
- **discover**: List all available commands and schemas

### Basic Statistics
- **descriptive**: Mean, median, SD, RSD%, range, quartiles, 95% CI, skewness, kurtosis
- **normality**: Shapiro-Wilk, Anderson-Darling, Lilliefors tests
- **outlier**: Grubbs, Dixon, IQR, Z-score outlier detection

### Hypothesis Testing
- **ttest**: One-sample, two-sample, paired t-tests
- **anova**: One-way, two-way ANOVA with Tukey post-hoc
- **nonparametric**: Mann-Whitney, Kruskal-Wallis, Wilcoxon, Chi-square, Friedman
- **homogeneity**: Levene, Bartlett variance tests
- **multiple-comparison**: Tukey, Bonferroni, Scheffe
- **equivalence**: TOST (Two One-Sided Tests)
- **power**: Sample size and power analysis (t-test, ANOVA, proportion)

### Regression & Correlation
- **regression**: Linear, quadratic, polynomial, multiple, logistic, stepwise
- **correlation**: Pearson, Spearman, Kendall

### SPC / Quality Control
- **control-chart**: X-bar, R, I-MR, p, np, c, u, EWMA, CUSUM
- **capability**: Cp, Cpk, Pp, Ppk, Cpm, Box-Cox non-normal capability
- **trend**: CUSUM, EWMA, runs test

### MSA (Measurement System Analysis)
- **gage-rr**: Crossed, nested, attribute, bias, linearity, stability

### Reliability
- **reliability**: Weibull, Kaplan-Meier, distribution fitting, stability/shelf life

### Multivariate
- **multivariate**: PCA, cluster (hierarchical/k-means), discriminant, correlation matrix

### Time Series
- **timeseries**: Exponential smoothing, ARIMA, decomposition, ACF/PACF

### DOE
- **doe**: Full factorial, fractional factorial, response surface, Taguchi

### Data Processing
- **clean**: Drop, impute mean/median, winsorize, clip
- **transform**: Log, sqrt, Box-Cox, Johnson, rank, standardize, reciprocal

### Reporting
- **report**: Comprehensive HTML report
- **run**: Execute custom Python scripts

---

## Installation

```bash
# From source
git clone https://github.com/haowuxiwang/stats-cli-py.git
cd stats-cli-py
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| scipy | Statistical tests, distributions |
| statsmodels | ANOVA, regression, time series |
| pandas | Data loading, manipulation |
| numpy | Numerical computation |
| scikit-learn | PCA, clustering, discriminant |
| openpyxl | Excel file support |

---

## Quick Start

### From stdin (JSON)

```bash
# Descriptive statistics
echo '{"command":"descriptive","params":{"values":[10.2,10.5,10.3,10.1,10.4]}}' | python main.py

# Normality test
echo '{"command":"normality","params":{"values":[10.2,10.5,10.3,10.1,10.4]}}' | python main.py

# t-test (two-sample)
echo '{"command":"ttest","params":{"values":[10.2,10.5,10.3],"values2":[11.1,11.3,11.5]}}' | python main.py
```

### From file

```bash
# Write JSON to file
echo '{"command":"capability","params":{"file":"data.csv","column":"weight","usl":11.0,"lsl":9.0}}' > input.json

# Run
python main.py input.json
```

### From Python

```python
from main import handler

result = handler({
    "command": "descriptive",
    "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4]}
})
print(result)
```

---

## All Commands

| Category | Command | Description |
|----------|---------|-------------|
| Data | `explore` | Explore data file structure |
| Data | `discover` | List all commands and schemas |
| Basic | `descriptive` | Descriptive statistics |
| Basic | `normality` | Normality tests |
| Basic | `outlier` | Outlier detection |
| Hypothesis | `ttest` | t-tests (one/two-sample, paired) |
| Hypothesis | `anova` | ANOVA (one-way, two-way) |
| Hypothesis | `nonparametric` | Non-parametric tests |
| Hypothesis | `homogeneity` | Variance homogeneity tests |
| Hypothesis | `multiple_comparison` | Multiple comparison tests |
| Hypothesis | `equivalence` | TOST equivalence test |
| Hypothesis | `power` | Power and sample size analysis |
| Regression | `regression` | Regression analysis |
| Regression | `correlation` | Correlation analysis |
| SPC | `control_chart` | Control charts (X-bar, R, I-MR, etc.) |
| SPC | `capability` | Process capability (Cp, Cpk, etc.) |
| SPC | `trend` | Trend analysis |
| MSA | `gage_rr` | Gage R&R study |
| Reliability | `reliability` | Reliability analysis |
| Multivariate | `multivariate` | PCA, clustering, discriminant |
| Time Series | `timeseries` | Time series analysis |
| DOE | `doe` | Design of experiments |
| Processing | `clean` | Data cleaning |
| Processing | `transform` | Data transformation |
| Reporting | `report` | HTML report generation |
| Reporting | `run` | Execute custom Python scripts |

---

## Decision Trees for AI Agents

### Comparing Groups
```
2 groups -> normality check -> t-test or Mann-Whitney
3+ groups -> normality check -> ANOVA or Kruskal-Wallis
Paired -> normality check -> paired t-test or Wilcoxon
```

### Relationship Analysis
```
2 continuous -> correlation or linear regression
Multiple predictors -> multiple or stepwise regression
Binary outcome -> logistic regression
Many variables -> PCA or cluster
```

### Quality Control
```
Monitor stability -> control-chart imr/xbar
Assess capability -> capability with USL/LSL
Verify measurement -> gage-rr
Analyze failures -> reliability weibull
```

---

## Output Format

All commands return JSON with a standard envelope:

### Success
```json
{
  "status": "success",
  "version": "1.0.0",
  "timestamp": "2026-06-09T10:00:00Z",
  "data": {
    "n": 5,
    "mean": 10.3,
    "std": 0.1581,
    "interpretation": "Data shows no significant deviation from normality."
  }
}
```

### Error
```json
{
  "status": "error",
  "error_type": "VALIDATION_ERROR",
  "message": "At least 3 values are required",
  "suggestion": "Provide more data points"
}
```

---

## File Support

| Format | Extension | Notes |
|--------|-----------|-------|
| Excel | .xlsx, .xls | Multi-sheet with `--sheet` |
| CSV | .csv | Auto-detect encoding and delimiter |
| JSON | .json | Structured data |
| Text | .txt | One value per line |

### Encoding Detection

CSV files are auto-detected in this order:
1. UTF-8 with BOM (utf-8-sig)
2. UTF-8
3. Latin-1
4. GBK / GB2312 (Chinese)

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_descriptive.py -v

# Run with coverage
python -m pytest tests/ -v --cov=stats_engine --cov-report=html
```

---

## Development

### Linting

```bash
# Install ruff
pip install ruff

# Check
ruff check .

# Format
ruff format .
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## Project Structure

```
stats-cli-py/
├── main.py                 # Entry point, JSON router
├── __main__.py             # python -m support
├── requirements.txt        # Runtime dependencies
├── requirements-test.txt   # Test dependencies
├── pyproject.toml          # Ruff config
├── SKILL.md                # Claude Code skill definition
├── install-skill.sh        # Skill installer
├── stats_engine/           # 26 command modules
│   ├── descriptive.py
│   ├── normality.py
│   ├── ttest.py
│   ├── anova.py
│   ├── regression.py
│   ├── control_chart.py
│   ├── capability.py
│   ├── gage_rr.py
│   ├── reliability.py
│   ├── multivariate.py
│   ├── timeseries.py
│   ├── doe.py
│   ├── discover.py         # Command registry
│   └── ...
├── utils/                  # Shared utilities
│   ├── output.py           # Standard JSON envelope
│   ├── validators.py       # Input validation
│   ├── data_loader.py      # File loading (Excel/CSV/JSON)
│   └── data_cleaner.py     # NaN/Inf cleaning
├── tests/                  # Pytest test suite
├── .github/workflows/      # CI pipeline
└── .gitignore
```

---

## License

MIT

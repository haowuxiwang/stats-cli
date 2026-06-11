# stats-cli-py

Pure Python statistical analysis CLI/library for manufacturing and quality engineering.

**Version**: 1.2.0
**Commands**: 33
**Test Coverage**: 92%
**Dependencies**: scipy, statsmodels, pandas, numpy, scikit-learn, openpyxl, fpdf2

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
- **anova**: One-way, two-way ANOVA with Tukey post-hoc (includes omega-squared effect size, df/ss/ms breakdown)
- **nonparametric**: Mann-Whitney, Kruskal-Wallis, Wilcoxon, Chi-square, Friedman
- **homogeneity**: Levene, Bartlett variance tests
- **multiple-comparison**: Tukey, Bonferroni, Scheffe
- **equivalence**: TOST (Two One-Sided Tests)
- **power**: Sample size and power analysis (t-test, ANOVA, proportion)

### Regression & Correlation
- **regression**: Linear, quadratic, polynomial, multiple, logistic, stepwise, **nonlinear** (exponential, power, logarithmic, sigmoid)
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
- **doe**: Full factorial, fractional factorial, response surface, Taguchi (supports int, list, and low/high factor formats)

### Data Processing
- **clean**: Drop, impute mean/median, winsorize, clip
- **transform**: Log, sqrt, Box-Cox, Johnson, rank, standardize, reciprocal

### Reporting
- **report**: Comprehensive HTML report
- **run**: Execute custom Python scripts

### Workflow Automation
- **workflow**: Multi-step analysis with automatic assumption checking
- **workflow_template**: Predefined templates (manufacturing, comparison, capability, exploration)
- **check_assumptions**: Verify statistical assumptions before analysis
- **recommend**: Get recommended statistical tests based on data description

---

## New in v1.1.0

### Nonlinear Regression
The `regression` command now supports nonlinear models via the `model` parameter:
- `exponential`: y = a * exp(b * x)
- `power`: y = a * x^b
- `logarithmic`: y = a + b * ln(x)
- `sigmoid`: y = 1 / (1 + exp(-(a + b*x)))

```bash
echo '{"command":"regression","params":{"x":[1,2,3,4,5],"y":[2.7,7.4,20.1,54.6,148.4],"model":"exponential"}}' | python main.py
```

### ANOVA Enhanced Output
The `anova` command now returns additional fields:
- `omega_squared`: Effect size measure (proportion of variance explained)
- `df_between`, `df_within`, `df_total`: Degrees of freedom breakdown
- `ss_between`, `ss_within`, `ss_total`: Sum of squares breakdown
- `ms_between`, `ms_within`: Mean squares breakdown

### DOE Factor Formats
The `doe` command accepts factors in three formats:
- **int**: `{"factors": 3}` -- uses 3 auto-named factors at 2 levels
- **list**: `{"factors": ["temp", "pressure", "time"]}` -- named factors at 2 levels
- **low/high**: `{"factors": {"temp": [100, 200], "pressure": [1, 5]}}` -- explicit level ranges

### Chart Generation
Several commands support chart generation via `chart: true`. Charts are saved as PNG files.

### 6-Digit Precision
Numeric output precision is now 6 decimal places by default (previously 4). Configurable via `precision` parameter.

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

## Chart Generation

Add `"chart": true` to any command that supports visualization. Charts are saved as PNG files in the current directory.

### Supported Commands
- `control_chart`: X-bar, R, I-MR charts with control limits
- `regression`: Scatter plot with fitted line and confidence interval
- `capability`: Process distribution vs spec limits
- `correlation`: Correlation heatmap
- `anova`: Box plots per group

### Example

```bash
echo '{"command":"regression","params":{"x":[1,2,3,4,5],"y":[2.1,4.0,5.9,8.1,10.0],"chart":true}}' | python main.py
```

Output includes the chart filename in the response:
```json
{
  "status": "success",
  "data": {
    "slope": 1.98,
    "intercept": 0.08,
    "chart": "regression_20260610_120000.png"
  }
}
```

---

## File Input

Commands that accept data from files use these parameters:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file` | Yes* | Path to data file (Excel, CSV, JSON, or text) |
| `column` | No | Column name to extract (for multi-column files) |
| `sheet` | No | Sheet name or index for Excel files (default: first sheet) |
| `header` | No | Row number to use as column headers (default: auto-detect). Set `0` for no header row |

*Either `file` or `values` must be provided.

### Examples

```bash
# Read a specific column from CSV
echo '{"command":"descriptive","params":{"file":"data.csv","column":"weight"}}' | python main.py

# Read from Excel with specific sheet and header row
echo '{"command":"descriptive","params":{"file":"data.xlsx","column":"weight","sheet":"Sheet2","header":1}}' | python main.py

# Read text file (one value per line, no header)
echo '{"command":"descriptive","params":{"file":"measurements.txt","header":0}}' | python main.py
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
| Regression | `regression` | Regression analysis (linear, polynomial, nonlinear) |
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
Nonlinear relationship -> regression with exponential/power/logarithmic/sigmoid model
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
  "version": "1.1.0",
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
| Excel | .xlsx, .xls | Multi-sheet with `sheet` parameter |
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

587 tests covering all commands and edge cases.

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
├── stats_engine/           # 27 command modules
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
├── tests/                  # Pytest test suite (587 tests)
├── .github/workflows/      # CI pipeline
└── .gitignore
```

---

## License

MIT

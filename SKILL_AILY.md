---
name: stats-cli-py
description: "制造业/质量工程统计分析CLI工具。基础: 描述统计、正态性、异常值。检验: t检验、ANOVA、卡方、非参数、等价检验、功效分析。回归: 线性/多项式/逻辑/非线性/PLS/GLM/LASSO/Ridge。SPC: 控制图(Xbar/R/IMR/EWMA/CUSUM/Hotelling T²/Z-MR)、过程能力(Cp/Cpk/Pp/Ppk)。质量: MSA/Gage R&R、可靠性(Weibull/ALT/Crow-AMSAA)、DOE(全因子/分数因子/RSM/Taguchi/DSD)、验收抽样。多元: PCA、聚类、判别、因子分析、MANOVA。贝叶斯估计/检验、FDA(基函数/平滑/导数/FPCA)、数据挖掘(分类/异常检测/关联规则)、蒙特卡洛/Sobol敏感性分析。NOT: 文本/图像/流数据。"
---

# stats-cli-py

Pure Python statistical analysis tool for manufacturing and quality engineering. All I/O is JSON.

**For AI Agents:** Call `discover` first to list all available commands with parameters and examples.

## Quick Start

```bash
# Pipe JSON via stdin
echo '{"command":"descriptive","params":{"values":[1,2,3]}}' | python main.py

# From Python code
from main import handler
result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
```

## All Commands (43 commands)

### Basic Statistics
```python
{"command": "descriptive", "params": {"values": [10.2, 10.5, 10.3]}}
{"command": "normality", "params": {"values": [10.2, 10.5, 10.3]}}
{"command": "outlier", "params": {"values": [10.2, 10.5, 10.3], "method": "grubbs"}}
```

### Hypothesis Testing
```python
{"command": "ttest", "params": {"test_type": "two_sample", "values": [10.2, 10.5], "values2": [11.3, 11.5]}}
{"command": "anova", "params": {"anova_type": "one_way", "groups": [[10.2, 10.5], [11.3, 11.5]]}}
{"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": [10.2, 10.5], "y": [11.3, 11.5]}}
{"command": "equivalence", "params": {"test_type": "tost", "values": [10.2, 10.5], "values2": [10.3, 10.6], "delta": 0.5}}
{"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}}
```

### Regression
```python
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}}
{"command": "regression", "params": {"x": [[1,2],[3,4],[5,6]], "y": [1,2,3], "reg_type": "pls"}}
{"command": "regression", "params": {"x": [[1],[2],[3],[4],[5]], "y": [2,4,6,8,10], "reg_type": "poisson"}}
{"command": "correlation", "params": {"x": [1, 2, 3], "y": [2, 4, 6]}}
```

### SPC / Quality Control
```python
{"command": "control_chart", "params": {"chart_type": "imr", "values": [10.2, 10.5, 10.3]}}
{"command": "capability", "params": {"values": [10.1,10.2,10.0,10.3], "usl": 11.0, "lsl": 9.0}}
{"command": "trend", "params": {"values": [10.1,10.2,10.0,10.3,10.1], "test_type": "cusum"}}
```

### DOE
```python
{"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "A", "levels": 3}, {"name": "B", "levels": 2}]}}
{"command": "doe", "params": {"doe_type": "definitive_screening", "factors": [{"name": "A", "low": -1, "high": 1}, {"name": "B", "low": -1, "high": 1}, {"name": "C", "low": -1, "high": 1}]}}
```

### MSA / Gage R&R
```python
{"command": "gage_rr", "params": {"analysis_type": "crossed", "measurements": [[10,11,10],[11,12,11]], "parts": ["P1","P2"], "operators": ["A","B"]}}
```

### Reliability
```python
{"command": "reliability", "params": {"analysis_type": "weibull", "times": [100, 200, 300], "status": [1, 1, 1]}}
{"command": "reliability", "params": {"analysis_type": "alt", "stress_levels": [350, 400, 450], "failure_times": [1000, 500, 200], "stress_model": "arrhenius", "use_stress": 300}}
```

### Multivariate
```python
{"command": "multivariate", "params": {"analysis_type": "pca", "columns": ["x1", "x2"], "file": "data.csv"}}
{"command": "multivariate", "params": {"analysis_type": "factor_analysis", "values": [[1,2],[3,4],[5,6]], "n_factors": 1}}
```

### Bayesian
```python
{"command": "bayesian", "params": {"analysis_type": "ttest", "values": [10.2, 10.5, 10.3], "values2": [11.3, 11.5, 11.1]}}
{"command": "bayesian", "params": {"analysis_type": "proportion", "successes": 7, "n": 10}}
```

### Distribution Analysis
```python
{"command": "distribution", "params": {"analysis_type": "select", "values": [1.2, 1.5, 1.3, 1.8, 2.1, 1.9]}}
```

### Data Mining
```python
{"command": "mining", "params": {"analysis_type": "classify", "features": [[1,2],[3,4],[5,6],[7,8]], "labels": [0,0,1,1]}}
{"command": "mining", "params": {"analysis_type": "anomaly", "values": [10, 11, 10, 12, 100]}}
{"command": "mining", "params": {"analysis_type": "associate", "transactions": [["a","b"],["b","c"],["a","b","c"]]}}
```

### Acceptance Sampling
```python
{"command": "acceptance_sampling", "params": {"analysis_type": "single_plan", "n": 50, "c": 2, "defect_rate": 0.05}}
{"command": "acceptance_sampling", "params": {"analysis_type": "find_plan", "AQL": 0.01, "LTPD": 0.05}}
```

### Sensitivity Analysis
```python
{"command": "sensitivity", "params": {"analysis_type": "monte_carlo", "inputs": {"x": {"dist": "normal", "params": {"mean": 10, "std": 1}}}, "formula": "x * 2", "n_simulations": 10000}}
{"command": "sensitivity", "params": {"analysis_type": "sobol", "inputs": {"x1": {"dist": "uniform", "params": {"low": 0, "high": 1}}}, "formula": "x1", "n_simulations": 5000}}
```

### Functional Data Analysis
```python
{"command": "functional", "params": {"analysis_type": "basis", "t": [0,0.5,1.0], "values": [0,1,0], "basis_type": "bspline", "n_basis": 3}}
{"command": "functional", "params": {"analysis_type": "fpca", "curves": [[1,2,3],[2,3,4]], "t": [0,0.5,1.0], "n_components": 1}}
```

### Advanced
```python
{"command": "advanced", "params": {"analysis_type": "bootstrap", "values": [1,2,3,4,5,6,7,8,9,10], "statistic": "mean"}}
{"command": "advanced", "params": {"analysis_type": "exact_test", "observed": [[10, 5], [3, 12]]}}
```

### Data Processing
```python
{"command": "clean", "params": {"values": [10.1, 10.2, null, 10.3], "method": "impute_mean"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6], "method": "boxcox"}}
```

### Workflow
```python
{"command": "workflow", "params": {"steps": [{"command": "descriptive"}, {"command": "normality"}], "values": [10.2, 10.5, 10.3]}}
{"command": "discover", "params": {}}
```

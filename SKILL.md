---
name: stats-cli-py
description: "Use when user needs statistical analysis, SPC control charts, process capability, hypothesis testing, regression, DOE, outlier detection, trend analysis, MSA/Gage R&R, data cleaning, data transformation, reliability analysis, multivariate analysis, time series, power analysis, chi-square tests, workflow automation, PDF/Excel export, or assumption checking. NOT for: real-time streaming, text analysis, image data, non-numeric datasets, database queries. Triggers: 统计分析, 控制图, 过程能力, t检验, ANOVA, 回归, DOE, 正态性检验, 异常值, 趋势分析, SPC, Cp, Cpk, capability, normality, regression, correlation, outlier, quality, manufacturing, 质量分析, 过程控制, 假设检验, 方差分析, 实验设计, MSA, Gage R&R, 测量系统分析, 数据清洗, 数据变换, Box-Cox, 可靠性, Weibull, 生存分析, PCA, 聚类, 判别分析, 时间序列, ARIMA, 功效分析, 样本量, 卡方检验, chi-square, 工作流, workflow, 假设检查, 方法推荐, PDF导出, Excel导出."
---

# stats-cli-py

Pure Python statistical analysis tool for manufacturing and quality engineering, powered by scipy + statsmodels. All I/O is JSON — designed to be called as an AI-agent skill.

**For AI Agents:** Not sure which command to use? Call `discover` first — it returns all available commands with their parameters and examples. Use `discover {"command_name": "xxx"}` to get required/optional parameter details before constructing your request.

## Installation

```bash
pip install -r requirements.txt
```

Pure Python — no R, no external runtime. Dependencies: scipy, statsmodels, pandas, numpy, openpyxl, scikit-learn, matplotlib, fpdf2.

### Invocation

```bash
# Pipe JSON via stdin
echo '{"command":"descriptive","params":{"values":[1,2,3]}}' | python main.py

# From a JSON file
python main.py input.json

# From Python code
from main import handler
result = handler({"command": "descriptive", "params": {"values": [1, 2, 3]}})
```

### Chart Generation

Add `"chart": true` to params to get a base64 PNG chart in the response. Supported commands: `descriptive`, `normality`, `control_chart`, `capability`, `correlation`, `regression`, `timeseries`, `report`, `ttest`, `anova`, `homogeneity`, `multiple_comparison`, `equivalence`, `power`, `multivariate`, `trend`, `outlier`, `reliability`, `gage_rr`, `nonparametric`, `explore`, `doe`.

```python
{"command": "descriptive", "params": {"values": [1, 2, 3, 4, 5], "chart": true}}
# Response includes: "chart_base64": "iVBORw0KGgo..."
```

### File-Based Input

Most commands accept `"file"` instead of `"values"` to load data from Excel/CSV/JSON/text files:

```python
{"command": "descriptive", "params": {"file": "data.xlsx", "column": "weight", "sheet": "Batch1"}}
{"command": "control_chart", "params": {"chart_type": "imr", "file": "data.csv", "column": "measurement"}}
{"command": "capability", "params": {"file": "data.xlsx", "column": "weight", "usl": 11.0, "lsl": 9.0}}
```

Parameters: `file` (path), `column` (name or index), `sheet` (Excel sheet name/index), `header` (row number for header, `null` for no header, default `0`).

---

## Quick Start: Don't Know What Analysis to Use?

**不确定用什么命令？先调用 `discover` 获取帮助：**

```python
{"command": "discover"}                                    # 列出所有命令
{"command": "discover", "params": {"category": "spc"}}     # 按类别筛选
{"command": "discover", "params": {"command_name": "ttest"}}  # 查看参数详情（含 required/type）
```

**Intent-to-Command 快速查找：**

| 用户意图 | 推荐命令 | 备注 |
|----------|----------|------|
| 描述数据特征 | `descriptive` | 均值、中位数、标准差、置信区间 |
| 检查是否正态 | `normality` | Shapiro-Wilk, Anderson-Darling |
| 比较两组数据 | `ttest` / `nonparametric` | 正态用 ttest，非正态用 nonparametric |
| 比较多组数据 | `anova` | 正态+等方差用 anova，否则用 nonparametric |
| 检查过程能力 | `capability` | 需要 USL/LSL |
| 监控过程稳定性 | `control_chart` | imr/xbar/r/p/np/c/u/ewma/cusum |
| 检测异常值 | `outlier` | grubbs/dixon/iqr/zscore |
| 分析相关性 | `correlation` | pearson/spearman/kendall |
| 回归/预测 | `regression` | linear/polynomial/logistic 等 10 种 |
| 时间序列分析 | `timeseries` | exp_smoothing/arima/decomposition |
| 趋势检测 | `trend` | cusum/ewma/runs |
| 测量系统分析 | `gage_rr` | crossed/nested/attribute |
| 可靠性分析 | `reliability` | weibull/kaplan_meier |
| 实验设计 | `doe` | full_factorial/fractional/taguchi |
| 多变量分析 | `multivariate` | pca/cluster/discriminant |
| 功效/样本量 | `power` | t_test/anova/proportion |
| 数据清洗 | `clean` | drop/impute/winsorize/clip |
| 数据变换 | `transform` | log/sqrt/boxcox/johnson |
| 不确定用什么 | `recommend` | 描述目标，工具推荐方法 |

**Follow this decision tree:**

1. **用户描述模糊时** → 先用 `explore` 查看数据结构
2. **比较数据** → 看下面的"比较分析决策树"
3. **关系分析** → 看下面的"关系分析决策树"
4. **预测/趋势** → 看下面的"预测分析决策树"
5. **质量控制** → 看下面的"质量控制决策树"

---

## 智能引导流程（用户模糊请求时）

**当用户只说"帮我分析数据"、"看看这个文件"等模糊请求时，按以下流程引导：**

```
用户上传文件 / 说"帮我分析数据"
    │
    ├── 第1步：explore 查看数据结构
    │   ├── 多 sheet？
    │   │   ├── 是 → 返回所有 sheet 名称，问用户："文件包含 N 个 sheet：[列表]，请指定要分析哪个 sheet"
    │   │   └── 否 → 继续
    │   ├── 多列？
    │   │   ├── 有多个数值列 → 列出列名和基本统计，问用户："发现以下数值列：[列表]，请指定要分析哪列"
    │   │   └── 只有1列 → 自动选中
    │   └── 无数值列 → 提示："未发现数值数据，请检查文件格式"
    │
    ├── 第2步：descriptive + normality（必做，建立数据基线）
    │   ├── descriptive：n, mean, std, RSD%, min, max, range, CI
    │   └── normality：Shapiro-Wilk + Anderson-Darling + Lilliefors → is_normal
    │
    ├── 第3步：根据数据特征推荐分析方向
    │   │
    │   ├── 有规格限（USL/LSL）？
    │   │   └── 推荐：capability（过程能力）+ control_chart（控制图）
    │   │
    │   ├── 用户提到"比较"、"差异"、"两批"？
    │   │   └── 推荐：ttest（正态）或 nonparametric（非正态）
    │   │
    │   ├── 用户提到"趋势"、"稳定性"、"监控"？
    │   │   └── 推荐：control_chart + trend
    │   │
    │   ├── 用户提到"关系"、"相关"、"影响"？
    │   │   └── 推荐：correlation + regression
    │   │
    │   ├── 用户提到"预测"、"未来"？
    │   │   └── 推荐：timeseries
    │   │
    │   └── 不确定？
    │       └── 输出数据摘要 + 列出所有适用的分析类型，让用户选择
    │
    └── 第4步：执行推荐的分析，输出结果和解读
```

**关键原则：**
- 永远先 `explore` 再分析，不要假设数据结构
- 多 sheet 文件必须让用户选择 sheet，不要默认第一个
- `explore` 返回的 `sheets` 字段包含所有可用 sheet 名称
- `explore` 返回的 `hint` 字段提示用户还有其他 sheet
- `load_data` 返回的 `_available_sheets` 字段也可用于发现 sheet

---

## Decision Tree 1: 比较分析（两组或多组数据比较）

```
用户想比较数据
    │
    ├── 只有2组？
    │   ├── 数据是正态的？ → ttest (先用 normality 检查)
    │   │   ├── 等方差？ → two_sample t-test
    │   │   └── 不等方差？ → two_sample t-test (自动检测)
    │   └── 数据非正态？ → nonparametric mann_whitney
    │       └── 也考虑：transform boxcox → 重新检验正态性 → 若正态则用 ttest
    │
    ├── 有3+组？
    │   ├── 数据是正态的？ → anova one_way
    │   │   └── 显著？ → multiple_comparison tukey
    │   └── 数据非正态？ → nonparametric kruskal_wallis
    │       └── 也考虑：transform boxcox → 重新检验正态性 → 若正态则用 anova
    │
    └── 配对数据？（同一样本前后对比）
        ├── 正态？ → ttest paired
        └── 非正态？ → nonparametric wilcoxon
            └── 也考虑：transform boxcox → 重新检验正态性 → 若正态则用 ttest paired
```

**前置检查流程：**
```python
# 1. 先检查正态性
{"command": "normality", "params": {"values": [10.2, 10.5, ...]}}

# 2. 根据结果选择检验方法
# 如果 p > 0.05 (正态) → 用参数检验 (ttest, anova)
# 如果 p <= 0.05 (非正态) → 用非参数检验 (nonparametric)
```

---

## Decision Tree 2: 关系分析（变量之间是否有关系）

```
用户想分析变量关系
    │
    ├── 两个连续变量？
    │   ├── 看相关性 → correlation
    │   └── 看因果关系 → regression linear
    │
    ├── 多个自变量影响一个因变量？
    │   ├── 线性关系 → regression multiple
    │   └── 自动选择重要变量 → regression stepwise
    │
    ├── 因变量是分类变量（0/1）？
    │   └── regression logistic
    │
    └── 多个变量同时分析？
        ├── 降维 → multivariate pca
        ├── 分组 → multivariate cluster
        └── 分类 → multivariate discriminant
```

---

## Decision Tree 3: 预测/趋势分析

```
用户想预测或分析趋势
    │
    ├── 数据有季节性？
    │   ├── 有 → timeseries decomposition (frequency=周期长度)
    │   └── 无 → 继续
    │
    ├── 想预测未来值？
    │   ├── 短期预测 → timeseries exp_smoothing
    │   └── 长期预测 → timeseries arima
    │
    └── 想检测趋势是否存在？
        └── trend (CUSUM/EWMA/Runs test)
```

---

## Decision Tree 4: 质量控制

```
用户想做质量控制/SPC
    │
    ├── 监控过程稳定性？
    │   ├── 个体值 → control_chart imr
    │   ├── 子组均值 → control_chart xbar
    │   └── 计数数据 → control_chart p/np/c/u
    │
    ├── 评估过程能力？
    │   ├── 先检查正态性
    │   ├── 正态 → capability (Cp/Cpk)
    │   └── 非正态 → capability capability_type=boxcox
    │
    ├── 验证测量系统？
    │   └── gage_rr crossed/nested/attribute
    │
    └── 分析失效/寿命？
        └── reliability weibull/stability
```

---

## Scenario-Based Workflows

### 场景1: "帮我分析这组数据"

用户给了你一堆数据，不知道从哪开始。标准流程：explore → descriptive → normality → 下一步。

```python
# Step 1: 查看数据结构（如果是文件）
{"command": "explore", "params": {"file": "data.xlsx"}}

# Step 2: 描述性统计 — 均值、标准差、分布概况
{"command": "descriptive", "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2, 10.5, 10.4]}}

# Step 3: 正态性检验 — 决定后续用什么方法
{"command": "normality", "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2, 10.5, 10.4]}}

# Step 4: 根据结果选择下一步
# 正态 (p > 0.05) → 可以做 t-test, ANOVA, capability
# 非正态 (p <= 0.05) → 用 nonparametric 或 transform boxcox 后再分析
# 有异常值？→ 先用 outlier 检测
```

### 场景2: "比较两批物料是否一致"

两批样品对比，核心问题是"差异是否显著"。

```python
# Step 1: 两组数据分别检查正态性
{"command": "normality", "params": {"values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3]}}
{"command": "normality", "params": {"values": [10.8, 11.0, 10.9, 10.7, 11.1, 10.8, 10.9]}}

# Step 2a: 两组都正态 → t-test（自动检测等方差）
{"command": "ttest", "params": {
    "test_type": "two_sample",
    "values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3],
    "values2": [10.8, 11.0, 10.9, 10.7, 11.1, 10.8, 10.9]
}}

# Step 2b: 非正态 → Mann-Whitney U 检验
{"command": "nonparametric", "params": {
    "test_type": "mann_whitney",
    "x": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3],
    "y": [10.8, 11.0, 10.9, 10.7, 11.1, 10.8, 10.9]
}}

# Step 3 (可选): 如果需要证明"等效"而非"不同" → 等效性检验
{"command": "equivalence", "params": {
    "test_type": "tost",
    "values": [10.2, 10.5, 10.3, 10.1, 10.4],
    "values2": [10.3, 10.4, 10.2, 10.5, 10.3],
    "delta": 0.5
}}
```

### 场景3: "评估工艺稳定性"

工艺是否稳定可控，需要控制图 + 过程能力 + 趋势分析三件套。

```python
# Step 1: 控制图 — 过程是否受控
{"command": "control_chart", "params": {
    "chart_type": "imr",
    "values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2, 10.5, 10.4,
               10.3, 10.2, 10.4, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2]
}}

# Step 2: 过程能力 — 是否满足规格要求
{"command": "capability", "params": {
    "values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2, 10.5, 10.4,
               10.3, 10.2, 10.4, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2],
    "usl": 11.0,
    "lsl": 9.0
}}

# Step 3: 趋势分析 — 是否有渐变/漂移
{"command": "trend", "params": {
    "values": [10.2, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2, 10.5, 10.4,
               10.3, 10.2, 10.4, 10.5, 10.3, 10.1, 10.4, 10.6, 10.3, 10.2],
    "test_type": "cusum"
}}

# 如果数据非正态，先做 Box-Cox 变换再算能力
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "boxcox"}}
```

### 场景4: "验证测量系统"

MSA/Gage R&R — 判断测量系统的重复性和再现性是否可接受。

```python
# Gage R&R 交叉实验（最常用：3个操作员 × 10个零件 × 3次测量）
{"command": "gage_rr", "params": {
    "analysis_type": "crossed",
    "measurements": [10.1, 10.2, 10.1, 10.3, 10.2, 10.1, 10.4, 10.3, 10.2,
                     10.2, 10.3, 10.2, 10.5, 10.4, 10.3, 10.1, 10.2, 10.1,
                     10.3, 10.4, 10.3, 10.2, 10.1, 10.2, 10.4, 10.5, 10.4,
                     10.1, 10.0, 10.1, 10.3, 10.2, 10.3, 10.2, 10.1, 10.2,
                     10.5, 10.4, 10.5, 10.3, 10.2, 10.3, 10.4, 10.3, 10.4,
                     10.2, 10.1, 10.2, 10.4, 10.3, 10.4, 10.1, 10.0, 10.1,
                     10.3, 10.4, 10.3, 10.5, 10.4, 10.5, 10.2, 10.1, 10.2,
                     10.4, 10.3, 10.4, 10.2, 10.1, 10.2, 10.3, 10.2, 10.3,
                     10.5, 10.6, 10.5, 10.3, 10.4, 10.3, 10.4, 10.5, 10.4,
                     10.1, 10.2, 10.1, 10.3, 10.2, 10.3, 10.2, 10.1, 10.2],
    "parts": [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,
              1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,
              1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10],
    "operators": ["A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A","A",
                  "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
                  "C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C","C"],
    "tolerance": 1.0
}}

# 结果解读：%GR&R < 10% 可接受, 10-30% 有条件接受, > 30% 不可接受
# ndc (可区分类别数) >= 5 表示测量系统有足够分辨力
```

### 场景5: "预测未来趋势"

基于历史数据预测未来走势。

```python
# Step 1: ACF 分析 — 查看数据模式（自相关性、周期性）
{"command": "timeseries", "params": {
    "analysis_type": "acf",
    "values": [10, 12, 11, 13, 14, 12, 15, 16, 14, 17, 18, 16, 19, 20, 18],
    "max_lag": 10
}}

# Step 2a: 短期预测 → 指数平滑
{"command": "timeseries", "params": {
    "analysis_type": "exp_smoothing",
    "values": [10, 12, 11, 13, 14, 12, 15, 16, 14, 17, 18, 16, 19, 20, 18],
    "n_forecast": 6
}}

# Step 2b: 长期预测 / 有季节性 → ARIMA
{"command": "timeseries", "params": {
    "analysis_type": "arima",
    "values": [10, 12, 11, 13, 14, 12, 15, 16, 14, 17, 18, 16, 19, 20, 18],
    "n_forecast": 12
}}

# Step 2c: 有明显季节性 → 先分解再预测
{"command": "timeseries", "params": {
    "analysis_type": "decomposition",
    "values": [100, 120, 110, 130, 105, 125, 115, 135, 108, 128, 118, 138,
               110, 130, 120, 140, 115, 135, 125, 145, 118, 138, 128, 148],
    "frequency": 12
}}
```

---

## All Commands (33 commands)

### Data Exploration
```python
{"command": "explore", "params": {"file": "data.xlsx"}}
{"command": "discover"}
{"command": "discover", "params": {"command_name": "capability"}}
{"command": "discover", "params": {"category": "spc"}}
```

### Basic Statistics
```python
{"command": "descriptive", "params": {"values": [10.2, 10.5, 10.3]}}
{"command": "normality", "params": {"values": [10.2, 10.5, 10.3]}}
{"command": "outlier", "params": {"values": [10.2, 10.5, 10.3], "method": "grubbs"}}
```

### Hypothesis Testing
```python
{"command": "ttest", "params": {"test_type": "one_sample", "values": [10.2, 10.5], "mu": 10.0}}
{"command": "ttest", "params": {"test_type": "two_sample", "values": [10.2, 10.5], "values2": [11.3, 11.5]}}
{"command": "ttest", "params": {"test_type": "paired", "values": [10.2, 10.5], "values2": [10.8, 11.0]}}
{"command": "anova", "params": {"anova_type": "one_way", "groups": [[10.2, 10.5], [11.3, 11.5], [12.1, 12.4]]}}
{"command": "anova", "params": {"anova_type": "two_way", "groups": [], "data": {"factor_a": ["A","A","A","B","B","B"], "factor_b": ["X","Y","X","Y","X","Y"], "values": [10,12,11,15,14,16]}}}
{"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": [10.2, 10.5], "y": [11.3, 11.5]}}
{"command": "nonparametric", "params": {"test_type": "kruskal_wallis", "groups": [[10.2, 10.5], [11.3, 11.5]]}}
{"command": "nonparametric", "params": {"test_type": "wilcoxon", "x": [10.2, 10.5], "y": [10.8, 11.0]}}
{"command": "nonparametric", "params": {"test_type": "chi_square", "observed": [50, 30, 20]}}
{"command": "nonparametric", "params": {"test_type": "friedman", "groups": [[10.2, 10.5], [11.3, 11.5], [12.1, 12.4]]}}
{"command": "homogeneity", "params": {"test_type": "levene", "groups": [[10.2, 10.5], [11.3, 11.5]]}}
{"command": "equivalence", "params": {"test_type": "tost", "values": [10.2, 10.5], "values2": [10.3, 10.6], "delta": 0.5}}
{"command": "multiple_comparison", "params": {"test_type": "tukey", "groups": [[10.2, 10.5], [11.3, 11.5], [12.1, 12.4]]}}
{"command": "power", "params": {"analysis_type": "t_test", "effect_size": 0.5, "power": 0.80}}
```

### Regression
```python
{"command": "regression", "params": {"x": [1, 2, 3], "y": [2, 4, 6]}}
{"command": "regression", "params": {"x": [1, 2, 3, 4], "y": [2, 4, 8, 16], "reg_type": "quadratic"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4], "y": [2, 4, 8, 16], "reg_type": "polynomial", "degree": 3}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [0, 0, 1, 1, 1], "reg_type": "logistic"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2.7, 7.4, 20, 54, 148], "reg_type": "exponential"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25], "reg_type": "power"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [0, 0.69, 1.1, 1.39, 1.61], "reg_type": "logarithmic"}}
{"command": "regression", "params": {"x": [1,2,3,4,5,6,7,8,9,10], "y": [0.1,0.1,0.2,0.5,0.8,0.9,0.95,0.98,0.99,1.0], "reg_type": "sigmoid"}}
{"command": "regression", "params": {"x_columns": ["x1", "x2"], "y_column": "y", "reg_type": "multiple", "file": "data.csv"}}
{"command": "regression", "params": {"x_columns": ["x1", "x2"], "y_column": "y", "reg_type": "stepwise", "file": "data.csv"}}
{"command": "correlation", "params": {"x": [1, 2, 3], "y": [2, 4, 6]}}
```

### SPC / Quality Control
```python
{"command": "control_chart", "params": {"chart_type": "imr", "values": [10.2, 10.5, 10.3]}}
{"command": "control_chart", "params": {"chart_type": "xbar", "values": [[10.1,10.2,10.0],[10.3,10.1,10.2],[10.0,10.1,10.3],[10.2,10.0,10.1],[10.1,10.3,10.2]], "subgroup_size": 3}}
{"command": "control_chart", "params": {"chart_type": "p", "values": [5,3,4,6,2,3,5,4,3,4], "subgroup_size": 100}}
{"command": "capability", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "usl": 11.0, "lsl": 9.0}}
{"command": "capability", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "usl": 11.0, "lsl": 9.0, "target": 10.0}}
{"command": "trend", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "test_type": "cusum"}}
{"command": "trend", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "test_type": "ewma"}}
{"command": "trend", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "test_type": "runs"}}
```

### DOE (Design of Experiments)

Factor formats: `levels: int` (number of levels), `levels: list` (explicit values), `low/high` (two levels).

```python
{"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "Temp", "levels": 3}, {"name": "Time", "levels": 2}]}}
{"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "Temp", "levels": [100, 150, 200]}, {"name": "Time", "levels": [30, 60]}]}}
{"command": "doe", "params": {"doe_type": "full_factorial", "factors": [{"name": "Temp", "low": 100, "high": 200}, {"name": "Time", "low": 30, "high": 60}]}}
{"command": "doe", "params": {"doe_type": "fractional_factorial", "factors": [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}, {"name": "C", "levels": 2}]}}
{"command": "doe", "params": {"doe_type": "taguchi", "factors": [{"name": "A", "levels": 3}, {"name": "B", "levels": 3}]}}
```

### MSA / Measurement System Analysis
```python
{"command": "gage_rr", "params": {"analysis_type": "crossed", "measurements": [[10,11,10],[11,12,11],[10,11,10]], "parts": ["P1","P2","P3"], "operators": ["A","B","A"], "tolerance": 10.0}}
{"command": "gage_rr", "params": {"analysis_type": "nested", "measurements": [[10,11,10],[11,12,11],[10,11,10]], "parts": ["P1","P2","P3"], "operators": ["A","B","A"]}}
{"command": "gage_rr", "params": {"analysis_type": "attribute", "measurements": [[1,0,1],[0,1,0],[1,1,0]], "parts": ["P1","P2","P3"], "operators": ["A","B","A"]}}
```

### Reliability / Survival Analysis
```python
{"command": "reliability", "params": {"analysis_type": "weibull", "times": [100, 200, 300], "status": [1, 1, 1]}}
{"command": "reliability", "params": {"analysis_type": "kaplan_meier", "times": [100, 200, 300], "status": [1, 0, 1]}}
{"command": "reliability", "params": {"analysis_type": "distribution", "times": [100, 200, 300], "status": [1, 1, 1]}}
{"command": "reliability", "params": {"analysis_type": "stability", "times": [0, 3, 6, 9, 12], "values": [100, 99, 98, 97, 96], "lsl": 90}}
```

### Multivariate Analysis
```python
{"command": "multivariate", "params": {"analysis_type": "pca", "columns": ["x1", "x2", "x3"], "file": "data.csv"}}
{"command": "multivariate", "params": {"analysis_type": "cluster", "method": "kmeans", "n_clusters": 3, "file": "data.csv"}}
{"command": "multivariate", "params": {"analysis_type": "discriminant", "columns": ["x1", "x2"], "group_column": "group", "file": "data.csv"}}
{"command": "multivariate", "params": {"analysis_type": "correlation_matrix", "file": "data.csv"}}
```

### Time Series
```python
{"command": "timeseries", "params": {"analysis_type": "exp_smoothing", "values": [10, 12, 11, 13, 14], "frequency": 4}}
{"command": "timeseries", "params": {"analysis_type": "arima", "values": [10,12,11,13,14,13,15,14,16,15,17,16], "n_forecast": 12}}
{"command": "timeseries", "params": {"analysis_type": "decomposition", "values": [10,12,11,13,14,13,15,14,16,15,17,16], "frequency": 12}}
{"command": "timeseries", "params": {"analysis_type": "acf", "values": [10,12,11,13,14,13,15,14,16,15,17,16], "max_lag": 20}}
```

### Advanced Statistical Methods
```python
{"command": "advanced", "params": {"analysis_type": "exact_test", "observed": [[10, 5], [3, 12]]}}
{"command": "advanced", "params": {"analysis_type": "mcnemar", "observed": [[10, 5], [15, 20]]}}
{"command": "advanced", "params": {"analysis_type": "cochran_q", "data": [[1,0,1,1,0], [1,1,0,1,1], [0,1,1,0,1]]}}
{"command": "advanced", "params": {"analysis_type": "mixed_effects", "groups": [[10,11,10],[11,12,11],[10,11,10]], "group_ids": ["A","B","A"]}}
```

### Data Processing
```python
{"command": "clean", "params": {"values": [10.1, 10.2, null, 10.3, 10.1], "method": "impute_mean"}}
{"command": "clean", "params": {"values": [10.1, 10.2, null, 10.3, 10.1], "method": "impute_median"}}
{"command": "clean", "params": {"values": [10.1, 10.2, 100.0, 10.3, 10.1], "method": "winsorize"}}
{"command": "clean", "params": {"values": [10.1, 10.2, 100.0, 10.3, 10.1], "method": "clip"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "boxcox"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "log"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "sqrt"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "johnson"}}
{"command": "transform", "params": {"values": [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.3], "method": "standardize"}}
```

### Reporting
```python
{"command": "report", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "usl": 11.0, "lsl": 9.0}}
```

### Custom Script
```python
{"command": "run", "params": {"script": "result = {'sum': sum(data['values']), 'count': len(data['values'])}", "data": {"values": [1, 2, 3]}}}
```

**Sandbox restrictions:** The `run` command executes in a restricted environment:
- Available builtins: `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `frozenset`, `int`, `isinstance`, `len`, `list`, `map`, `max`, `min`, `print`, `range`, `round`, `set`, `sorted`, `str`, `sum`, `tuple`, `type`, `zip`
- NOT available: `__import__`, `open`, `exec`, `eval`, `getattr`, `compile`, `globals`, `locals`
- No file I/O, no network access, no module imports
- Set `result` variable to return data from your script

### Workflow Automation (工作流自动化)

```python
# Multi-step workflow with automatic assumption checking
{"command": "workflow", "params": {
    "steps": [
        {"command": "clean", "params": {"method": "drop"}},
        {"command": "descriptive"},
        {"command": "normality"},
        {"command": "capability", "params": {"usl": 11.0, "lsl": 9.0}}
    ],
    "values": [10.2, 10.5, 10.3, 10.1, 10.4],
    "auto_check": true
}}

**`auto_check` behavior:**
- When `auto_check: true`, the tool checks statistical assumptions between steps
- If assumptions fail, the step still executes but the response includes `warnings` with recommendations
- The workflow continues even if a step fails — check `steps_results[i].status` for each step
- Failed steps return `{"status": "error", ...}` in their position in `steps_results`
- Use `recommendations` field to decide whether to retry with different parameters

# Predefined workflow templates
{"command": "workflow_template", "params": {
    "template_name": "manufacturing",
    "values": [10.2, 10.5, 10.3, 10.1, 10.4],
    "usl": 11.0, "lsl": 9.0
}}

# Check statistical assumptions before analysis
{"command": "check_assumptions", "params": {
    "values": [10.2, 10.5, 10.3, 10.1, 10.4],
    "test_type": "ttest"
}}

# Get recommended statistical test
{"command": "recommend", "params": {
    "data_description": {
        "goal": "compare means of two groups",
        "n_groups": 2
    }
}}
```

**Available templates:**
- `manufacturing`: clean → descriptive → normality → capability → report
- `comparison`: clean → descriptive → normality → homogeneity → ttest/anova
- `capability`: clean → descriptive → normality → capability
- `exploration`: clean → descriptive → normality → outlier
- `reliability`: descriptive → weibull analysis
- `doe`: full factorial design → regression analysis
- `timeseries`: descriptive → exponential smoothing
- `regression`: descriptive → correlation → regression
- `multivariate`: PCA → cluster analysis

### Report Export (报告导出)

```python
# Generate report first
report_data = handler({"command": "report", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "usl": 11.0, "lsl": 9.0}})

# Export to Excel
handler({"command": "export_excel", "params": {"report_data": report_data["data"], "output_path": "report.xlsx"}})

# Export to PDF
handler({"command": "export_pdf", "params": {"report_data": report_data["data"], "output_path": "report.pdf"}})
```

---

## Output Format

All commands return a standard JSON envelope:

### Success
```json
{
  "status": "success",
  "version": "1.2.1",
  "timestamp": "2026-06-09T10:00:00Z",
  "data": {
    "total": 103.5,
    "mean": 10.35,
    "std": 0.14,
    "n": 10
  }
}
```

**Top 5 command output examples:**

**descriptive** — `{"command":"descriptive","params":{"values":[10.1,10.2,10.0,10.3,10.1]}}`
```json
{"status":"success","data":{"n":5,"mean":10.14,"median":10.1,"std":0.114,"rsd_percent":1.1243,"min":10.0,"max":10.3,"range":0.3,"q1":10.1,"q3":10.2,"iqr":0.1,"ci_95_lower":10.0,"ci_95_upper":10.28,"skewness":-0.2693,"kurtosis":-1.1333}}
```

**normality** — `{"command":"normality","params":{"values":[10.1,10.2,10.0,10.3,10.1]}}`
```json
{"status":"success","data":{"shapiro_wilk":{"statistic":0.987,"p_value":0.97},"anderson_darling":{"statistic":0.22,"critical_values":{}},"lilliefors":{"statistic":0.15,"p_value":0.8},"is_normal":true}}
```

**ttest** — `{"command":"ttest","params":{"test_type":"two_sample","values":[10.2,10.5],"values2":[11.3,11.5]}}`
```json
{"status":"success","data":{"test_type":"two_sample","t_statistic":-4.24,"p_value":0.051,"significant":false,"ci_95":[-2.26,0.02]}}
```

**capability** — `{"command":"capability","params":{"values":[10.1,10.2,10.0,10.3],"usl":11.0,"lsl":9.0}}`
```json
{"status":"success","data":{"cp":2.72,"cpk":2.57,"pp":2.72,"ppk":2.57,"rating":"Excellent","performance":{"ppm_upper":0,"ppm_lower":0}}}
```

**control_chart** — `{"command":"control_chart","params":{"chart_type":"imr","values":[10.1,10.2,10.0,10.3,10.1]}}`
```json
{"status":"success","data":{"chart_type":"imr","chart":{"points":[10.1,10.2,10.0,10.3,10.1],"center":10.14,"ucl":10.56,"lcl":9.72},"summary":{"stable":true}}}
```

**Notes:**
- All numeric values are rounded to 6 decimal places (configurable via `DEFAULT_PRECISION` in `utils/output.py`)
- `NaN` and `Inf` values in input are automatically filtered
- Check `status` before reading `data`. On error, `suggestion` tells you how to fix it.

### Error
```json
{
  "status": "error",
  "error_type": "PARAM_ERROR",
  "message": "At least 3 values are required for normality test",
  "suggestion": "Provide more data points or use descriptive command instead"
}
```

**Error types:**
| error_type | Meaning |
|------------|---------|
| `INVALID_INPUT` | JSON parse failure or malformed request |
| `MISSING_COMMAND` | No `command` field in input |
| `PARAM_ERROR` | Invalid parameters (missing required fields, wrong types, out-of-range values) |
| `DATA_ERROR` | Insufficient data, bad format, or non-numeric values |
| `COMPUTATION_ERROR` | Mathematical limitation (e.g. division by zero, singular matrix) |
| `FILE_NOT_FOUND` | Specified file does not exist |
| `MISSING_DEPENDENCY` | Python package not installed |
| `MEMORY_ERROR` | Out of memory, reduce data size |
| `INTERNAL_ERROR` | Unexpected error (includes exception type) |

Every response includes `status`, so check it before reading `data`. On error, `suggestion` tells you how to fix it.

**Error recovery workflows:**

| Error | Recovery |
|-------|----------|
| `DATA_ERROR` (insufficient data) | Try `clean` to remove NaN, or `explore` to check data size |
| `PARAM_ERROR` (missing required param) | Call `discover {"command_name": "xxx"}` to see required params |
| `FILE_NOT_FOUND` | Call `explore {"file": "path"}` to verify file exists and path is correct |
| `MISSING_DEPENDENCY` | Run `pip install -r requirements.txt`, then retry |
| `MEMORY_ERROR` | Use `file` input instead of `values`, or reduce data size |
| `INVALID_INPUT` | Validate JSON format, check for trailing commas or unquoted keys |

---

## File Support

| Format | Extension | Notes |
|--------|-----------|-------|
| Excel | .xlsx, .xls | Multi-sheet supported (use `sheet` param) |
| CSV | .csv | Auto-detect encoding (utf-8, gbk, gb2312, latin-1) and delimiter |
| JSON | .json | Structured data |
| Text | .txt | One value per line |

All file-based commands accept `file` + optional `column` and `sheet` params:
```python
{"command": "descriptive", "params": {"file": "data.xlsx", "column": "weight", "sheet": "Sheet1"}}
```

---

## Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| scipy | >= 1.10 | Core statistical tests, distributions |
| statsmodels | >= 0.14 | Time series, regression, ANOVA |
| pandas | >= 1.5 | Data loading, DataFrame operations |
| numpy | >= 1.23 | Numerical computation |
| openpyxl | >= 3.0 | Excel file support (.xlsx) |
| scikit-learn | >= 1.0 | Clustering, PCA, discriminant analysis |

Install all at once:
```bash
pip install -r requirements.txt
```

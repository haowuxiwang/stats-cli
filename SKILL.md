---
name: stats-cli-py
description: "Use when user needs statistical analysis for manufacturing/quality data. Covers: descriptive stats, hypothesis testing, regression, SPC, DOE, MSA, reliability, multivariate, Bayesian, FDA, data mining, sensitivity analysis, acceptance sampling. NOT for: text analysis, image recognition, streaming data."
---

## Intent → Command Quick Lookup

| User Intent | Command | Why |
|---|---|---|
| 看数据基本特征 | `descriptive` | 均值/中位数/标准差/分位数 |
| 数据是否正态 | `normality` | Shapiro-Wilk/Anderson-Darling |
| 两组有无差异 | `ttest` / `nonparametric` | 正态用 ttest，非正态用 Mann-Whitney |
| 多组有无差异 | `anova` / `nonparametric` | 正态用 ANOVA，非正态用 Kruskal-Wallis |
| 两变量有无关系 | `correlation` / `regression` | 相关用 correlation，因果用 regression |
| 过程是否稳定 | `control_chart` | X-bar/R/IMR/EWMA/CUSUM/T² |
| 过程能力如何 | `capability` | Cp/Cpk/Pp/Ppk/Johnson |
| 测量系统可否接受 | `gage_rr` | 交叉/嵌套/属性/破坏性 |
| 产品能用多久 | `reliability` | Weibull/KM/ALT/Crow-AMSAA |
| 实验怎么设计 | `doe` | 全因子/分数因子/RSM/Taguchi/DSD |
| 多变量降维 | `multivariate` | PCA/聚类/判别/因子/MANOVA |
| 贝叶斯分析 | `bayesian` | 估计/t检验/比例/ANOVA |
| 分布拟合 | `distribution` | MLE/MOM/AIC/BIC 选择 |
| 异常值检测 | `outlier` | Grubbs/Dixon/IQR/Z-score/MAD |
| 数据清洗变换 | `clean` / `transform` | 插补/裁剪/Box-Cox/Johnson |
| 批量抽检方案 | `acceptance_sampling` | OC 曲线/AQL/LTPD |
| 蒙特卡洛模拟 | `sensitivity` | Monte Carlo/Sobol/龙卷风图 |
| 函数型数据 | `functional` | 基函数/平滑/导数/FPCA/FANOVA |

# stats-cli-py

Pure Python statistical analysis tool for manufacturing and quality engineering, powered by scipy + statsmodels. All I/O is JSON — designed to be called as an AI-agent skill.

**For AI Agents:** Not sure which command to use? Call `discover` first — it returns all available commands with their parameters and examples. Use `discover {"command_name": "xxx"}` to get required/optional parameter details before constructing your request.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start-dont-know-what-analysis-to-use)
- [智能引导流程](#智能引导流程用户模糊请求时)
- [Decision Trees](#decision-tree-1-比较分析两组或多组数据比较)
- [Scenario-Based Workflows](#scenario-based-workflows)
- [All Commands](#all-commands-39-commands)
- [Output Interpretation Guide](#output-interpretation-guide)
- [Performance & Scale Guidance](#performance--scale-guidance)
- [Output Format](#output-format)
- [File Support](#file-support)
- [Dependencies](#dependencies)

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

### Common Python Patterns

```python
# Pattern 1: Batch analysis multiple files
import glob
from main import handler

for file in glob.glob("data/*.xlsx"):
    result = handler({"command": "descriptive", "params": {"file": file, "column": "weight"}})
    if result["status"] == "success":
        print(f"{file}: mean={result['data']['mean']:.2f}")

# Pattern 2: Export report to Excel
report = handler({"command": "report", "params": {"values": data, "usl": 11.0, "lsl": 9.0}})
handler({"command": "export_excel", "params": {"report_data": report["data"], "output_path": "report.xlsx"}})

# Pattern 3: Decision workflow with assumption checking
check = handler({"command": "check_assumptions", "params": {"values": data, "test_type": "ttest"}})
if check["data"]["overall_assumptions_met"]:
    result = handler({"command": "ttest", "params": {"values": group1, "values2": group2}})
else:
    result = handler({"command": "nonparametric", "params": {"test_type": "mann_whitney", "x": group1, "y": group2}})

# Pattern 4: CLI wrapper with error handling
import sys, json
input_data = json.loads(sys.stdin.read())
result = handler(input_data)
if result["status"] == "error":
    print(f"Error: {result['message']}", file=sys.stderr)
    sys.exit(1)
print(json.dumps(result, indent=2))
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

## All Commands (39 commands)

### Data Exploration
```python
{"command": "explore", "params": {"file": "data.xlsx"}}
```

### Using `discover` for Command Exploration

```python
# List all commands
{"command": "discover", "params": {}}

# Filter by category
{"command": "discover", "params": {"category": "spc"}}       # SPC 命令
{"command": "discover", "params": {"category": "hypothesis"}} # 假设检验
{"command": "discover", "params": {"category": "regression"}} # 回归分析

# Get detailed parameter schema for a specific command
{"command": "discover", "params": {"command_name": "regression"}}
# Returns: params with required/type/desc, output_fields, example

# Example: before calling regression, check what reg_type values are valid
{"command": "discover", "params": {"command_name": "regression"}}
# → reg_type desc includes: "linear, quadratic, polynomial, multiple, stepwise,
#   logistic, exponential, power, logarithmic, sigmoid, lasso, ridge, elastic_net,
#   robust, pls, poisson, gamma, negbin, cross_validate"
```

**When to use discover:**
- Before first call to an unfamiliar command
- When a command returns PARAM_ERROR — check required params
- When exploring what analysis options are available for your data

**Key Output Fields (explore):**
- `sheets`: List of all sheet names in Excel file — use this to let user choose which sheet
- `hint`: Present when multiple sheets exist — show this to user
- `numeric_columns`: Auto-detected numeric columns — use for analysis
- `detected_specs`: Auto-detected USL/LSL/Target from column names

**Output Example (explore):**
```json
{
  "status": "success",
  "data": {
    "file": "data.xlsx",
    "n_rows": 100,
    "n_columns": 5,
    "numeric_columns": ["weight", "hardness"],
    "text_columns": ["batch"],
    "columns": [
      {"name": "weight", "dtype": "float64", "non_null": 100, "null": 0, "mean": 10.25, "min": 9.8, "max": 10.7}
    ],
    "sample_data": [{"weight": 10.2, "hardness": 45.3, "batch": "A"}],
    "sheet": "Sheet1",
    "n_sheets": 2,
    "sheets": ["Sheet1", "Summary"],
    "hint": "This workbook has 2 sheets. Use 'sheet' parameter to explore other sheets: ['Sheet1', 'Summary']"
  }
}
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
{"command": "regression", "params": {"x_columns": ["x1", "x2"], "y_column": "y", "reg_type": "pls", "file": "data.csv"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5], "reg_type": "poisson"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5], "reg_type": "gamma"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5], "reg_type": "negbin"}}
{"command": "regression", "params": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "reg_type": "linear", "cv": 5}}
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
{"command": "doe", "params": {"doe_type": "response_surface", "factors": [{"name": "Temp", "low": 100, "high": 200}, {"name": "Time", "low": 30, "high": 60}]}}
{"command": "doe", "params": {"doe_type": "definitive_screening", "factors": [{"name": "A", "low": -1, "high": 1}, {"name": "B", "low": -1, "high": 1}, {"name": "C", "low": -1, "high": 1}]}}
```

### MSA / Measurement System Analysis
```python
{"command": "gage_rr", "params": {"analysis_type": "crossed", "measurements": [10, 11, 10, 11, 12, 11, 10, 11, 10], "parts": ["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3"], "operators": ["A", "B", "A", "A", "B", "A", "A", "B", "A"], "tolerance": 10.0}}
{"command": "gage_rr", "params": {"analysis_type": "nested", "measurements": [10, 11, 10, 11, 12, 11, 10, 11, 10], "parts": ["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3"], "operators": ["A", "B", "A", "A", "B", "A", "A", "B", "A"]}}
{"command": "gage_rr", "params": {"analysis_type": "attribute", "measurements": [1, 0, 1, 0, 1, 0, 1, 1, 0], "parts": ["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3"], "operators": ["A", "B", "A", "A", "B", "A", "A", "B", "A"]}}
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
{"command": "advanced", "params": {"analysis_type": "bootstrap", "values": [10.1, 10.2, 10.0, 10.3, 10.1], "statistic": "mean", "n_bootstrap": 10000}}
```

### Distribution Analysis
```python
# Fit a distribution to data
{"command": "distribution", "params": {"analysis_type": "fit", "values": [1.2, 1.5, 1.3, 1.8, 2.1, 1.9], "dist_name": "normal"}}
# Goodness-of-fit test
{"command": "distribution", "params": {"analysis_type": "gof", "values": [1.2, 1.5, 1.3, 1.8, 2.1, 1.9], "dist_name": "normal"}}
# Compare multiple distributions, rank by AIC/BIC
{"command": "distribution", "params": {"analysis_type": "select", "values": [1.2, 1.5, 1.3, 1.8, 2.1, 1.9]}}
# Supported distributions: normal, lognormal, exponential, gamma, weibull, beta, logistic, gumbel
```

### Bayesian Statistics
```python
# Bayesian estimation with conjugate normal prior
{"command": "bayesian", "params": {"analysis_type": "estimate", "values": [10.2, 10.5, 10.3, 10.1, 10.4]}}
# Bayesian t-test with Bayes Factor
{"command": "bayesian", "params": {"analysis_type": "ttest", "values": [10.2, 10.5, 10.3], "values2": [11.3, 11.5, 11.1]}}
# Bayesian proportion test (Beta-Binomial)
{"command": "bayesian", "params": {"analysis_type": "proportion", "successes": 7, "n": 10}}
# Bayesian one-way ANOVA
{"command": "bayesian", "params": {"analysis_type": "anova", "groups": [[10, 11, 12], [20, 21, 22], [30, 31, 32]]}}
```

### Data Mining
```python
# Classification: decision_tree or random_forest
{"command": "mining", "params": {"analysis_type": "classify", "features": [[1,2],[3,4],[5,6],[7,8]], "labels": [0,0,1,1], "method": "random_forest"}}
# Anomaly detection: isolation_forest, lof, zscore, ensemble
{"command": "mining", "params": {"analysis_type": "anomaly", "values": [10, 11, 10, 12, 100, 11, 10, 500]}}
# Association rules (Apriori)
{"command": "mining", "params": {"analysis_type": "associate", "transactions": [["bread","milk"],["bread","diapers","beer"],["milk","diapers","beer"]], "min_support": 0.3, "min_confidence": 0.5}}
```

### Sensitivity Analysis
```python
# Monte Carlo simulation
{"command": "sensitivity", "params": {"analysis_type": "monte_carlo", "inputs": {"x1": {"dist": "normal", "params": {"mean": 10, "std": 1}}, "x2": {"dist": "uniform", "params": {"low": 5, "high": 15}}}, "formula": "x1 * x2", "n_simulations": 10000}}
# Tornado diagram (one-at-a-time sensitivity)
{"command": "sensitivity", "params": {"analysis_type": "tornado", "inputs": {"x1": {"dist": "normal", "params": {"mean": 10, "std": 1}}, "x2": {"dist": "normal", "params": {"mean": 5, "std": 0.5}}}, "formula": "x1 * x2"}}
# Sobol sensitivity indices
{"command": "sensitivity", "params": {"analysis_type": "sobol", "inputs": {"x1": {"dist": "uniform", "params": {"low": 0, "high": 1}}, "x2": {"dist": "uniform", "params": {"low": 0, "high": 1}}}, "formula": "x1 + x2", "n_simulations": 5000}}
```

### Acceptance Sampling
```python
# Single sampling plan
{"command": "acceptance_sampling", "params": {"analysis_type": "single_plan", "n": 50, "c": 2, "defect_rate": 0.05}}
# Double sampling plan
{"command": "acceptance_sampling", "params": {"analysis_type": "double_plan", "n1": 30, "c1": 1, "d1": 4, "n2": 30, "c2": 3, "defect_rate": 0.05}}
# Generate OC curve
{"command": "acceptance_sampling", "params": {"analysis_type": "oc_curve", "n": 50, "c": 2}}
# Find optimal plan given AQL/LTPD
{"command": "acceptance_sampling", "params": {"analysis_type": "find_plan", "AQL": 0.01, "LTPD": 0.05, "alpha": 0.05, "beta": 0.10}}
```

### Functional Data Analysis (FDA)
```python
# Basis function representation (B-spline, Fourier, polynomial)
{"command": "functional", "params": {"analysis_type": "basis", "t": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "values": [0, 0.3, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0], "basis_type": "bspline", "n_basis": 5}}
# Data smoothing (spline, kernel, lowess)
{"command": "functional", "params": {"analysis_type": "smooth", "t": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "values": [0.1, 0.35, 0.55, 0.85, 0.95, 1.05, 0.85, 0.55, 0.25, 0.15, -0.05], "method": "spline"}}
# Derivative estimation
{"command": "functional", "params": {"analysis_type": "derivative", "t": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "values": [0, 0.3, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0], "order": 1}}
# Functional PCA
{"command": "functional", "params": {"analysis_type": "fpca", "curves": [[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]], "t": [0, 0.25, 0.5, 0.75, 1.0], "n_components": 2}}
# Functional regression (scalar-on-function)
{"command": "functional", "params": {"analysis_type": "regression", "curves": [[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]], "t": [0, 0.25, 0.5, 0.75, 1.0], "y": [10, 20, 30], "mode": "scalar_on_function"}}
# Functional ANOVA
{"command": "functional", "params": {"analysis_type": "fanova", "groups": [[[1,2,3,4,5],[2,3,4,5,6]], [[10,11,12,13,14],[11,12,13,14,15]]], "t": [0, 0.25, 0.5, 0.75, 1.0]}}
# Functional clustering
{"command": "functional", "params": {"analysis_type": "cluster", "curves": [[1,2,3,4,5],[2,3,4,5,6],[10,11,12,13,14],[11,12,13,14,15]], "t": [0, 0.25, 0.5, 0.75, 1.0], "n_clusters": 2}}
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
- **Timeout:** Default 5 seconds. Add `"timeout": 10` to params for longer scripts. Infinite loops are automatically terminated.

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
```

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

**Output Example (recommend):**
```json
{
  "status": "success",
  "data": {
    "goal": "compare means of two groups",
    "recommendations": [
      {"test": "two_sample_ttest", "reason": "Two independent groups, use two-sample t-test"},
      {"test": "mann_whitney", "reason": "Non-parametric alternative to two-sample t-test"}
    ],
    "top_recommendation": {"test": "two_sample_ttest", "reason": "Two independent groups, use two-sample t-test"}
  }
}
```

**Output Example (check_assumptions):**
```json
{
  "status": "success",
  "data": {
    "test_type": "ttest",
    "assumptions": {
      "normality": {"test": "shapiro_wilk", "passed": true, "p_value": 0.85, "interpretation": "Data appears normal"},
      "sample_size": {"n": 10, "minimum": 2, "recommended": 30, "passed": true, "adequate": false}
    },
    "recommendations": {"primary": "ttest", "variant": "one_sample", "reason": "Assumptions met"},
    "overall_assumptions_met": true
  }
}
```

**Output Example (workflow_template):**
```json
{
  "status": "success",
  "data": {
    "template": "manufacturing",
    "steps_results": [
      {"command": "clean", "status": "success", "data": {"method": "drop", "n_removed": 0}},
      {"command": "descriptive", "status": "success", "data": {"mean": 10.3, "std": 0.14}},
      {"command": "normality", "status": "success", "data": {"is_normal": true}},
      {"command": "capability", "status": "success", "data": {"cp": 2.5, "cpk": 2.3}},
      {"command": "report", "status": "success", "data": {"summary": "..."}}
    ],
    "overall_status": "success"
  }
}
```

### Report Export (报告导出)

```python
# Generate report first
report_data = handler({"command": "report", "params": {"values": [10.1,10.2,10.0,10.3,10.1,10.2,10.0,10.1,10.3,10.2], "usl": 11.0, "lsl": 9.0}})

# Export to Excel
handler({"command": "export_excel", "params": {"report_data": report_data["data"], "output_path": "report.xlsx"}})

# Export to PDF
handler({"command": "export_pdf", "params": {"report_data": report_data["data"], "output_path": "report.pdf"}})
```

**Output Example (export_excel / export_pdf):**
```json
{
  "status": "success",
  "data": {
    "output_path": "report.xlsx",
    "sheets": ["Summary", "Descriptive", "Normality", "Capability"],
    "message": "Report exported to report.xlsx"
  }
}
```

---

## Output Format

All commands return a standard JSON envelope:

### Success
```json
{
  "status": "success",
  "version": "1.4.0",
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
| matplotlib | >= 3.5 | Chart generation |
| fpdf2 | >= 2.8 | PDF report export |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Output Interpretation Guide

### Success Response Format
Every command returns a JSON envelope:
```json
{
  "status": "success",
  "data": { /* command-specific results */ },
  "version": "1.4.0"
}
```

### Key Output Fields by Category

**Descriptive Statistics:**
```json
{
  "mean": 10.25,        // 中心趋势
  "std": 0.5,           // 离散程度（越小越稳定）
  "rsd_percent": 4.88,   // 相对标准偏差（<5% 优秀，5-10% 可接受，>10% 需调查）
  "ci_95_lower": 10.0,   // 95% 置信区间下限
  "ci_95_upper": 10.5    // 95% 置信区间上限
}
```

**Hypothesis Tests (t-test, ANOVA, etc.):**
```json
{
  "p_value": 0.003,      // p < 0.05 → 显著，p >= 0.05 → 不显著
  "significant": true,    // 直接告诉你是否显著
  "cohens_d": 0.8,       // 效应量（0.2 小，0.5 中，0.8 大）
  "interpretation": "..." // 一句话结论
}
```

**Process Capability:**
```json
{
  "cp": 1.33,            // Cp >= 1.33 → 能力充足
  "cpk": 1.21,           // Cpk >= 1.33 → 过程居中且有能力
  "rating": "B",         // A(优秀) B(良好) C(勉强) D(不足)
  "ppm": 6210            // 预期不良品率（越低越好）
}
```

**Control Charts:**
```json
{
  "out_of_control_points": [15, 23],  // 失控点索引（空数组 = 过程稳定）
  "summary": {"stable": false, "message": "Process has 2 out-of-control point(s)"}
}
```

**Bayesian Analysis:**
```json
{
  "bayes_factor_10": 150.0,   // BF10 > 10 → 强证据支持 H1
  "credible_interval": [0.2, 0.8],  // 95% 可信区间
  "bf_interpretation": "strong evidence for H1"
}
```

**Regression:**
```json
{
  "r_squared": 0.95,     // R² 越接近 1 越好
  "coefficients": {...},  // 回归系数
  "interpretation": "..." // 一句话结论
}
```

---

## Performance & Scale Guidance

| Command | Recommended Max N | Notes |
|---|---|---|
| descriptive | 1,000,000 | Linear O(n), very fast |
| regression | 100,000 | O(n) for simple, O(n²) for polynomial |
| anova | 10,000 per group | Fast, but very large groups add little value |
| monte_carlo | 1,000,000 sims | Vectorized, ~150ms for 1M |
| sobol | 50,000 sims | ~5s for 50k with 3 variables |
| mining/classify | 100,000 samples | sklearn handles well |
| functional/fpca | 1,000 curves × 100 points | Matrix operations, O(n·p²) |
| bayesian | 10,000 | MLE optimization, fast |
| bootstrap | 100,000 resamples | Vectorized, fast |

**Tips:**
- For very large datasets (>100k), consider sampling before analysis
- Monte Carlo and Sobol are vectorized — use high simulation counts freely
- PCA/FPCA scale with min(n_samples, n_features)²

---

## Error Handling

When a command returns `status: "error"`, check `error_type` for the cause:

| error_type | Meaning | Fix |
|---|---|---|
| `PARAM_ERROR` | Missing or wrong parameter | Use `discover` to check required params |
| `DATA_ERROR` | Insufficient or bad data | Check data size, format, NaN values |
| `COMPUTATION_ERROR` | Math error (zero variance, singular matrix) | Check data variability |
| `FILE_NOT_FOUND` | File path wrong | Verify file path |
| `MISSING_DEPENDENCY` | Package not installed | `pip install -r requirements.txt` |
| `MEMORY_ERROR` | Too much data | Reduce data size |
| `INTERNAL_ERROR` | Bug | Report issue |
| `INVALID_INPUT` | Input format wrong (e.g. non-numeric values) | Check input data types |
| `MISSING_COMMAND` | Unknown command name | Use `discover` to list available commands |

Example error response:
```json
{
  "status": "error",
  "error_type": "PARAM_ERROR",
  "message": "Unknown command: xxx. Use 'discover' to list available commands."
}
```

---

## Documentation

Additional documentation available in `docs/`:
- `user-guide-cn.pdf` — 用户指南（中文）
- `user-guide-cn-v2.pdf` — 用户指南 v2（中文）
- `presentation-cn.pdf` — 产品介绍（中文）
- `skill-architecture.pdf` — 技能架构设计
- `skill-architecture-cn.pdf` — 技能架构设计（中文）

---

## Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v
```

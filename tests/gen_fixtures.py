"""Generate 100 synthetic Excel files for integration testing.

Run: python tests/gen_fixtures.py
Output: tests/fixtures/excel/ (100 .xlsx files)
"""

import os

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "excel")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED)
counter = 0


def next_name(prefix):
    global counter
    counter += 1
    return f"{counter:03d}_{prefix}.xlsx"


def save(df_or_dict, name, sheet_name="Sheet1"):
    """Save DataFrame or dict of DataFrames to Excel."""
    path = os.path.join(OUT_DIR, name)
    if isinstance(df_or_dict, dict):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sname, df in df_or_dict.items():
                df.to_excel(writer, sheet_name=sname, index=False)
    else:
        df_or_dict.to_excel(path, sheet_name=sheet_name, index=False)
    return path


# ============================================================================
# Group 1: Data Scale (1-20)
# ============================================================================

# 1-4: Single column, different sizes
for n in [5, 50, 500, 5000]:
    df = pd.DataFrame({"value": np.random.normal(100, 15, n)})
    save(df, next_name(f"singlecol_{n}rows"))

# 5-8: Three columns, different sizes
for n in [10, 100, 1000, 5000]:
    df = pd.DataFrame(
        {
            "x1": np.random.normal(50, 10, n),
            "x2": np.random.normal(60, 12, n),
            "x3": np.random.normal(55, 8, n),
        }
    )
    save(df, next_name(f"threecol_{n}rows"))

# 9-12: Many columns
for ncols in [5, 10, 20, 50]:
    data = {f"col{i}": np.random.normal(0, 1, 100) for i in range(ncols)}
    df = pd.DataFrame(data)
    save(df, next_name(f"{ncols}cols_100rows"))

# 13-16: Large dataset variants
for n in [100, 500, 2000, 10000]:
    df = pd.DataFrame(
        {
            "measurement": np.random.normal(10, 2, n),
            "batch": np.random.choice(["A", "B", "C"], n),
        }
    )
    save(df, next_name(f"large_{n}rows"))

# 17-20: Wide datasets
for ncols in [3, 10, 30, 100]:
    data = {f"feature_{i}": np.random.normal(0, 1, 50) for i in range(ncols)}
    df = pd.DataFrame(data)
    save(df, next_name(f"wide_{ncols}cols"))


# ============================================================================
# Group 2: Data Distributions (21-35)
# ============================================================================

# 21: Normal
df = pd.DataFrame({"value": np.random.normal(100, 15, 200)})
save(df, next_name("dist_normal"))

# 22: Uniform
df = pd.DataFrame({"value": np.random.uniform(0, 100, 200)})
save(df, next_name("dist_uniform"))

# 23: Exponential
df = pd.DataFrame({"value": np.random.exponential(2, 200)})
save(df, next_name("dist_exponential"))

# 24: Log-normal (skewed)
df = pd.DataFrame({"value": np.random.lognormal(3, 1, 200)})
save(df, next_name("dist_lognormal"))

# 25: Bimodal
v1 = np.random.normal(30, 5, 100)
v2 = np.random.normal(70, 5, 100)
df = pd.DataFrame({"value": np.concatenate([v1, v2])})
save(df, next_name("dist_bimodal"))

# 26: Chi-square (right-skewed)
df = pd.DataFrame({"value": np.random.chisquare(2, 200)})
save(df, next_name("dist_chisquare"))

# 27: Binomial
df = pd.DataFrame({"value": np.random.binomial(100, 0.5, 200)})
save(df, next_name("dist_binomial"))

# 28: Poisson
df = pd.DataFrame({"value": np.random.poisson(5, 200)})
save(df, next_name("dist_poisson"))

# 29: Beta
df = pd.DataFrame({"value": np.random.beta(2, 5, 200)})
save(df, next_name("dist_beta"))

# 30: Gamma
df = pd.DataFrame({"value": np.random.gamma(5, 2, 200)})
save(df, next_name("dist_gamma"))

# 31: Constant values
df = pd.DataFrame({"value": [42.0] * 100})
save(df, next_name("dist_constant"))

# 32: Near-constant (tiny variance)
df = pd.DataFrame({"value": 100 + np.random.normal(0, 0.0001, 100)})
save(df, next_name("dist_near_constant"))

# 33: Integers only
df = pd.DataFrame({"value": np.random.randint(1, 100, 200)})
save(df, next_name("dist_integers"))

# 34: Small decimals
df = pd.DataFrame({"value": np.random.uniform(0.001, 0.01, 200)})
save(df, next_name("dist_small_decimals"))

# 35: Mixed scales
df = pd.DataFrame(
    {
        "small": np.random.uniform(0.001, 0.01, 100),
        "medium": np.random.uniform(10, 100, 100),
        "large": np.random.uniform(10000, 100000, 100),
    }
)
save(df, next_name("dist_mixed_scales"))


# ============================================================================
# Group 3: Missing Data (36-45)
# ============================================================================

# 36: No missing data
df = pd.DataFrame({"value": np.random.normal(50, 10, 100)})
save(df, next_name("missing_none"))

# 37: Random 5% NaN
vals = np.random.normal(50, 10, 200)
mask = np.random.random(200) < 0.05
vals[mask] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_random_5pct"))

# 38: Random 20% NaN
vals = np.random.normal(50, 10, 200)
mask = np.random.random(200) < 0.20
vals[mask] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_random_20pct"))

# 39: Random 50% NaN
vals = np.random.normal(50, 10, 200)
mask = np.random.random(200) < 0.50
vals[mask] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_random_50pct"))

# 40: First 10 rows NaN
vals = np.random.normal(50, 10, 100)
vals[:10] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_first10"))

# 41: Last 10 rows NaN
vals = np.random.normal(50, 10, 100)
vals[-10:] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_last10"))

# 42: Entire column NaN
df = pd.DataFrame(
    {
        "good": np.random.normal(50, 10, 100),
        "all_nan": np.nan,
    }
)
save(df, next_name("missing_entire_col"))

# 43: Alternating NaN
vals = np.random.normal(50, 10, 100)
vals[::2] = np.nan
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_alternating"))

# 44: Multi-column with different missing patterns
df = pd.DataFrame(
    {
        "a": np.random.normal(0, 1, 100),
        "b": np.append(np.nan, np.random.normal(0, 1, 99)),
        "c": np.append(np.random.normal(0, 1, 99), np.nan),
    }
)
save(df, next_name("missing_multi_pattern"))

# 45: Only one valid value in a column
vals = np.full(100, np.nan)
vals[50] = 42.0
df = pd.DataFrame({"value": vals})
save(df, next_name("missing_single_valid"))


# ============================================================================
# Group 4: Mixed Types (46-55)
# ============================================================================

# 46: Numeric + text column
df = pd.DataFrame(
    {
        "numeric": np.random.normal(50, 10, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
    }
)
save(df, next_name("mixed_num_text"))

# 47: Numeric + boolean
df = pd.DataFrame(
    {
        "value": np.random.normal(50, 10, 100),
        "flag": np.random.choice([True, False], 100),
    }
)
save(df, next_name("mixed_num_bool"))

# 48: All text (should fail gracefully)
df = pd.DataFrame(
    {
        "name": [f"item_{i}" for i in range(50)],
        "label": np.random.choice(["x", "y", "z"], 50),
    }
)
save(df, next_name("mixed_all_text"))

# 49: Numeric stored as string
df = pd.DataFrame(
    {
        "value": [str(round(x, 2)) for x in np.random.normal(50, 10, 100)],
    }
)
save(df, next_name("mixed_num_as_string"))

# 50: Mixed numeric and string in same column
vals = list(np.random.normal(50, 10, 80))
vals += [
    "N/A",
    "missing",
    "-",
    "",
    "null",
    "NA",
    "n/a",
    "#REF!",
    "inf",
    "-inf",
    "1.5e10",
    "3.14",
    "0",
    "-999",
    "100.00",
    "abc",
    "12.5",
    "6.02e23",
    "NaN",
    "Inf",
]
np.random.shuffle(vals)
df = pd.DataFrame({"value": vals})
save(df, next_name("mixed_messy_column"))

# 51: Date column + numeric
dates = pd.date_range("2024-01-01", periods=100, freq="D")
df = pd.DataFrame(
    {
        "date": dates,
        "measurement": np.random.normal(100, 15, 100),
    }
)
save(df, next_name("mixed_date_numeric"))

# 52: Multiple numeric + text
df = pd.DataFrame(
    {
        "x1": np.random.normal(0, 1, 100),
        "x2": np.random.normal(0, 1, 100),
        "label": np.random.choice(["group1", "group2"], 100),
        "y": np.random.normal(0, 1, 100),
    }
)
save(df, next_name("mixed_multi_num_text"))

# 53: Column with Inf values
vals = np.random.normal(50, 10, 100)
vals[0] = float("inf")
vals[1] = float("-inf")
df = pd.DataFrame({"value": vals})
save(df, next_name("mixed_with_inf"))

# 54: Column with very long strings
df = pd.DataFrame(
    {
        "value": np.random.normal(50, 10, 50),
        "description": ["A" * 1000] * 50,
    }
)
save(df, next_name("mixed_long_strings"))

# 55: Unicode column names
df = pd.DataFrame(
    {
        "测量值": np.random.normal(50, 10, 100),
        "温度": np.random.normal(25, 5, 100),
        "压力": np.random.normal(101.3, 2, 100),
    }
)
save(df, next_name("mixed_unicode_cols"))


# ============================================================================
# Group 5: Edge Cases (56-65)
# ============================================================================

# 56: Single row
df = pd.DataFrame({"value": [42.0]})
save(df, next_name("edge_single_row"))

# 57: Two rows
df = pd.DataFrame({"value": [1.0, 2.0]})
save(df, next_name("edge_two_rows"))

# 58: Very large values
df = pd.DataFrame({"value": [1e15, 1e16, 1e17, 1e18, 1e19]})
save(df, next_name("edge_very_large"))

# 59: Very small values
df = pd.DataFrame({"value": [1e-15, 1e-16, 1e-17, 1e-18, 1e-19]})
save(df, next_name("edge_very_small"))

# 60: All zeros
df = pd.DataFrame({"value": [0.0] * 50})
save(df, next_name("edge_all_zeros"))

# 61: Negative values
df = pd.DataFrame({"value": np.random.uniform(-100, -1, 100)})
save(df, next_name("edge_all_negative"))

# 62: Mixed positive/negative
df = pd.DataFrame({"value": np.random.uniform(-50, 50, 100)})
save(df, next_name("edge_pos_neg_mix"))

# 63: Extreme range (min to max float-ish)
df = pd.DataFrame({"value": [-1e12, -1, 0, 1, 1e12]})
save(df, next_name("edge_extreme_range"))

# 64: Scientific notation values
df = pd.DataFrame({"value": [1.5e10, 3.14e-5, 6.02e23, 1.6e-19, 9.81]})
save(df, next_name("edge_scientific"))

# 65: Repeated values (low cardinality)
df = pd.DataFrame({"value": np.random.choice([10, 20, 30], 100)})
save(df, next_name("edge_low_cardinality"))


# ============================================================================
# Group 6: Sheet Structure (66-73)
# ============================================================================

# 66: Single sheet (default name)
df = pd.DataFrame({"value": np.random.normal(50, 10, 50)})
save(df, next_name("sheet_single"))

# 67: Two sheets
save(
    {
        "Data": pd.DataFrame({"value": np.random.normal(50, 10, 50)}),
        "Summary": pd.DataFrame({"stat": ["mean", "std"], "val": [50, 10]}),
    },
    next_name("sheet_two"),
)

# 68: Three sheets with different data
save(
    {
        "Batch1": pd.DataFrame({"value": np.random.normal(10, 1, 30)}),
        "Batch2": pd.DataFrame({"value": np.random.normal(20, 2, 30)}),
        "Batch3": pd.DataFrame({"value": np.random.normal(30, 3, 30)}),
    },
    next_name("sheet_three_batches"),
)

# 69: Sheet with Chinese name
df = pd.DataFrame({"测量值": np.random.normal(50, 10, 50)})
save({"测量数据": df}, next_name("sheet_chinese_name"))

# 70: Sheet with spaces in name
df = pd.DataFrame({"value": np.random.normal(50, 10, 50)})
save({"My Data": df}, next_name("sheet_spaces_name"))

# 71: Sheet with special characters
df = pd.DataFrame({"value": np.random.normal(50, 10, 50)})
save({"Data (v2.0)": df}, next_name("sheet_special_chars"))

# 72: Many sheets (10)
sheets = {}
for i in range(10):
    sheets[f"Sheet{i}"] = pd.DataFrame({"value": np.random.normal(i * 10, 1, 20)})
save(sheets, next_name("sheet_many_10"))

# 73: Empty sheet + data sheet
save(
    {
        "Empty": pd.DataFrame(),
        "Data": pd.DataFrame({"value": np.random.normal(50, 10, 50)}),
    },
    next_name("sheet_empty_plus_data"),
)


# ============================================================================
# Group 7: Header Variants (74-80)
# ============================================================================

# 74: Standard header
df = pd.DataFrame({"temperature": np.random.normal(25, 5, 50), "pressure": np.random.normal(101.3, 2, 50)})
save(df, next_name("header_standard"))

# 75: Numeric header (column names are numbers)
df = pd.DataFrame({1: np.random.normal(0, 1, 50), 2: np.random.normal(0, 1, 50), 3: np.random.normal(0, 1, 50)})
save(df, next_name("header_numeric"))

# 76: Single letter header
df = pd.DataFrame({"A": np.random.normal(0, 1, 50), "B": np.random.normal(0, 1, 50)})
save(df, next_name("header_single_letter"))

# 77: Very long column names
df = pd.DataFrame(
    {
        "this_is_a_very_long_column_name_that_goes_on_and_on_and_on": np.random.normal(0, 1, 50),
        "another_extremely_long_column_name_for_testing_purposes": np.random.normal(0, 1, 50),
    }
)
save(df, next_name("header_long_names"))

# 78: Duplicate column names (pandas allows this)
df = pd.DataFrame(np.random.normal(0, 1, (50, 3)))
df.columns = ["value", "value", "value"]
save(df, next_name("header_duplicate_names"))

# 79: Empty string header
df = pd.DataFrame({"": np.random.normal(0, 1, 50), " ": np.random.normal(0, 1, 50)})
save(df, next_name("header_empty_names"))

# 80: Header with special characters
df = pd.DataFrame(
    {
        "col-1": np.random.normal(0, 1, 50),
        "col.2": np.random.normal(0, 1, 50),
        "col (3)": np.random.normal(0, 1, 50),
    }
)
save(df, next_name("header_special_chars"))


# ============================================================================
# Group 8: Statistical Scenarios (81-100)
# ============================================================================

# 81: Strong linear relationship
x = np.linspace(0, 10, 100)
y = 2 * x + 3 + np.random.normal(0, 0.5, 100)
df = pd.DataFrame({"x": x, "y": y})
save(df, next_name("stat_linear"))

# 82: Quadratic relationship
x = np.linspace(-5, 5, 100)
y = x**2 + np.random.normal(0, 1, 100)
df = pd.DataFrame({"x": x, "y": y})
save(df, next_name("stat_quadratic"))

# 83: No relationship (random)
df = pd.DataFrame(
    {
        "x": np.random.normal(0, 1, 100),
        "y": np.random.normal(0, 1, 100),
    }
)
save(df, next_name("stat_no_relationship"))

# 84: Three distinct groups
df = pd.DataFrame(
    {
        "value": np.concatenate(
            [
                np.random.normal(10, 1, 30),
                np.random.normal(20, 1, 30),
                np.random.normal(30, 1, 30),
            ]
        ),
        "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
    }
)
save(df, next_name("stat_three_groups"))

# 85: Time series (trend + seasonality)
t = np.arange(200)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, 200)
df = pd.DataFrame({"value": trend + seasonal + noise})
save(df, next_name("stat_timeseries"))

# 86: Process data with shift
vals = np.random.normal(100, 5, 200)
vals[100:] += 10  # process shift at midpoint
df = pd.DataFrame({"value": vals})
save(df, next_name("stat_process_shift"))

# 87: Capability study data (known Cp ~ 1.33)
vals = np.random.normal(10, 0.5, 200)
df = pd.DataFrame({"value": vals})
save(df, next_name("stat_capability"))

# 88: Outlier-heavy data
vals = np.random.normal(50, 5, 100)
vals[0] = 200
vals[1] = -100
vals[50] = 300
df = pd.DataFrame({"value": vals})
save(df, next_name("stat_outliers"))

# 89: Correlation matrix data (3 correlated variables)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n)
x3 = 0.5 * x1 + 0.5 * x2 + 0.3 * np.random.normal(0, 1, n)
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
save(df, next_name("stat_correlated"))

# 90: PCA data (2 components)
n = 100
comp1 = np.random.normal(0, 3, n)
comp2 = np.random.normal(0, 1, n)
x1 = comp1 + 0.1 * comp2
x2 = comp1 - 0.1 * comp2
x3 = comp2 + 0.1 * comp1
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
save(df, next_name("stat_pca"))

# 91: Survival/reliability data
times = np.random.exponential(100, 100)
status = np.random.choice([0, 1], 100, p=[0.3, 0.7])
df = pd.DataFrame({"time": times, "status": status})
save(df, next_name("stat_survival"))

# 92: Paired data (before/after)
before = np.random.normal(50, 10, 50)
after = before + np.random.normal(5, 3, 50)
df = pd.DataFrame({"before": before, "after": after})
save(df, next_name("stat_paired"))

# 93: Count data
df = pd.DataFrame({"count": np.random.poisson(10, 200)})
save(df, next_name("stat_counts"))

# 94: Proportion data
df = pd.DataFrame({"proportion": np.random.beta(5, 10, 200)})
save(df, next_name("stat_proportions"))

# 95: Multi-factor DOE data
df = pd.DataFrame(
    {
        "A": np.random.choice(["low", "high"], 100),
        "B": np.random.choice(["low", "high"], 100),
        "C": np.random.choice(["low", "high"], 100),
        "response": np.random.normal(50, 10, 100),
    }
)
save(df, next_name("stat_doe"))

# 96: Gage R&R data
parts = np.repeat(range(1, 11), 6)
operators = np.tile(np.repeat(range(1, 4), 2), 10)
measurements = 100 + np.random.normal(0, 1, 60) + np.repeat(np.random.normal(0, 2, 10), 6)
df = pd.DataFrame(
    {
        "part": parts,
        "operator": operators,
        "measurement": measurements,
    }
)
save(df, next_name("stat_gage_rr"))

# 97: Control chart data (in-control)
df = pd.DataFrame({"value": np.random.normal(100, 2, 300)})
save(df, next_name("stat_control_in"))

# 98: Control chart data (out-of-control)
vals = np.random.normal(100, 2, 300)
vals[200:210] += 6  # shift
df = pd.DataFrame({"value": vals})
save(df, next_name("stat_control_ooc"))

# 99: Regression with multiple predictors
n = 100
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
y = 2 * x1 + 3 * x2 - x3 + np.random.normal(0, 1, n)
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
save(df, next_name("stat_multi_regression"))

# 100: Comprehensive multi-column dataset
n = 200
df = pd.DataFrame(
    {
        "id": range(1, n + 1),
        "measurement": np.random.normal(100, 15, n),
        "temperature": np.random.normal(25, 5, n),
        "pressure": np.random.normal(101.3, 2, n),
        "batch": np.random.choice(["B1", "B2", "B3", "B4"], n),
        "pass_fail": np.random.choice([0, 1], n, p=[0.1, 0.9]),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
    }
)
save(df, next_name("stat_comprehensive"))


print(f"Generated {counter} Excel files in {OUT_DIR}")

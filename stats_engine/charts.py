"""Chart generation module - returns base64 PNG for JSON embedding."""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def histogram(values, title="Distribution", bins=30, normal_curve=True):
    """Generate histogram with optional normal curve overlay."""
    from scipy import stats as sp_stats

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")

    if normal_curve and len(arr) > 2:
        x = np.linspace(arr.min(), arr.max(), 200)
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma > 0:
            ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal fit")
            ax.legend()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def control_chart_plot(values, chart_type="imr", ucl=None, cl=None, lcl=None, out_of_control=None):
    """Generate control chart plot."""
    arr = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(arr)), arr, "b-o", markersize=4, linewidth=1, label="Data")

    if cl is not None:
        ax.axhline(y=cl, color="green", linestyle="-", linewidth=1.5, label=f"CL={cl:.2f}")
    if ucl is not None:
        ax.axhline(y=ucl, color="red", linestyle="--", linewidth=1.5, label=f"UCL={ucl:.2f}")
    if lcl is not None:
        ax.axhline(y=lcl, color="red", linestyle="--", linewidth=1.5, label=f"LCL={lcl:.2f}")

    if out_of_control:
        ooc_x = [i for i in out_of_control if i < len(arr)]
        ooc_y = [arr[i] for i in ooc_x]
        ax.scatter(ooc_x, ooc_y, color="red", s=80, zorder=5, label="Out of Control")

    ax.set_title(f"{chart_type.upper()} Control Chart", fontsize=13)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def boxplot(groups, labels=None, title="Box Plot"):
    """Generate box plot for one or more groups."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=labels)

    colors = ["steelblue", "coral", "green", "purple", "orange"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def scatter_with_regression(x, y, title="Scatter Plot", slope=None, intercept=None, r_squared=None):
    """Generate scatter plot with optional regression line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, color="steelblue", edgecolor="white", s=50)

    if slope is not None and intercept is not None:
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        sign = "+" if intercept >= 0 else "-"
        label = f"y={slope:.2f}x{sign}{abs(intercept):.2f}"
        if r_squared is not None:
            label += f" (R²={r_squared:.4f})"
        ax.plot(x_line, y_line, "r-", linewidth=2, label=label)
        ax.legend()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def qq_plot(values, title="Q-Q Plot"):
    """Generate Q-Q plot for normality check."""
    from scipy import stats as sp_stats

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(6, 6))
    sp_stats.probplot(arr, dist="norm", plot=ax)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def capability_plot(values, usl=None, lsl=None, target=None, cp=None, cpk=None):
    """Generate process capability histogram with spec limits."""
    from scipy import stats as sp_stats

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(arr, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")

    # Normal curve
    mu, sigma = np.mean(arr), np.std(arr, ddof=1)
    if sigma > 0:
        x = np.linspace(arr.min() - sigma, arr.max() + sigma, 200)
        ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), "b-", linewidth=2)

    # Spec limits
    if usl is not None:
        ax.axvline(x=usl, color="red", linestyle="--", linewidth=2, label=f"USL={usl}")
    if lsl is not None:
        ax.axvline(x=lsl, color="red", linestyle="--", linewidth=2, label=f"LSL={lsl}")
    if target is not None:
        ax.axvline(x=target, color="green", linestyle="-", linewidth=2, label=f"Target={target}")

    # Mean
    ax.axvline(x=mu, color="blue", linestyle="-", linewidth=1.5, label=f"Mean={mu:.4f}")

    # Title with capability indices
    title_parts = ["Process Capability"]
    if cp is not None:
        title_parts.append(f"Cp={cp:.2f}")
    if cpk is not None:
        title_parts.append(f"Cpk={cpk:.2f}")
    ax.set_title(" | ".join(title_parts), fontsize=13)

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def time_series_plot(values, title="Time Series", fitted=None, forecast=None):
    """Generate time series plot with optional fitted values and forecast."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(values)), values, "b-o", markersize=3, linewidth=1, label="Observed")

    if fitted is not None and len(fitted) == len(values):
        ax.plot(range(len(fitted)), fitted, "r-", linewidth=1.5, label="Fitted")

    if forecast is not None:
        forecast_x = range(len(values), len(values) + len(forecast))
        ax.plot(forecast_x, forecast, "g--", linewidth=2, marker="s", markersize=4, label="Forecast")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def heatmap(matrix, labels=None, title="Heatmap"):
    """Generate heatmap for correlation matrix or similar."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    # Add values
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im)
    ax.set_title(title, fontsize=13)
    return _fig_to_base64(fig)


def acf_plot(values, max_lag=20, title="Autocorrelation Function"):
    """Generate ACF plot with confidence intervals."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 3:
        return None

    # Calculate ACF
    mean = np.mean(arr)
    var = np.var(arr)
    if var == 0:
        return None

    max_lag = min(max_lag, n - 1)
    acf_values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            acf = np.sum((arr[: n - lag] - mean) * (arr[lag:] - mean)) / (n * var)
            acf_values.append(acf)

    # Confidence interval (95%)
    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(acf_values)), acf_values, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(y=ci, color="red", linestyle="--", linewidth=1, alpha=0.7, label="95% CI")
    ax.axhline(y=-ci, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def pareto_plot(effects, names, title="Pareto Chart of Effects"):
    """Generate Pareto chart for DOE factor effects."""
    # Sort by absolute value descending
    pairs = sorted(zip(names, effects), key=lambda x: abs(x[1]), reverse=True)
    sorted_names = [p[0] for p in pairs]
    sorted_effects = [abs(p[1]) for p in pairs]

    # Cumulative percentage
    total = sum(sorted_effects)
    cumulative = np.cumsum(sorted_effects) / total * 100 if total > 0 else np.zeros(len(sorted_effects))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar chart
    colors = ["steelblue" if e > 0 else "coral" for _, e in pairs]
    ax1.bar(range(len(sorted_names)), sorted_effects, color=colors, alpha=0.7)
    ax1.set_xlabel("Factor")
    ax1.set_ylabel("Absolute Effect")
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha="right")

    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(sorted_names)), cumulative, "r-o", linewidth=2, markersize=6)
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    ax1.set_title(title, fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def outlier_plot(values, outlier_indices=None, title="Outlier Detection"):
    """Generate index plot highlighting detected outliers."""
    arr = np.array(values, dtype=float)
    indices = np.arange(len(arr))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot all points
    ax.scatter(indices, arr, color="steelblue", alpha=0.6, s=50, label="Normal")

    # Highlight outliers
    if outlier_indices:
        outlier_x = [i for i in outlier_indices if i < len(arr)]
        outlier_y = [arr[i] for i in outlier_x]
        ax.scatter(outlier_x, outlier_y, color="red", s=100, zorder=5, label="Outlier", edgecolors="darkred")

    # Add mean and ±2σ lines
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ax.axhline(y=mean, color="green", linestyle="-", linewidth=1.5, label=f"Mean={mean:.2f}")
    ax.axhline(y=mean + 2 * std, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="±2σ")
    ax.axhline(y=mean - 2 * std, color="orange", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def residual_plot(x, y, slope=None, intercept=None, title="Residual Plot"):
    """Generate residual plot for regression diagnostics."""
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)

    if slope is not None and intercept is not None:
        predicted = slope * arr_x + intercept
        residuals = arr_y - predicted
    else:
        residuals = arr_y - np.mean(arr_y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted
    axes[0].scatter(
        predicted if slope is not None else arr_x, residuals, color="steelblue", alpha=0.6, s=50, edgecolor="white"
    )
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs Fitted", fontsize=11)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].grid(alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=20, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Distribution of Residuals", fontsize=11)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _fig_to_base64(fig)


def weibull_plot(times, shape=None, scale=None, title="Weibull Probability Plot"):
    """Generate Weibull probability plot."""
    arr = np.array(times, dtype=float)
    arr = np.sort(arr[arr > 0])  # Weibull requires positive values
    n = len(arr)
    if n < 3:
        return None

    # Median rank (Bernard's approximation)
    ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

    # Transform to Weibull scale: ln(ln(1/(1-F))) vs ln(t)
    try:
        y_weibull = np.log(np.log(1 / (1 - ranks)))
        x_weibull = np.log(arr)
    except (ValueError, RuntimeWarning):
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    ax.scatter(x_weibull, y_weibull, color="steelblue", s=50, alpha=0.7, label="Data")

    # Plot fitted line if parameters provided
    if shape is not None and scale is not None and shape > 0 and scale > 0:
        x_line = np.linspace(x_weibull.min(), x_weibull.max(), 100)
        y_line = shape * (x_line - np.log(scale))
        ax.plot(x_line, y_line, "r-", linewidth=2, label=f"Weibull (β={shape:.2f}, η={scale:.2f})")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("ln(Time)")
    ax.set_ylabel("ln(ln(1/(1-F)))")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def variance_component_plot(components, title="Variance Components"):
    """Generate bar chart for Gage R&R variance components."""
    if not components or not isinstance(components, dict):
        return None

    names = list(components.keys())
    values = list(components.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["steelblue", "coral", "green", "purple", "orange"]
    bars = ax.bar(range(len(names)), values, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Contribution (%)")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def ttest_plot(group1, group2=None, p_value=None, ci=None, title="t-Test"):
    """Generate grouped boxplot for t-test with p-value annotation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = [np.array(g, dtype=float) for g in [group1] if g is not None]
    labels = ["Group 1"]
    if group2 is not None:
        groups.append(np.array(group2, dtype=float))
        labels = ["Group 1", "Group 2"]

    bp = ax.boxplot(groups, patch_artist=True, tick_labels=labels)
    colors = ["steelblue", "coral"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    # Add p-value annotation
    if p_value is not None:
        sig = "*" if p_value < 0.05 else "ns"
        y_max = max(np.max(g) for g in groups)
        y_pos = y_max * 1.1
        ax.annotate(
            f"p={p_value:.4f} {sig}",
            xy=(1.5, y_pos),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="red" if p_value < 0.05 else "gray",
        )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def oc_curve_plot(defect_rates, accept_probs, title="Operating Characteristic Curve"):
    """OC curve for acceptance sampling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(defect_rates, accept_probs, "b-", linewidth=2)
    ax.set_xlabel("Defect Rate")
    ax.set_ylabel("Probability of Acceptance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(defect_rates))
    ax.set_ylim(0, 1.05)
    return _fig_to_base64(fig)


def tornado_plot(variables, sensitivities, title="Tornado Diagram"):
    """Tornado diagram for sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(variables) * 0.5)))
    y_pos = range(len(variables))
    ax.barh(y_pos, sensitivities, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Sensitivity (Swing)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _fig_to_base64(fig)


def missing_values_plot(columns, missing_counts, title="Missing Values"):
    """Generate bar chart showing missing values per column."""
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(columns)), missing_counts, color="steelblue", alpha=0.7)

    # Add value labels
    for bar, count in zip(bars, missing_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_ylabel("Missing Count")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def means_comparison_plot(names, means, ci_lowers, ci_uppers, significant_pairs=None, title="Means Comparison"):
    """Generate means plot with confidence intervals for multiple comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(names))
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(ci_lowers), np.array(ci_uppers) - np.array(means)],
        fmt="o",
        color="steelblue",
        markersize=8,
        capsize=5,
        linewidth=2,
    )

    # Connect significant pairs
    if significant_pairs:
        for i, j in significant_pairs:
            if i < len(means) and j < len(means):
                y_max = max(ci_uppers[i], ci_uppers[j]) * 1.05
                ax.plot([i, j], [y_max, y_max], "r-", linewidth=1.5)
                ax.text((i + j) / 2, y_max * 1.02, "*", ha="center", fontsize=12, color="red")

    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean")
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def tost_plot(observed_diff, ci_lower, ci_upper, delta, title="TOST Equivalence Test"):
    """Generate TOST diagram showing CI vs equivalence bounds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Equivalence bounds
    ax.axvline(x=-delta, color="red", linestyle="--", linewidth=2, label=f"Bounds (±{delta:.2f})")
    ax.axvline(x=delta, color="red", linestyle="--", linewidth=2)
    ax.axvspan(-delta, delta, alpha=0.1, color="green")

    # Observed CI
    ax.plot([ci_lower, ci_upper], [0.5, 0.5], "b-", linewidth=4, label="90% CI")
    ax.plot(observed_diff, 0.5, "bo", markersize=10, label=f"Diff={observed_diff:.4f}")

    # Zero line
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    # Determine result
    equivalent = ci_lower > -delta and ci_upper < delta
    result_text = "EQUIVALENT" if equivalent else "NOT EQUIVALENT"
    result_color = "green" if equivalent else "red"

    ax.set_title(f"{title} - {result_text}", fontsize=13, color=result_color)
    ax.set_xlabel("Difference")
    ax.set_yticks([])
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    return _fig_to_base64(fig)


def bootstrap_plot(original_stat, ci_lower, ci_upper, bootstrap_mean, title="Bootstrap Distribution"):
    """Bootstrap confidence interval plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvline(original_stat, color="blue", linewidth=2, label=f"Original: {original_stat:.4f}")
    ax.axvline(
        bootstrap_mean, color="green", linewidth=1, linestyle="--", label=f"Bootstrap Mean: {bootstrap_mean:.4f}"
    )
    ax.axvspan(ci_lower, ci_upper, alpha=0.3, color="orange", label=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    ax.set_xlabel("Statistic Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def posterior_plot(prior_mean, prior_std, posterior_mean, posterior_std, credible_interval, title="Bayesian Posterior"):
    """Prior vs posterior distribution plot."""
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 5))
    x_min = min(prior_mean - 3 * prior_std, posterior_mean - 3 * posterior_std)
    x_max = max(prior_mean + 3 * prior_std, posterior_mean + 3 * posterior_std)
    x = np.linspace(x_min, x_max, 200)
    ax.plot(x, sp_stats.norm.pdf(x, prior_mean, prior_std), "b--", label="Prior", alpha=0.7)
    ax.plot(x, sp_stats.norm.pdf(x, posterior_mean, posterior_std), "r-", linewidth=2, label="Posterior")
    ax.axvline(credible_interval[0], color="gray", linestyle=":", alpha=0.5)
    ax.axvline(credible_interval[1], color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(
        x,
        0,
        sp_stats.norm.pdf(x, posterior_mean, posterior_std),
        where=(x >= credible_interval[0]) & (x <= credible_interval[1]),
        alpha=0.2,
        color="red",
        label=f"{credible_interval[0]:.2f}–{credible_interval[1]:.2f}",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def functional_plot(t, curves, mean_curve=None, title="Functional Data"):
    """Plot multiple functional curves with optional mean."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _i, curve in enumerate(curves):
        ax.plot(t, curve, alpha=0.3, color="blue", linewidth=0.5)
    if mean_curve is not None:
        ax.plot(t, mean_curve, "r-", linewidth=2, label="Mean Function")
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def cluster_scatter_2d(points, labels, centers=None, title="Cluster Visualization"):
    """2D scatter plot of clusters (use PCA if >2D)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(points[mask, 0], points[mask, 1], c=[colors[i]], label=f"Cluster {label}", alpha=0.6)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, linewidths=3, label="Centers")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def power_curve(effect_size, n_values, target_power=0.8, title="Power Curve"):
    """Generate power curve showing power vs sample size."""
    from scipy import stats as sp_stats

    powers = []
    for n in n_values:
        # Approximate power for two-sample t-test
        df = 2 * n - 2
        ncp = effect_size * np.sqrt(n / 2)
        t_crit = sp_stats.t.ppf(0.975, df)
        power = 1 - sp_stats.nct.cdf(t_crit, df, ncp)
        powers.append(power)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(n_values, powers, "b-", linewidth=2, label=f"Effect size={effect_size:.2f}")
    ax.axhline(y=target_power, color="red", linestyle="--", linewidth=1.5, label=f"Target power={target_power:.2f}")

    # Find minimum n for target power
    for i, p in enumerate(powers):
        if p >= target_power:
            ax.axvline(x=n_values[i], color="green", linestyle=":", linewidth=1)
            ax.text(n_values[i], 0.1, f"n={n_values[i]}", ha="center", fontsize=10, color="green")
            break

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Power (1-β)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)

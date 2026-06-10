"""Chart generation module - returns base64 PNG for JSON embedding."""

import base64
import io

import numpy as np


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def histogram(values, title="Distribution", bins=30, normal_curve=True):
    """Generate histogram with optional normal curve overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    if normal_curve and len(arr) > 2:
        x = np.linspace(arr.min(), arr.max(), 200)
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma > 0:
            ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
            ax.legend()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.3)
    return _fig_to_base64(fig)


def control_chart_plot(values, chart_type='imr', ucl=None, cl=None, lcl=None, out_of_control=None):
    """Generate control chart plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arr = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(arr)), arr, 'b-o', markersize=4, linewidth=1, label='Data')

    if cl is not None:
        ax.axhline(y=cl, color='green', linestyle='-', linewidth=1.5, label=f'CL={cl:.2f}')
    if ucl is not None:
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL={ucl:.2f}')
    if lcl is not None:
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL={lcl:.2f}')

    if out_of_control:
        ooc_x = [i for i in out_of_control if i < len(arr)]
        ooc_y = [arr[i] for i in ooc_x]
        ax.scatter(ooc_x, ooc_y, color='red', s=80, zorder=5, label='Out of Control')

    ax.set_title(f'{chart_type.upper()} Control Chart', fontsize=13)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def boxplot(groups, labels=None, title="Box Plot"):
    """Generate box plot for one or more groups."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, patch_artist=True, labels=labels)

    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    return _fig_to_base64(fig)


def scatter_with_regression(x, y, title="Scatter Plot", slope=None, intercept=None, r_squared=None):
    """Generate scatter plot with optional regression line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, color='steelblue', edgecolor='white', s=50)

    if slope is not None and intercept is not None:
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        label = f'y={slope:.2f}x+{intercept:.2f}'
        if r_squared is not None:
            label += f' (R²={r_squared:.4f})'
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=label)
        ax.legend()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def qq_plot(values, title="Q-Q Plot"):
    """Generate Q-Q plot for normality check."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(arr, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Normal curve
    mu, sigma = np.mean(arr), np.std(arr, ddof=1)
    if sigma > 0:
        x = np.linspace(arr.min() - sigma, arr.max() + sigma, 200)
        ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2)

    # Spec limits
    if usl is not None:
        ax.axvline(x=usl, color='red', linestyle='--', linewidth=2, label=f'USL={usl}')
    if lsl is not None:
        ax.axvline(x=lsl, color='red', linestyle='--', linewidth=2, label=f'LSL={lsl}')
    if target is not None:
        ax.axvline(x=target, color='green', linestyle='-', linewidth=2, label=f'Target={target}')

    # Mean
    ax.axvline(x=mu, color='blue', linestyle='-', linewidth=1.5, label=f'Mean={mu:.4f}')

    # Title with capability indices
    title_parts = ['Process Capability']
    if cp is not None:
        title_parts.append(f'Cp={cp:.2f}')
    if cpk is not None:
        title_parts.append(f'Cpk={cpk:.2f}')
    ax.set_title(' | '.join(title_parts), fontsize=13)

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    return _fig_to_base64(fig)


def time_series_plot(values, title="Time Series", fitted=None, forecast=None):
    """Generate time series plot with optional fitted values and forecast."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(values)), values, 'b-o', markersize=3, linewidth=1, label='Observed')

    if fitted is not None and len(fitted) == len(values):
        ax.plot(range(len(fitted)), fitted, 'r-', linewidth=1.5, label='Fitted')

    if forecast is not None:
        forecast_x = range(len(values), len(values) + len(forecast))
        ax.plot(forecast_x, forecast, 'g--', linewidth=2, marker='s', markersize=4, label='Forecast')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def heatmap(matrix, labels=None, title="Heatmap"):
    """Generate heatmap for correlation matrix or similar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    # Add values
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center', fontsize=9)

    fig.colorbar(im)
    ax.set_title(title, fontsize=13)
    return _fig_to_base64(fig)

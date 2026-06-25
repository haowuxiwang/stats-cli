"""Multivariate analysis: PCA, cluster, discriminant, correlation matrix, factor analysis, MANOVA."""

import numpy as np

from utils.output import DEFAULT_PRECISION, r


def multivariate(analysis_type, **kwargs):
    """Perform multivariate analysis.

    Args:
        analysis_type: 'pca', 'cluster', 'discriminant', 'correlation_matrix',
                       'factor_analysis', 'manova'

    Returns:
        Dict with multivariate results
    """
    if analysis_type == "pca":
        return _pca(**kwargs)
    elif analysis_type == "cluster":
        return _cluster(**kwargs)
    elif analysis_type == "discriminant":
        return _discriminant(**kwargs)
    elif analysis_type == "correlation_matrix":
        return _correlation_matrix(**kwargs)
    elif analysis_type == "factor_analysis":
        return _factor_analysis(**kwargs)
    elif analysis_type == "manova":
        return _manova(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _pca(values=None, file=None, columns=None, n_components=None, **kwargs):
    """Principal Component Analysis."""
    if file and columns:
        from utils.data_loader import read_dataframe

        df = read_dataframe(file, columns=columns)
        X = df.dropna().values
    elif values is not None:
        X = np.array(values, dtype=float)
    else:
        raise ValueError("Provide 'values' or 'file' + 'columns'")

    # Standardize
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

    # Covariance matrix
    cov_matrix = np.cov(X_std.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cum_var = np.cumsum(var_explained)

    if n_components is None:
        # Keep components explaining 95% variance
        n_components = int(np.searchsorted(cum_var, 0.95) + 1)
        n_components = min(n_components, len(eigenvalues))

    # Loadings
    loadings = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])

    col_names = columns if columns else [f"Var{i + 1}" for i in range(X.shape[1])]

    return {
        "analysis_type": "pca",
        "n_observations": X.shape[0],
        "n_variables": X.shape[1],
        "n_components": n_components,
        "eigenvalues": [r(v) for v in eigenvalues],
        "variance_explained": [r(v) for v in var_explained],
        "cumulative_variance": [r(v) for v in cum_var],
        "loadings": {col_names[i]: [r(loadings[i, j]) for j in range(n_components)] for i in range(len(col_names))},
        "interpretation": f"First {n_components} components explain {r(cum_var[n_components - 1] * 100, 1)}% of variance",
    }


def _cluster(values=None, file=None, columns=None, method="kmeans", n_clusters=3, **kwargs):
    """Cluster analysis."""
    if file and columns:
        from utils.data_loader import read_dataframe

        df = read_dataframe(file, columns=columns)
        X = df.dropna().values
    elif values is not None:
        X = np.array(values, dtype=float)
    else:
        raise ValueError("Provide 'values' or 'file' + 'columns'")

    # Standardize
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

    if method == "kmeans":
        return _kmeans(X_std, n_clusters)
    elif method == "hierarchical":
        return _hierarchical(X_std, n_clusters)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hierarchical'")


def _kmeans(X, n_clusters):
    """K-means clustering."""
    from scipy.cluster.vq import kmeans, vq

    centroids, _ = kmeans(X, n_clusters)
    labels, _ = vq(X, centroids)

    # Cluster sizes
    cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]

    # Within-cluster sum of squares
    wcss = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)

    return {
        "analysis_type": "cluster",
        "method": "kmeans",
        "n_clusters": n_clusters,
        "n_observations": X.shape[0],
        "cluster_sizes": cluster_sizes,
        "centroids": [[r(v) for v in c] for c in centroids],
        "wcss": r(wcss),
        "labels": [int(lbl) for lbl in labels],
    }


def _hierarchical(X, n_clusters):
    """Hierarchical clustering."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    Z = linkage(pdist(X), method="ward")
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_sizes = [int(np.sum(labels == i)) for i in range(1, n_clusters + 1)]

    return {
        "analysis_type": "cluster",
        "method": "hierarchical",
        "n_clusters": n_clusters,
        "n_observations": X.shape[0],
        "cluster_sizes": cluster_sizes,
        "labels": [int(lbl) for lbl in labels],
    }


def _discriminant(values=None, file=None, columns=None, group_column=None, **kwargs):
    """Discriminant analysis (LDA)."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError:
        raise ImportError("scikit-learn required for discriminant analysis")

    if file and columns and group_column:
        from utils.data_loader import read_dataframe

        # Include group_column in the columns list to load it
        all_columns = list(dict.fromkeys(columns + [group_column]))  # deduplicate preserving order
        df = read_dataframe(file, columns=all_columns)
        X = df[columns].dropna().values
        y = df[group_column].dropna().values
    else:
        raise ValueError("Provide 'file', 'columns', and 'group_column'")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    return {
        "analysis_type": "discriminant",
        "n_observations": X.shape[0],
        "n_variables": X.shape[1],
        "n_groups": len(np.unique(y)),
        "accuracy": r(lda.score(X, y)),
        "explained_variance_ratio": [r(v) for v in lda.explained_variance_ratio_],
    }


def _correlation_matrix(values=None, file=None, columns=None, method="pearson", **kwargs):
    """Correlation matrix."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for correlation matrix")

    if file:
        from utils.data_loader import read_dataframe

        df = read_dataframe(file, columns=columns)
        if columns:
            pass  # columns already selected by read_dataframe
        df = df.select_dtypes(include=["number"]).dropna()
    elif values is not None:
        df = pd.DataFrame(values)
    else:
        raise ValueError("Provide 'values' or 'file'")

    if method == "pearson":
        corr = df.corr(method="pearson")
    elif method == "spearman":
        corr = df.corr(method="spearman")
    elif method == "kendall":
        corr = df.corr(method="kendall")
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "analysis_type": "correlation_matrix",
        "method": method,
        "n_variables": len(corr.columns),
        "variables": list(corr.columns),
        "matrix": corr.round(DEFAULT_PRECISION).to_dict(),
    }


def _varimax_loadings(loadings, max_iter=100, tol=1e-6):
    """Apply varimax rotation to loading matrix.

    Uses Kaiser's pairwise rotation algorithm.

    Args:
        loadings: (p x k) loading matrix
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        Rotated loading matrix
    """
    p, k = loadings.shape
    rotation = np.eye(k)

    for _ in range(max_iter):
        old_rotation = rotation.copy()
        rotated = loadings @ rotation

        # Pairwise rotation for all factor pairs
        for j in range(k):
            for m in range(j + 1, k):
                # Columns j and m
                lj = rotated[:, j]
                lm = rotated[:, m]

                # Varimax criterion gradient components
                u = lj**2 - lm**2
                v = 2 * lj * lm

                # Compute rotation angle
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)

                numerator = D - 2 * A * B / p
                denominator = C - (A**2 - B**2) / p

                angle = 0.25 * np.arctan2(numerator, denominator)

                # Apply Givens rotation to columns j and m
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.eye(k)
                rot[j, j] = cos_a
                rot[m, m] = cos_a
                rot[j, m] = -sin_a
                rot[m, j] = sin_a
                rotation = rotation @ rot

        if np.max(np.abs(rotation - old_rotation)) < tol:
            break

    return loadings @ rotation


def _promax_loadings(loadings, power=4):
    """Apply promax (oblique) rotation to loading matrix.

    Args:
        loadings: (p x k) loading matrix (should be varimax-rotated first)
        power: power to raise target matrix (default 4)

    Returns:
        Rotated loading matrix and factor correlation matrix
    """
    p, k = loadings.shape

    # Start from varimax rotation
    varimax_load = _varimax_loadings(loadings)

    # Create target matrix by raising absolute values to power
    target = np.sign(varimax_load) * np.abs(varimax_load) ** power

    # Find transformation matrix via procrustes rotation
    # T = (L' L)^-1 L' T_target  (simplified)
    LtL = varimax_load.T @ varimax_load
    LtT = varimax_load.T @ target

    try:
        transform = np.linalg.solve(LtL, LtT)
    except np.linalg.LinAlgError:
        return varimax_load, np.eye(k)

    # Normalize columns
    for j in range(k):
        norm = np.sqrt(transform[:, j] @ transform[:, j])
        if norm > 1e-12:
            transform[:, j] /= norm

    rotated = varimax_load @ transform

    # Factor correlation matrix
    factor_corr = transform.T @ transform

    return rotated, factor_corr


def _factor_analysis(values=None, file=None, columns=None, n_factors=None, rotation="varimax", **kwargs):
    """Factor Analysis.

    Args:
        values: 2D array of data
        file: data file path
        columns: column names
        n_factors: number of factors to extract
        rotation: 'varimax', 'promax', or 'none'

    Returns:
        Dict with factor analysis results
    """
    if file and columns:
        from utils.data_loader import read_dataframe

        df = read_dataframe(file, columns=columns)
        X = df.dropna().values
    elif values is not None:
        X = np.array(values, dtype=float)
    else:
        raise ValueError("Provide 'values' or 'file' + 'columns'")

    n_obs, n_vars = X.shape

    if n_vars < 2:
        raise ValueError("Factor analysis requires at least 2 variables")

    # Standardize
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

    # Correlation matrix
    corr = np.corrcoef(X_std.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Default n_factors: Kaiser criterion (eigenvalue > 1)
    if n_factors is None:
        n_factors = int(np.sum(eigenvalues > 1.0))
        if n_factors == 0:
            n_factors = 1

    if n_factors < 1 or n_factors > n_vars:
        raise ValueError(f"n_factors must be between 1 and {n_vars}")

    # Extract initial loadings (principal component method)
    # L = V * sqrt(lambda)
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))

    # Apply rotation
    factor_corr = None
    if rotation == "varimax":
        loadings = _varimax_loadings(loadings)
    elif rotation == "promax":
        loadings, factor_corr = _promax_loadings(loadings)
    elif rotation == "none":
        pass
    else:
        raise ValueError(f"Unknown rotation: {rotation}. Use 'varimax', 'promax', or 'none'")

    # Communalities (sum of squared loadings per variable)
    communalities = np.sum(loadings**2, axis=1)

    # Uniquenesses
    uniquenesses = 1.0 - communalities

    # Variance explained by each factor (after rotation)
    var_explained = np.sum(loadings**2, axis=0)
    total_var = np.sum(communalities)
    var_explained_pct = var_explained / n_vars  # percentage of total variance

    col_names = columns if columns else [f"Var{i + 1}" for i in range(n_vars)]

    result = {
        "analysis_type": "factor_analysis",
        "n_observations": n_obs,
        "n_variables": n_vars,
        "n_factors": n_factors,
        "rotation": rotation,
        "eigenvalues": [r(v) for v in eigenvalues],
        "loadings": {col_names[i]: [r(loadings[i, j]) for j in range(n_factors)] for i in range(n_vars)},
        "communalities": {col_names[i]: r(communalities[i]) for i in range(n_vars)},
        "uniquenesses": {col_names[i]: r(uniquenesses[i]) for i in range(n_vars)},
        "variance_explained": [r(v) for v in var_explained_pct],
        "total_variance_explained": r(total_var / n_vars),
        "interpretation": (
            f"{n_factors} factor(s) with {rotation} rotation explain "
            f"{r(total_var / n_vars * 100, 1)}% of total variance"
        ),
    }

    if factor_corr is not None:
        result["factor_correlation"] = [[r(factor_corr[i, j]) for j in range(n_factors)] for i in range(n_factors)]

    return result


def _manova(groups=None, file=None, columns=None, group_column=None, **kwargs):
    """Multivariate Analysis of Variance (MANOVA).

    Args:
        groups: list of 2D arrays, each group has multiple dependent variables
        file: data file path
        columns: dependent variable column names
        group_column: column identifying groups

    Returns:
        Dict with MANOVA results (Wilks' Lambda, Pillai's Trace, etc.)
    """
    if file and columns and group_column:
        from utils.data_loader import read_dataframe

        all_columns = list(dict.fromkeys(columns + [group_column]))
        df = read_dataframe(file, columns=all_columns)
        X = df[columns].dropna().values
        y = df[group_column].dropna().values
        unique_groups = np.unique(y)
        group_data = [X[y == g] for g in unique_groups]
    elif groups is not None:
        group_data = [np.array(g, dtype=float) for g in groups]
        unique_groups = list(range(len(group_data)))
    else:
        raise ValueError("Provide 'groups' or 'file' + 'columns' + 'group_column'")

    n_groups = len(group_data)
    if n_groups < 2:
        raise ValueError("MANOVA requires at least 2 groups")

    n_vars = group_data[0].shape[1]
    for i, g in enumerate(group_data):
        if g.ndim == 1:
            group_data[i] = g.reshape(-1, 1)
        if group_data[i].shape[1] != n_vars:
            raise ValueError("All groups must have the same number of dependent variables")
        if group_data[i].shape[0] < 2:
            raise ValueError(f"Group {i} must have at least 2 observations")

    n_per_group = [g.shape[0] for g in group_data]
    n_total = sum(n_per_group)
    p = n_vars  # number of dependent variables
    k = n_groups  # number of groups

    # Grand mean
    all_data = np.vstack(group_data)
    grand_mean = np.mean(all_data, axis=0)

    # Group means
    group_means = [np.mean(g, axis=0) for g in group_data]

    # Hypothesis matrix H (between-groups SSCP)
    H = np.zeros((p, p))
    for i in range(k):
        diff = (group_means[i] - grand_mean).reshape(-1, 1)
        H += n_per_group[i] * (diff @ diff.T)

    # Error matrix W (within-groups SSCP)
    W = np.zeros((p, p))
    for i in range(k):
        centered = group_data[i] - group_means[i]
        W += centered.T @ centered

    # Degrees of freedom
    df_hypothesis = k - 1
    df_error = n_total - k

    # Compute the four MANOVA test statistics
    # Eigenvalues of W^-1 H
    try:
        W_inv = np.linalg.inv(W)
        eigs = np.linalg.eigvalsh(W_inv @ H)
        eigs = np.maximum(np.real(eigs), 0)  # ensure non-negative
        eigs = np.sort(eigs)[::-1]
    except np.linalg.LinAlgError:
        # W is singular; use pseudo-inverse
        W_inv = np.linalg.pinv(W)
        eigs = np.linalg.eigvalsh(W_inv @ H)
        eigs = np.maximum(np.real(eigs), 0)
        eigs = np.sort(eigs)[::-1]

    s = min(p, df_hypothesis)  # number of non-zero eigenvalues

    # 1. Wilks' Lambda = product of 1/(1 + eigenvalue)
    wilks_lambda = np.prod(1.0 / (1.0 + eigs[:s]))

    # 2. Pillai's Trace = sum of eigenvalue/(1 + eigenvalue)
    pillai_trace = np.sum(eigs[:s] / (1.0 + eigs[:s]))

    # 3. Hotelling-Lawley Trace = sum of eigenvalues
    hotelling_trace = np.sum(eigs[:s])

    # 4. Roy's Largest Root = max eigenvalue
    roy_root = eigs[0] if len(eigs) > 0 else 0.0

    # F-approximations
    from scipy import stats

    # Wilks' Lambda F-approximation (Rao's)
    if s == 1:
        # Exact F
        f_wilks = (1 - wilks_lambda) / wilks_lambda * df_error / df_hypothesis
        df1_wilks = df_hypothesis
        df2_wilks = df_error
    elif s == 2:
        f_wilks = (1 - np.sqrt(wilks_lambda)) / np.sqrt(wilks_lambda) * (df_error - 1) / df_hypothesis
        df1_wilks = 2 * df_hypothesis
        df2_wilks = 2 * (df_error - 1)
    else:
        # Bartlett's approximation for general case
        chi_sq = -(df_error - (p - df_hypothesis + 1) / 2) * np.log(wilks_lambda)
        df_wilks_chi = p * df_hypothesis
        p_wilks = 1 - stats.chi2.cdf(chi_sq, df_wilks_chi)
        f_wilks = None
        df1_wilks = None
        df2_wilks = None

    if f_wilks is not None:
        df1_wilks = int(df1_wilks)
        df2_wilks = int(df2_wilks)
        p_wilks = 1 - stats.f.cdf(f_wilks, df1_wilks, df2_wilks)

    # Pillai's Trace F-approximation
    f_pillai = (
        (pillai_trace / s) / ((1 - pillai_trace) / (df_error - p + s)) if (1 - pillai_trace) > 0 else float("inf")
    )
    df1_pillai = int(s * (p + s))
    df2_pillai = int(s * (df_error - p + s))
    p_pillai = 1 - stats.f.cdf(f_pillai, df1_pillai, df2_pillai)

    # Hotelling-Lawley F-approximation
    f_hl = (hotelling_trace / s) * ((df_error - p - 1) / p) if p > 0 else 0
    df1_hl = int(s * p)
    df2_hl = int(s * (df_error - p - 1))
    if df2_hl > 0:
        p_hl = 1 - stats.f.cdf(f_hl, df1_hl, df2_hl)
    else:
        p_hl = None

    # Roy's Root F-approximation (upper bound)
    f_roy = roy_root * (df_error - p - 1) / p if p > 0 else 0
    df1_roy = int(p)
    df2_roy = int(df_error - p - 1)
    if df2_roy > 0:
        p_roy = 1 - stats.f.cdf(f_roy, df1_roy, df2_roy)
    else:
        p_roy = None

    group_labels = list(unique_groups) if file else list(range(n_groups))

    return {
        "analysis_type": "manova",
        "n_groups": n_groups,
        "n_variables": p,
        "n_total": n_total,
        "n_per_group": {str(group_labels[i]): n_per_group[i] for i in range(n_groups)},
        "df_hypothesis": int(df_hypothesis),
        "df_error": int(df_error),
        "statistics": {
            "wilks_lambda": r(wilks_lambda),
            "pillai_trace": r(pillai_trace),
            "hotelling_trace": r(hotelling_trace),
            "roys_root": r(roy_root),
        },
        "f_values": {
            "wilks_lambda": r(f_wilks) if f_wilks is not None else None,
            "pillai_trace": r(f_pillai),
            "hotelling_trace": r(f_hl),
            "roys_root": r(f_roy),
        },
        "p_values": {
            "wilks_lambda": r(p_wilks),
            "pillai_trace": r(p_pillai),
            "hotelling_trace": r(p_hl) if p_hl is not None else None,
            "roys_root": r(p_roy) if p_roy is not None else None,
        },
        "df": {
            "wilks_lambda": {"df1": df1_wilks, "df2": df2_wilks}
            if f_wilks is not None
            else {"chi2_df": int(p * df_hypothesis)},
            "pillai_trace": {"df1": df1_pillai, "df2": df2_pillai},
            "hotelling_trace": {"df1": df1_hl, "df2": df2_hl},
            "roys_root": {"df1": df1_roy, "df2": df2_roy},
        },
        "interpretation": (
            f"MANOVA with {n_groups} groups and {p} dependent variables: "
            f"Wilks' Lambda = {r(wilks_lambda)}, "
            f"significant at p = {r(p_wilks)}"
        ),
    }

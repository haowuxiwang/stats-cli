"""Multivariate analysis: PCA, cluster, discriminant, correlation matrix."""

import numpy as np

from utils.output import DEFAULT_PRECISION, r


def multivariate(analysis_type, **kwargs):
    """Perform multivariate analysis.

    Args:
        analysis_type: 'pca', 'cluster', 'discriminant', 'correlation_matrix'

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
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _pca(values=None, file=None, columns=None, n_components=None, **kwargs):
    """Principal Component Analysis."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for PCA")

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
        "loadings": {
            col_names[i]: [r(loadings[i, j]) for j in range(n_components)] for i in range(len(col_names))
        },
        "interpretation": f"First {n_components} components explain {r(cum_var[n_components - 1] * 100, 1)}% of variance",
    }


def _cluster(values=None, file=None, columns=None, method="kmeans", n_clusters=3, **kwargs):
    """Cluster analysis."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for cluster analysis")

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
        import pandas as pd
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError:
        raise ImportError("scikit-learn required for discriminant analysis")

    if file and columns and group_column:
        from utils.data_loader import read_dataframe
        df = read_dataframe(file, columns=columns)
        X = df.dropna().values
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

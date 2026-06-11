"""Tests for stats_engine/multivariate.py."""

import numpy as np
import pytest

from stats_engine.multivariate import multivariate


@pytest.fixture
def pca_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 50)
    x2 = x1 * 0.8 + np.random.normal(0, 0.5, 50)
    x3 = x1 * 0.5 + np.random.normal(0, 0.8, 50)
    return np.column_stack([x1, x2, x3])


def test_pca(pca_data):
    columns = ["x1", "x2", "x3"]
    values = pca_data.tolist()
    result = multivariate(analysis_type="pca", values=values, columns=columns)
    assert result["analysis_type"] == "pca"
    assert "eigenvalues" in result
    assert "loadings" in result
    assert "variance_explained" in result


def test_cluster_kmeans(pca_data):
    values = pca_data.tolist()
    result = multivariate(analysis_type="cluster", values=values, method="kmeans", n_clusters=2)
    assert result["analysis_type"] == "cluster"
    assert result["method"] == "kmeans"
    assert "labels" in result
    assert "centroids" in result


def test_cluster_hierarchical(pca_data):
    values = pca_data.tolist()
    result = multivariate(analysis_type="cluster", values=values, method="hierarchical", n_clusters=2)
    assert result["analysis_type"] == "cluster"
    assert result["method"] == "hierarchical"


def test_correlation_matrix(pca_data):
    columns = ["x1", "x2", "x3"]
    values = pca_data.tolist()
    result = multivariate(analysis_type="correlation_matrix", values=values, columns=columns)
    assert result["analysis_type"] == "correlation_matrix"
    assert "matrix" in result


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        multivariate(analysis_type="invalid", values=[[1, 2]])


def test_pca_no_values():
    """PCA without values should raise error."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="pca")


def test_cluster_no_values():
    """Cluster without values should raise error."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="cluster")


def test_cluster_unknown_method():
    """Unknown cluster method should raise error."""
    with pytest.raises(ValueError, match="Unknown method"):
        multivariate(analysis_type="cluster", values=[[1, 2], [3, 4]], method="invalid")


def test_correlation_matrix_spearman(pca_data):
    """Spearman correlation matrix."""
    result = multivariate(analysis_type="correlation_matrix", values=pca_data.tolist(), method="spearman")
    assert result["method"] == "spearman"
    assert "matrix" in result


def test_correlation_matrix_kendall(pca_data):
    """Kendall correlation matrix."""
    result = multivariate(analysis_type="correlation_matrix", values=pca_data.tolist(), method="kendall")
    assert result["method"] == "kendall"
    assert "matrix" in result


def test_correlation_matrix_unknown_method(pca_data):
    """Unknown correlation method should raise error."""
    with pytest.raises(ValueError, match="Unknown method"):
        multivariate(analysis_type="correlation_matrix", values=pca_data.tolist(), method="invalid")


def test_correlation_matrix_no_values():
    """Correlation matrix without values should raise error."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="correlation_matrix")


def test_discriminant_no_file():
    """Discriminant without file should raise error."""
    with pytest.raises(ValueError, match="Provide"):
        multivariate(analysis_type="discriminant")

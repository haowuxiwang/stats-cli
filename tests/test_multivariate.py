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

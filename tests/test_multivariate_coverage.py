"""Additional tests for stats_engine/multivariate.py to improve coverage."""

import numpy as np

from main import handler


class TestMultivariateEdgeCases:
    """Test multivariate edge cases."""

    def test_pca_basic(self):
        """PCA with valid data."""
        data = np.random.randn(50, 3).tolist()
        result = handler(
            {
                "command": "multivariate",
                "params": {
                    "analysis_type": "pca",
                    "values": data,
                },
            }
        )
        assert result["status"] == "success"
        assert "n_components" in result["data"]

    def test_correlation_matrix_basic(self):
        """Correlation matrix with valid data."""
        data = np.random.randn(50, 3).tolist()
        result = handler(
            {
                "command": "multivariate",
                "params": {
                    "analysis_type": "correlation_matrix",
                    "values": data,
                },
            }
        )
        assert result["status"] == "success"
        assert "matrix" in result["data"]

    def test_discriminant_with_file(self, tmp_path):
        """Discriminant analysis with file input."""
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "x2", "group"])
            for i in range(20):
                writer.writerow([i, i * 2, "A"])
            for i in range(20):
                writer.writerow([i + 100, i * 2 + 100, "B"])

        result = handler(
            {
                "command": "multivariate",
                "params": {
                    "analysis_type": "discriminant",
                    "file": str(csv_file),
                    "columns": ["x1", "x2"],
                    "group_column": "group",
                },
            }
        )
        assert result["status"] == "success"
        assert "accuracy" in result["data"]

    def test_cluster_basic(self):
        """Cluster analysis with valid data."""
        data = np.random.randn(30, 2).tolist()
        result = handler(
            {
                "command": "multivariate",
                "params": {
                    "analysis_type": "cluster",
                    "values": data,
                    "n_clusters": 3,
                },
            }
        )
        assert result["status"] == "success"
        assert "labels" in result["data"]

"""Tests for stats_engine/mining.py."""

import json

import numpy as np
import pytest

from stats_engine.mining import mining


class TestClassify:
    def test_decision_tree(self):
        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        result = mining("classify", features=features, labels=labels, method="decision_tree")
        assert result["analysis_type"] == "classify"
        assert result["method"] == "decision_tree"
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert "feature_importance" in result
        assert "classification_report" in result

    def test_random_forest(self):
        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        result = mining("classify", features=features, labels=labels, method="random_forest")
        assert result["method"] == "random_forest"
        assert result["accuracy"] >= 0
        assert result["accuracy"] <= 1

    def test_classify_multi_class(self):
        np.random.seed(42)
        features = np.random.randn(60, 3).tolist()
        labels = [0] * 20 + [1] * 20 + [2] * 20
        result = mining("classify", features=features, labels=labels)
        assert result["n_classes"] == 3

    def test_classify_feature_importance_sorted(self):
        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        result = mining("classify", features=features, labels=labels)
        importances = [f["importance"] for f in result["feature_importance"]]
        assert importances == sorted(importances, reverse=True)

    def test_classify_json_serializable(self):
        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        result = mining("classify", features=features, labels=labels)
        json.dumps(result)  # Should not raise


class TestAnomaly:
    def test_isolation_forest(self):
        values = [10, 11, 10, 12, 10, 11, 10, 100, 11, 10, 500, 10, 11, 12, 10]
        result = mining("anomaly", values=values, method="isolation_forest")
        assert result["analysis_type"] == "anomaly"
        assert result["method"] == "isolation_forest"
        assert "n_anomalies" in result
        assert "anomaly_indices" in result
        assert "anomaly_rate" in result

    def test_lof(self):
        values = [10, 11, 10, 12, 10, 11, 10, 100, 11, 10, 500, 10, 11, 12, 10]
        result = mining("anomaly", values=values, method="lof")
        assert result["method"] == "lof"

    def test_zscore(self):
        values = [10, 11, 10, 12, 10, 11, 10, 100, 11, 10, 500, 10, 11, 12, 10]
        result = mining("anomaly", values=values, method="zscore")
        assert result["method"] == "zscore"

    def test_ensemble(self):
        values = [10, 11, 10, 12, 10, 11, 10, 100, 11, 10, 500, 10, 11, 12, 10]
        result = mining("anomaly", values=values, method="ensemble")
        assert result["method"] == "ensemble"
        assert "details" in result
        assert "ensemble" in result["details"]

    def test_anomaly_multivariate(self):
        np.random.seed(42)
        data = np.random.randn(50, 3).tolist()
        # Add some outliers
        data.append([10, 10, 10])
        data.append([-10, -10, -10])
        result = mining("anomaly", data=data, method="isolation_forest")
        assert result["n_dimensions"] == 3

    def test_anomaly_json_serializable(self):
        values = [10, 11, 10, 12, 10, 11, 10, 100, 11, 10, 500, 10, 11, 12, 10]
        result = mining("anomaly", values=values)
        json.dumps(result)  # Should not raise


class TestAssociate:
    def test_basic_association(self):
        transactions = [
            ["bread", "milk"],
            ["bread", "diapers", "beer"],
            ["milk", "diapers", "beer"],
            ["bread", "milk", "diapers"],
            ["bread", "milk", "beer"],
        ]
        result = mining("associate", transactions=transactions, min_support=0.3, min_confidence=0.5)
        assert result["analysis_type"] == "associate"
        assert result["n_transactions"] == 5
        assert "frequent_itemsets" in result
        assert "rules" in result

    def test_association_rules_sorted_by_lift(self):
        transactions = [
            ["a", "b", "c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
            ["a", "b", "c"],
            ["a", "b"],
        ]
        result = mining("associate", transactions=transactions, min_support=0.3, min_confidence=0.5)
        lifts = [r["lift"] for r in result["rules"]]
        assert lifts == sorted(lifts, reverse=True)

    def test_association_high_support(self):
        transactions = [
            ["a", "b"],
            ["a", "b"],
            ["a", "b"],
            ["a", "b"],
            ["a", "b"],
        ]
        result = mining("associate", transactions=transactions, min_support=0.9, min_confidence=0.9)
        # a and b should have support 1.0
        for itemset in result["frequent_itemsets"]:
            if set(itemset["items"]) == {"a", "b"}:
                assert itemset["support"] == 1.0

    def test_association_json_serializable(self):
        transactions = [
            ["bread", "milk"],
            ["bread", "diapers", "beer"],
            ["milk", "diapers", "beer"],
            ["bread", "milk", "diapers"],
            ["bread", "milk", "beer"],
        ]
        result = mining("associate", transactions=transactions, min_support=0.3, min_confidence=0.5)
        json.dumps(result)  # Should not raise


class TestMiningGuards:
    """Guard tests for mining.py error paths."""

    def test_classify_1d_features(self):
        with pytest.raises(ValueError, match="2D"):
            mining("classify", features=[1, 2, 3], labels=[0, 1, 1])

    def test_classify_mismatched_labels(self):
        with pytest.raises(ValueError, match="same length"):
            mining("classify", features=[[1, 2], [3, 4]], labels=[0])

    def test_classify_bad_method(self):
        np.random.seed(42)
        features = np.random.randn(50, 3).tolist()
        labels = [0] * 25 + [1] * 25
        with pytest.raises(ValueError, match="Unknown method"):
            mining("classify", features=features, labels=labels, method="svm")

    def test_anomaly_bad_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            mining("anomaly", values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], method="invalid")

    def test_anomaly_no_data(self):
        with pytest.raises(ValueError, match="Provide either"):
            mining("anomaly")

    def test_anomaly_too_few(self):
        with pytest.raises(ValueError, match="at least"):
            mining("anomaly", values=[1, 2, 3])

    def test_anomaly_data_1d(self):
        """Pass 1D array as data should reshape to 2D."""
        result = mining("anomaly", data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], method="isolation_forest")
        assert result["n_dimensions"] == 1

    def test_associate_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            mining("associate", transactions=[])

    def test_associate_all_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            mining("associate", transactions=[[], []])

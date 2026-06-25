"""Data mining: classification, anomaly detection, and association rules."""

import numpy as np

from utils.output import r
from utils.validators import to_array


def mining(analysis_type, **kwargs):
    """Perform data mining analysis.

    Args:
        analysis_type: 'classify', 'anomaly', 'associate'

    Returns:
        Dict with mining results
    """
    if analysis_type == "classify":
        return _classify(**kwargs)
    elif analysis_type == "anomaly":
        return _anomaly(**kwargs)
    elif analysis_type == "associate":
        return _associate(**kwargs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Use: classify, anomaly, associate")


def _classify(features, labels, method="random_forest", test_size=0.3, random_state=42, **kwargs):
    """Classify data using decision tree or random forest.

    Args:
        features: 2D array of feature values (list of lists)
        labels: 1D array of class labels
        method: 'decision_tree' or 'random_forest'
        test_size: Fraction of data for testing (default 0.3)
        random_state: Random seed for reproducibility

    Returns:
        Dict with classification results, accuracy, feature importance, confusion matrix
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    features = np.array(features, dtype=float)
    labels = np.array(labels)

    if features.ndim != 2:
        raise ValueError("features must be a 2D array (list of lists)")
    if len(labels) != len(features):
        raise ValueError(f"features ({len(features)}) and labels ({len(labels)}) must have same length")

    n_samples, n_features = features.shape
    if n_samples < 10:
        raise ValueError("Need at least 10 samples for classification")

    method = method.lower()
    if method not in ("decision_tree", "random_forest"):
        raise ValueError(f"Unknown method: {method}. Use 'decision_tree' or 'random_forest'")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    # Train model
    if method == "decision_tree":
        model = DecisionTreeClassifier(random_state=random_state, max_depth=10)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(labels))

    # Feature importance
    importances = model.feature_importances_
    feature_importance = [{"feature": f"f{i}", "importance": r(imp)} for i, imp in enumerate(importances)]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    result = {
        "analysis_type": "classify",
        "method": method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": len(classes),
        "classes": [str(c) for c in classes],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": r(accuracy),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "classification_report": {
            str(k): {kk: r(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in report.items()
            if k in [str(c) for c in classes] or k in ("macro avg", "weighted avg")
        },
        "interpretation": (
            f"{method.replace('_', ' ').title()} achieved {r(accuracy * 100, 2)}% accuracy "
            f"on {n_samples} samples with {n_features} features. "
            f"Top feature: {feature_importance[0]['feature']} ({feature_importance[0]['importance']})"
        ),
    }

    return result


def _anomaly(values=None, data=None, method="isolation_forest", contamination=0.05, **kwargs):
    """Detect anomalies using Isolation Forest, LOF, or Z-score ensemble.

    Args:
        values: 1D numeric data (for univariate detection)
        data: 2D array (for multivariate detection)
        method: 'isolation_forest', 'lof', 'zscore', or 'ensemble'
        contamination: Expected proportion of anomalies (default 0.05)

    Returns:
        Dict with anomaly detection results
    """
    method = method.lower()
    if method not in ("isolation_forest", "lof", "zscore", "ensemble"):
        raise ValueError(f"Unknown method: {method}. Use: isolation_forest, lof, zscore, ensemble")

    # Prepare data
    if data is not None:
        X = np.array(data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        is_multivariate = X.shape[1] > 1
    elif values is not None:
        arr = to_array(values, min_n=5, name="values")
        X = arr.reshape(-1, 1)
        is_multivariate = False
    else:
        raise ValueError("Provide either 'values' (1D) or 'data' (2D)")

    n_samples = X.shape[0]
    if n_samples < 10:
        raise ValueError("Need at least 10 data points for anomaly detection")

    # Z-score method (univariate only or column-wise)
    def _zscore_detect(data_2d, contam):
        stds = np.std(data_2d, axis=0, ddof=1)
        stds[stds < 1e-10] = 1.0  # avoid division by zero
        scores = np.abs((data_2d - np.mean(data_2d, axis=0)) / stds)
        max_scores = np.max(scores, axis=1)
        threshold = np.percentile(max_scores, 100 * (1 - contam))
        return max_scores >= threshold, max_scores

    # Isolation Forest
    def _isoforest_detect(data_2d, contam):
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(contamination=contam, random_state=42)
        preds = model.fit_predict(data_2d)
        scores = -model.decision_function(data_2d)
        return preds == -1, scores

    # LOF
    def _lof_detect(data_2d, contam):
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(n_neighbors=min(20, n_samples - 1), contamination=contam)
        preds = model.fit_predict(data_2d)
        scores = -model.negative_outlier_factor_
        return preds == -1, scores

    results = {}

    if method == "zscore" or method == "ensemble":
        mask, scores = _zscore_detect(X, contamination)
        results["zscore"] = {"n_anomalies": int(np.sum(mask)), "anomaly_indices": np.where(mask)[0].tolist()}

    if method == "isolation_forest" or method == "ensemble":
        mask, scores = _isoforest_detect(X, contamination)
        results["isolation_forest"] = {"n_anomalies": int(np.sum(mask)), "anomaly_indices": np.where(mask)[0].tolist()}

    if method == "lof" or method == "ensemble":
        mask, scores = _lof_detect(X, contamination)
        results["lof"] = {"n_anomalies": int(np.sum(mask)), "anomaly_indices": np.where(mask)[0].tolist()}

    # For ensemble: majority vote
    if method == "ensemble":
        vote_matrix = np.zeros(n_samples, dtype=int)
        for m in ["zscore", "isolation_forest", "lof"]:
            for idx in results[m]["anomaly_indices"]:
                vote_matrix[idx] += 1
        ensemble_mask = vote_matrix >= 2  # at least 2 out of 3 methods agree
        results["ensemble"] = {
            "n_anomalies": int(np.sum(ensemble_mask)),
            "anomaly_indices": np.where(ensemble_mask)[0].tolist(),
            "vote_threshold": 2,
        }
        primary = results["ensemble"]
    else:
        primary = results[method]

    # Compute statistics for normal vs anomalous points
    anomaly_indices = set(primary["anomaly_indices"])
    normal_mask = np.array([i not in anomaly_indices for i in range(n_samples)])
    anomaly_mask = np.array([i in anomaly_indices for i in range(n_samples)])

    stats = {}
    if not is_multivariate:
        normal_data = X[normal_mask, 0]
        anomaly_data = X[anomaly_mask, 0]
        if len(normal_data) > 1:
            stats["normal_mean"] = r(np.mean(normal_data))
            stats["normal_std"] = r(np.std(normal_data, ddof=1))
        elif len(normal_data) == 1:
            stats["normal_mean"] = r(normal_data[0])
        if len(anomaly_data) > 1:
            stats["anomaly_mean"] = r(np.mean(anomaly_data))
            stats["anomaly_std"] = r(np.std(anomaly_data, ddof=1))
        elif len(anomaly_data) == 1:
            stats["anomaly_mean"] = r(anomaly_data[0])

    return {
        "analysis_type": "anomaly",
        "method": method,
        "n_samples": n_samples,
        "n_dimensions": X.shape[1],
        "contamination": contamination,
        "n_anomalies": primary["n_anomalies"],
        "anomaly_indices": primary["anomaly_indices"],
        "anomaly_rate": r(primary["n_anomalies"] / n_samples, 4),
        "details": results,
        "stats": stats,
        "interpretation": (
            f"Detected {primary['n_anomalies']} anomalies "
            f"({r(primary['n_anomalies'] / n_samples * 100, 2)}%) "
            f"using {method} (contamination={contamination})"
        ),
    }


def _associate(transactions, min_support=0.3, min_confidence=0.7, max_length=3, **kwargs):
    """Mine association rules using Apriori algorithm.

    Args:
        transactions: List of transactions, each transaction is a list of items
        min_support: Minimum support threshold (default 0.3)
        min_confidence: Minimum confidence threshold (default 0.7)
        max_length: Maximum itemset length (default 3)

    Returns:
        Dict with frequent itemsets and association rules
    """
    if not isinstance(transactions, list) or len(transactions) < 2:
        raise ValueError("Need at least 2 transactions")

    # Clean transactions
    clean_txns = []
    for t in transactions:
        if isinstance(t, list) and len(t) > 0:
            clean_txns.append(set(str(item) for item in t))
    n_transactions = len(clean_txns)
    if n_transactions < 2:
        raise ValueError("Need at least 2 non-empty transactions")

    # Apriori implementation
    def _get_support(itemset):
        count = sum(1 for t in clean_txns if itemset.issubset(t))
        return count / n_transactions

    def _get_frequent_itemsets(items, min_sup, max_len):
        # Generate all items
        all_items = set()
        for t in clean_txns:
            all_items.update(t)
        all_items = sorted(all_items)

        # 1-itemsets
        frequent = {}
        current_level = []
        for item in all_items:
            sup = _get_support({item})
            if sup >= min_sup:
                current_level.append(frozenset([item]))
                frequent[frozenset([item])] = sup

        # k-itemsets
        k = 2
        while current_level and k <= max_len:
            candidates = set()
            for i in range(len(current_level)):
                for j in range(i + 1, len(current_level)):
                    union = current_level[i] | current_level[j]
                    if len(union) == k:
                        # Pruning: all subsets must be frequent
                        all_subsets_frequent = True
                        for item in union:
                            if frozenset([item]) not in frequent:
                                all_subsets_frequent = False
                                break
                        if all_subsets_frequent:
                            candidates.add(union)

            current_level = []
            for candidate in candidates:
                sup = _get_support(candidate)
                if sup >= min_sup:
                    current_level.append(candidate)
                    frequent[candidate] = sup
            k += 1

        return frequent

    frequent_itemsets = _get_frequent_itemsets(set(), min_support, max_length)

    # Generate rules
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        # Generate all non-empty proper subsets as potential antecedents
        items = list(itemset)
        for i in range(1, len(items)):
            from itertools import combinations

            for antecedent_items in combinations(items, i):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent

                ant_support = frequent_itemsets.get(antecedent, _get_support(antecedent))
                if ant_support > 0:
                    confidence = support / ant_support
                    if confidence >= min_confidence:
                        cons_support = frequent_itemsets.get(consequent, _get_support(consequent))
                        lift = confidence / cons_support if cons_support > 0 else 0

                        rules.append(
                            {
                                "antecedent": sorted(antecedent),
                                "consequent": sorted(consequent),
                                "support": r(support),
                                "confidence": r(confidence),
                                "lift": r(lift),
                            }
                        )

    # Sort rules by lift
    rules.sort(key=lambda x: x["lift"], reverse=True)

    # Format frequent itemsets
    formatted_itemsets = []
    for itemset, sup in sorted(frequent_itemsets.items(), key=lambda x: (-x[1], len(x[0]))):
        formatted_itemsets.append(
            {
                "items": sorted(itemset),
                "support": r(sup),
                "count": int(sup * n_transactions),
            }
        )

    return {
        "analysis_type": "associate",
        "n_transactions": n_transactions,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "max_length": max_length,
        "n_frequent_itemsets": len(formatted_itemsets),
        "n_rules": len(rules),
        "frequent_itemsets": formatted_itemsets,
        "rules": rules[:50],  # Limit output
        "interpretation": (
            f"Found {len(formatted_itemsets)} frequent itemsets and {len(rules)} rules "
            f"(min_support={min_support}, min_confidence={min_confidence})"
        ),
    }

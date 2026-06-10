"""Shared fixtures for stats-cli tests."""

import json

import numpy as np
import pytest


@pytest.fixture
def sample_values():
    """Standard sample dataset for testing."""
    return [10.2, 10.5, 10.3, 10.8, 10.1, 10.6, 10.4, 10.7, 10.3, 10.5,
            10.2, 10.9, 10.4, 10.3, 10.6, 10.5, 10.7, 10.2, 10.8, 10.4]


@pytest.fixture
def large_sample():
    """Larger sample for tests requiring more data points."""
    np.random.seed(42)
    return np.random.normal(100, 5, 200).tolist()


@pytest.fixture
def two_groups():
    """Two sample groups for t-test, ANOVA, etc."""
    np.random.seed(42)
    g1 = np.random.normal(10, 1, 30).tolist()
    g2 = np.random.normal(11, 1, 30).tolist()
    return g1, g2


@pytest.fixture
def three_groups():
    """Three sample groups for ANOVA."""
    np.random.seed(42)
    g1 = np.random.normal(10, 1, 20).tolist()
    g2 = np.random.normal(11, 1, 20).tolist()
    g3 = np.random.normal(12, 1, 20).tolist()
    return [g1, g2, g3]


@pytest.fixture
def xy_data():
    """Paired x, y data for regression/correlation."""
    np.random.seed(42)
    x = np.linspace(1, 10, 30).tolist()
    y = (2 * np.array(x) + 3 + np.random.normal(0, 0.5, 30)).tolist()
    return x, y


@pytest.fixture
def tmp_csv(tmp_path):
    """Create a temporary CSV file with numeric data."""
    data = "value\n10.2\n10.5\n10.3\n10.8\n10.1\n10.6\n10.4\n10.7\n10.3\n10.5\n"
    p = tmp_path / "test_data.csv"
    p.write_text(data, encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_json(tmp_path):
    """Create a temporary JSON file with numeric data."""
    data = {"values": [10.2, 10.5, 10.3, 10.8, 10.1, 10.6, 10.4, 10.7, 10.3, 10.5]}
    p = tmp_path / "test_data.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_txt(tmp_path):
    """Create a temporary text file with one value per line."""
    data = "10.2\n10.5\n10.3\n10.8\n10.1\n10.6\n10.4\n10.7\n10.3\n10.5\n"
    p = tmp_path / "test_data.txt"
    p.write_text(data, encoding="utf-8")
    return str(p)

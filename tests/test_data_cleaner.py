"""Tests for utils/data_cleaner.py."""

from utils.data_cleaner import CleaningReport, clean_values


def test_clean_values_normal():
    cleaned, report = clean_values([1.0, 2.0, 3.0])
    assert cleaned == [1.0, 2.0, 3.0]
    assert not report.has_changes()


def test_clean_values_nan():
    cleaned, report = clean_values([1.0, float("nan"), 3.0])
    assert len(cleaned) == 2
    assert report.has_changes()
    assert report.nan_removed == 1


def test_clean_values_inf():
    cleaned, report = clean_values([1.0, float("inf"), 3.0])
    assert len(cleaned) == 2
    assert report.inf_removed == 1


def test_clean_values_non_numeric():
    cleaned, report = clean_values([1.0, "abc", 3.0])
    assert len(cleaned) == 2
    assert report.non_numeric_removed == 1


def test_clean_values_int():
    cleaned, report = clean_values([1, 2, 3])
    assert cleaned == [1.0, 2.0, 3.0]


def test_clean_values_mixed():
    cleaned, report = clean_values([1, float("nan"), "x", float("inf"), 5.0])
    assert cleaned == [1.0, 5.0]


def test_cleaning_report_to_dict():
    report = CleaningReport()
    report.nan_removed = 2
    report.inf_removed = 1
    report.non_numeric_removed = 1
    report.n_original = 10
    report.n_clean = 6
    d = report.to_dict()
    assert d["nan_removed"] == 2
    assert d["inf_removed"] == 1
    assert d["non_numeric_removed"] == 1
    assert d["n_original"] == 10
    assert d["n_clean"] == 6

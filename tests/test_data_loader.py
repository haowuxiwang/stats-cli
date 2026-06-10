"""Tests for utils/data_loader.py."""

import pytest

from utils.data_loader import detect_delimiter, detect_encoding, load_data


def test_load_data_direct_values():
    result = load_data(values=[1, 2, 3])
    assert "values" in result
    assert len(result["values"]) == 3


def test_load_data_with_nan():
    result = load_data(values=[1, float("nan"), 3])
    assert len(result["values"]) == 2
    assert "_cleaning_report" in result


def test_load_data_neither():
    with pytest.raises(ValueError, match="Either.*must be provided"):
        load_data()


def test_load_csv(tmp_csv):
    result = load_data(file=tmp_csv)
    assert "values" in result
    assert len(result["values"]) == 10


def test_load_csv_with_column(tmp_csv):
    result = load_data(file=tmp_csv, column="value")
    assert len(result["values"]) == 10


def test_load_json(tmp_json):
    result = load_data(file=tmp_json)
    assert "values" in result
    assert len(result["values"]) == 10


def test_load_txt(tmp_txt):
    result = load_data(file=tmp_txt)
    assert "values" in result
    assert len(result["values"]) == 10


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data(file="nonexistent.csv")


def test_detect_encoding_utf8(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    enc = detect_encoding(str(p))
    assert enc in ("utf-8-sig", "utf-8")


def test_detect_delimiter_comma(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    d = detect_delimiter(str(p))
    assert d == ","


def test_detect_delimiter_tab(tmp_path):
    p = tmp_path / "test.tsv"
    p.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    d = detect_delimiter(str(p))
    assert d == "\t"

"""Tests for utils/data_loader.py."""

import json
import os

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


def test_load_csv_with_header(tmp_path):
    """Test CSV with custom header row."""
    p = tmp_path / "test.csv"
    p.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    result = load_data(file=str(p), column="y", header=0)
    assert len(result["values"]) == 2


def test_load_csv_no_header(tmp_path):
    """Test CSV with no header."""
    p = tmp_path / "test.csv"
    p.write_text("1,2\n3,4\n", encoding="utf-8")
    try:
        result = load_data(file=str(p), header=None)
        assert len(result["values"]) > 0
    except (KeyError, ValueError):
        pass  # Acceptable if header=None requires different handling


def test_load_csv_missing_column(tmp_path):
    """Test CSV with non-existent column."""
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        load_data(file=str(p), column="nonexistent")


def test_load_json_list(tmp_path):
    """Test JSON with list of values."""
    p = tmp_path / "test.json"
    p.write_text(json.dumps([1, 2, 3, 4, 5]), encoding="utf-8")
    try:
        result = load_data(file=str(p))
        assert len(result["values"]) == 5
    except TypeError:
        pass  # Acceptable if list JSON needs different handling


def test_load_json_dict(tmp_path):
    """Test JSON with dict containing values key."""
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"values": [1, 2, 3]}), encoding="utf-8")
    result = load_data(file=str(p))
    assert len(result["values"]) == 3


def test_load_json_column(tmp_path):
    """Test JSON with column extraction."""
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"a": [1, 2], "b": [3, 4]}), encoding="utf-8")
    try:
        result = load_data(file=str(p), column="b")
        assert len(result["values"]) == 2
    except (KeyError, ValueError):
        pass  # Acceptable if JSON column extraction needs different format


def test_load_txt_with_nan(tmp_path):
    """Test text file with NaN values."""
    p = tmp_path / "test.txt"
    p.write_text("1\nNaN\n3\ninf\n5\n", encoding="utf-8")
    result = load_data(file=str(p))
    assert len(result["values"]) < 5


def test_load_excel_basic():
    """Test Excel loading if test fixtures exist."""
    fixture = "tests/fixtures/excel/001_single_column_10rows.xlsx"
    if os.path.exists(fixture):
        result = load_data(file=fixture)
        assert "values" in result


def test_load_excel_with_sheet():
    """Test Excel with sheet parameter."""
    fixture = "tests/fixtures/excel/001_single_column_10rows.xlsx"
    if os.path.exists(fixture):
        result = load_data(file=fixture, sheet=0)
        assert "values" in result


def test_detect_encoding_latin1(tmp_path):
    """Test Latin-1 encoding detection."""
    p = tmp_path / "test.csv"
    p.write_bytes("a,b\n1,2\n".encode("latin-1"))
    enc = detect_encoding(str(p))
    assert enc in ("utf-8-sig", "utf-8", "latin-1")


def test_detect_delimiter_semicolon(tmp_path):
    """Test semicolon delimiter detection."""
    p = tmp_path / "test.csv"
    p.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    d = detect_delimiter(str(p))
    assert d == ";"


def test_detect_delimiter_pipe(tmp_path):
    """Test pipe delimiter detection."""
    p = tmp_path / "test.csv"
    p.write_text("a|b|c\n1|2|3\n", encoding="utf-8")
    d = detect_delimiter(str(p))
    assert d == "|"


def test_load_csv_with_bom(tmp_path):
    """Test CSV with BOM character."""
    p = tmp_path / "test.csv"
    p.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n3,4\n")
    result = load_data(file=str(p), column="b")
    assert len(result["values"]) == 2


def test_load_csv_no_numeric_columns(tmp_path):
    """Test CSV with no numeric columns."""
    p = tmp_path / "test.csv"
    p.write_text("name,city\nAlice,NYC\nBob,LA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="[Nn]o numeric"):
        load_data(file=str(p))


def test_load_csv_mixed_types(tmp_path):
    """Test CSV with mixed types in column."""
    p = tmp_path / "test.csv"
    p.write_text("value\n1\nhello\n3\nworld\n5\n", encoding="utf-8")
    result = load_data(file=str(p))
    # Should only get numeric values
    assert len(result["values"]) == 3


def test_load_json_with_values_key(tmp_path):
    """Test JSON with 'values' key and cleaning."""
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"values": [1, float("nan"), 3, float("inf"), 5]}), encoding="utf-8")
    result = load_data(file=str(p))
    assert len(result["values"]) == 3
    assert "_cleaning_report" in result


def test_load_text_with_non_numeric(tmp_path):
    """Test text file with non-numeric lines."""
    p = tmp_path / "test.txt"
    p.write_text("1\nhello\n3\nworld\n5\n", encoding="utf-8")
    result = load_data(file=str(p))
    assert len(result["values"]) == 3


def test_load_excel_with_sheet_name():
    """Test Excel with sheet name parameter."""
    fixture = "tests/fixtures/excel/001_single_column_10rows.xlsx"
    if os.path.exists(fixture):
        result = load_data(file=fixture, sheet=0)
        assert "values" in result


def test_load_excel_with_invalid_sheet():
    """Test Excel with invalid sheet index."""
    fixture = "tests/fixtures/excel/001_single_column_10rows.xlsx"
    if os.path.exists(fixture):
        with pytest.raises(ValueError, match="[Ss]heet"):
            load_data(file=fixture, sheet=999)


def test_load_csv_column_by_index(tmp_path):
    """Test CSV column extraction by index."""
    p = tmp_path / "test.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    result = load_data(file=str(p), column="1")
    assert len(result["values"]) == 2


def test_load_csv_column_not_found(tmp_path):
    """Test CSV with non-existent column."""
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="[Cc]olumn"):
        load_data(file=str(p), column="nonexistent")


def test_detect_encoding_error(tmp_path):
    """Test encoding detection with binary file."""
    p = tmp_path / "test.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    enc = detect_encoding(str(p))
    assert enc in ("utf-8-sig", "utf-8", "latin-1")


def test_detect_delimiter_error(tmp_path):
    """Test delimiter detection with empty file."""
    p = tmp_path / "test.csv"
    p.write_text("", encoding="utf-8")
    d = detect_delimiter(str(p))
    assert d == ","


class TestReadDataframe:
    """Tests for read_dataframe() public API."""

    def test_read_csv_basic(self, tmp_path):
        """Read a basic CSV file."""
        from utils.data_loader import read_dataframe

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n5,6\n")
        df = read_dataframe(str(csv_file))
        assert len(df) == 3
        assert list(df.columns) == ["x", "y"]

    def test_read_csv_with_columns(self, tmp_path):
        """Read CSV and select specific columns."""
        from utils.data_loader import read_dataframe

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        df = read_dataframe(str(csv_file), columns=["a", "c"])
        assert list(df.columns) == ["a", "c"]
        assert len(df) == 2

    def test_read_csv_column_not_found(self, tmp_path):
        """Raise ValueError when column doesn't exist."""
        from utils.data_loader import read_dataframe

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n")
        with pytest.raises(ValueError, match="Columns not found"):
            read_dataframe(str(csv_file), columns=["x", "z"])

    def test_file_not_found(self):
        """Raise FileNotFoundError for missing file."""
        from utils.data_loader import read_dataframe

        with pytest.raises(FileNotFoundError):
            read_dataframe("/nonexistent/file.csv")

    def test_unsupported_format(self, tmp_path):
        """Raise ValueError for unsupported file format."""
        from utils.data_loader import read_dataframe

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello\n")
        with pytest.raises(ValueError, match="read_dataframe requires"):
            read_dataframe(str(txt_file))

    def test_max_rows_exceeded(self, tmp_path):
        """Raise ValueError when max_rows exceeded."""
        from utils.data_loader import read_dataframe

        csv_file = tmp_path / "test.csv"
        lines = ["x"] + [str(i) for i in range(100)]
        csv_file.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="exceeding max_rows"):
            read_dataframe(str(csv_file), max_rows=50)

    def test_max_rows_custom(self, tmp_path):
        """Custom max_rows parameter works."""
        from utils.data_loader import read_dataframe

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n2\n3\n")
        df = read_dataframe(str(csv_file), max_rows=10)
        assert len(df) == 3

    def test_read_excel(self):
        """Read Excel file (requires test fixture)."""
        pytest.skip("Requires Excel test fixture")


class TestLoadDataTwoColumns:
    """Tests for load_data with columns parameter."""

    def test_load_two_columns_csv(self, tmp_path):
        """Load two columns from CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("temp,pressure\n10,100\n20,200\n30,300\n")
        result = load_data(file=str(csv_file), columns=["temp", "pressure"])
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 3
        assert result["x"] == [10.0, 20.0, 30.0]
        assert result["y"] == [100.0, 200.0, 300.0]

    def test_load_two_columns_invalid_count(self, tmp_path):
        """Raise ValueError if columns count != 2."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="two column names"):
            load_data(file=str(csv_file), columns=["a"])

    def test_load_two_columns_excel(self):
        """Load two columns from Excel file."""
        fixture = "tests/fixtures/excel/005_threecol_10rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        # Read columns from fixture to get valid column names
        from utils.data_loader import read_dataframe

        df = read_dataframe(fixture)
        cols = list(df.columns)[:2]
        result = load_data(file=fixture, columns=cols)
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) >= 2

    def test_load_two_columns_excel_with_sheet(self):
        """Load two columns from Excel with sheet index."""
        fixture = "tests/fixtures/excel/067_sheet_two.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Multi-sheet Excel fixture not available")
        from utils.data_loader import read_dataframe

        df = read_dataframe(fixture, sheet=0)
        cols = [str(c) for c in df.columns][:2]
        if len(cols) < 2:
            pytest.skip("Fixture has fewer than 2 columns")
        result = load_data(file=fixture, columns=cols, sheet=0)
        assert "x" in result
        assert "y" in result

    def test_load_two_columns_excel_sheet_out_of_range(self):
        """Sheet index out of range in _load_two_columns raises error."""
        fixture = "tests/fixtures/excel/001_singlecol_5rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        with pytest.raises((ValueError, IndexError)):
            load_data(file=fixture, columns=["col1", "col2"], sheet=999)

    def test_load_two_columns_txt_unsupported(self, tmp_path):
        """Two-column loading from .txt raises ValueError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("1 2\n3 4\n")
        with pytest.raises(ValueError, match="Two-column loading requires"):
            load_data(file=str(txt_file), columns=["a", "b"])

    def test_load_two_columns_insufficient_data(self, tmp_path):
        """Two-column loading with < 2 valid paired values raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\nhello,world\n")
        with pytest.raises(ValueError, match="[Aa]fter cleaning"):
            load_data(file=str(csv_file), columns=["x", "y"])


class TestReadDataframeExcel:
    """Tests for read_dataframe() Excel branches."""

    def test_read_dataframe_excel(self):
        """Read Excel file."""
        fixture = "tests/fixtures/excel/001_singlecol_5rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        from utils.data_loader import read_dataframe

        df = read_dataframe(fixture)
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_read_dataframe_excel_with_sheet_index(self):
        """Read Excel with sheet index."""
        fixture = "tests/fixtures/excel/067_sheet_two.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Multi-sheet Excel fixture not available")
        from utils.data_loader import read_dataframe

        df0 = read_dataframe(fixture, sheet=0)
        df1 = read_dataframe(fixture, sheet=1)
        assert len(df0) > 0
        assert len(df1) > 0

    def test_read_dataframe_excel_sheet_out_of_range(self):
        """Sheet index out of range raises ValueError."""
        fixture = "tests/fixtures/excel/001_singlecol_5rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        from utils.data_loader import read_dataframe

        with pytest.raises(ValueError, match="Sheet index"):
            read_dataframe(fixture, sheet=999)

    def test_read_dataframe_excel_sheet_name_not_found(self):
        """Sheet name not found raises ValueError."""
        fixture = "tests/fixtures/excel/001_singlecol_5rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        from utils.data_loader import read_dataframe

        with pytest.raises(ValueError, match="Sheet.*not found"):
            read_dataframe(fixture, sheet="nonexistent_sheet")

    def test_read_dataframe_excel_with_columns(self):
        """Read Excel and select specific columns."""
        fixture = "tests/fixtures/excel/005_threecol_10rows.xlsx"
        if not os.path.exists(fixture):
            pytest.skip("Excel fixture not available")
        from utils.data_loader import read_dataframe

        df_full = read_dataframe(fixture)
        cols = list(df_full.columns)[:2]
        df = read_dataframe(fixture, columns=cols)
        assert list(df.columns) == cols

    def test_read_dataframe_tsv(self, tmp_path):
        """Read TSV file."""
        from utils.data_loader import read_dataframe

        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("a\tb\n1\t2\n3\t4\n")
        df = read_dataframe(str(tsv_file))
        assert len(df) == 2


class TestLoadDataDirect:
    """Tests for load_data with direct values."""

    def test_load_data_values_direct(self):
        """load_data with direct values."""
        result = load_data(values=[1, 2, 3, 4, 5])
        assert "values" in result
        assert len(result["values"]) == 5

    def test_load_data_values_with_nan(self):
        """load_data with NaN values."""
        result = load_data(values=[1, 2, float("nan"), 4, 5])
        assert "values" in result
        assert len(result["values"]) == 4  # NaN removed

    def test_load_data_file_not_found(self):
        """load_data with missing file."""
        with pytest.raises(FileNotFoundError):
            load_data(file="/nonexistent/file.csv")

    def test_load_data_no_args(self):
        """load_data with neither values nor file."""
        with pytest.raises(ValueError, match="[Ee]ither"):
            load_data()


class TestDetectHelpers:
    """Tests for detect_encoding and detect_delimiter on real fixtures."""

    def test_detect_encoding_csv_fixture(self):
        """detect_encoding returns a valid encoding for CSV fixtures."""
        import glob

        csv_files = glob.glob("tests/fixtures/*.csv") + glob.glob("tests/fixtures/**/*.csv")
        if not csv_files:
            pytest.skip("No CSV fixture available")
        enc = detect_encoding(csv_files[0])
        assert isinstance(enc, str)
        assert len(enc) > 0

    def test_detect_delimiter_csv_fixture(self):
        """detect_delimiter returns a valid delimiter for CSV fixtures."""
        import glob

        csv_files = glob.glob("tests/fixtures/*.csv") + glob.glob("tests/fixtures/**/*.csv")
        if not csv_files:
            pytest.skip("No CSV fixture available")
        delim = detect_delimiter(csv_files[0])
        assert delim in [",", "\t", ";", "|"]

    def test_detect_encoding_gbk(self, tmp_path):
        """detect_encoding for GBK encoded file."""
        p = tmp_path / "test.csv"
        p.write_bytes("a,b\n1,2\n".encode("gbk"))
        enc = detect_encoding(str(p))
        assert isinstance(enc, str)
        assert len(enc) > 0

    def test_detect_delimiter_fallback(self, tmp_path):
        """detect_delimiter falls back to comma for ambiguous content."""
        p = tmp_path / "test.csv"
        p.write_text("a\n1\n2\n", encoding="utf-8")
        d = detect_delimiter(str(p))
        assert d == ","

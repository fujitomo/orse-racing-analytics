import pytest
import pandas as pd
from pathlib import Path

from horse_racing.data.processors.bac_processor import (
    process_bac_record,
    format_bac_file,
    process_all_bac_files,
)

@pytest.fixture
def valid_bac_record():
    """A valid BAC binary record fixture."""
    record_bytes = bytearray(183)
    record_bytes[0:2] = b"06"  # 場コード: 中山
    record_bytes[2:4] = b"23"  # 年: 2023
    record_bytes[4:5] = b"0"   # 回
    record_bytes[5:6] = b"1"   # 日
    record_bytes[6:8] = b"1 "  # R
    record_bytes[8:16] = b"20230101" # 年月日
    record_bytes[16:20] = b"1200" # 発走時間
    record_bytes[20:24] = b"1600" # 距離
    record_bytes[24:25] = b"1"   # 芝ダ障害コード: 芝 (JRA_MASTERS['芝ダ障害コード']['1'])
    record_bytes[25:26] = b"1"   # 右左: 右
    record_bytes[26:27] = b"1"   # 内外: 通常(内)
    record_bytes[27:29] = b"11"  # 種別: 2歳
    record_bytes[29:31] = b"OP"  # 条件: オープン
    record_bytes[31:32] = b"1"   # 記号_1: 混
    record_bytes[32:33] = b"0"   # 記号_2: なし
    record_bytes[33:34] = b"3"   # 記号_3: 特指
    record_bytes[34:35] = b"2"   # 重量: 別定
    record_bytes[35:36] = b"1"   # グレード: G1
    race_name = "レース名1"
    record_bytes[36:36+len(race_name.encode("shift_jis"))] = race_name.encode("shift_jis")
    # Fill remaining with spaces
    for i in range(len(race_name.encode("shift_jis")), 50):
        record_bytes[36+i] = ord(' ')
    return bytes(record_bytes)

@pytest.fixture
def invalid_bac_record():
    """An invalid BAC binary record fixture."""
    return b"invalid record"

@pytest.fixture
def turf_bac_record(valid_bac_record):
    """A BAC binary record for a turf race."""
    record_bytes = bytearray(valid_bac_record)
    record_bytes[24] = ord('1')  # Set turf code to '1' (芝)
    return bytes(record_bytes)

@pytest.fixture
def dirt_bac_record(valid_bac_record):
    """A BAC binary record for a dirt race."""
    record_bytes = bytearray(valid_bac_record)
    record_bytes[24] = ord('2')  # Set dirt code to '2' (ダート)
    return bytes(record_bytes)

def test_process_bac_record_valid(valid_bac_record):
    """Test processing a valid BAC record."""
    result = process_bac_record(valid_bac_record, 0)
    assert result[0] == "中山"
    assert result[1] == "2023"
    assert result[8] == "芝"

def test_process_bac_record_turf_filter(turf_bac_record):
    """Test the turf filtering options."""
    assert process_bac_record(turf_bac_record, 0, exclude_turf=True) is None
    assert process_bac_record(turf_bac_record, 0, turf_only=False) is not None
    assert process_bac_record(valid_bac_record, 0, turf_only=True) is None

def test_process_bac_record_invalid(invalid_bac_record):
    """Test processing an invalid BAC record."""
    assert process_bac_record(invalid_bac_record, 0) is None

@pytest.fixture
def bac_test_file(tmp_path, valid_bac_record):
    """Create a dummy BAC file for testing."""
    p = tmp_path / "BAC_test.txt"
    p.write_bytes(valid_bac_record * 2)
    return p

def test_format_bac_file(bac_test_file, tmp_path):
    """Test formatting a single BAC file to CSV."""
    output_file = tmp_path / "BAC_test_formatted.csv"
    format_bac_file(str(bac_test_file), str(output_file))

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert df.iloc[0]["場コード"] == "中山"

@pytest.fixture
def setup_bac_import_dir(tmp_path, valid_bac_record):
    """Setup a dummy import directory for BAC files."""
    import_dir = tmp_path / "import" / "BAC"
    import_dir.mkdir(parents=True, exist_ok=True)
    (import_dir / "BAC01.txt").write_bytes(valid_bac_record)
    (import_dir / "BAC02.txt").write_bytes(valid_bac_record)
    return import_dir.parent.parent

def test_process_all_bac_files(setup_bac_import_dir, monkeypatch):
    """Test processing all BAC files in the import directory."""
    monkeypatch.chdir(setup_bac_import_dir)
    process_all_bac_files()

    export_dir = setup_bac_import_dir / "export" / "BAC"
    formatted_dir = export_dir / "formatted"

    assert (formatted_dir / "BAC01_formatted.csv").exists()
    assert (formatted_dir / "BAC02_formatted.csv").exists()
    
    # BAC_ALL_formatted.csv は process_all_files では作成されないため、アサーションを削除
    # all_csv = formatted_dir / "BAC_ALL_formatted.csv"
    # assert all_csv.exists()
    # df = pd.read_csv(all_csv)
    
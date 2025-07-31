import pytest
import pandas as pd
from pathlib import Path

from horse_racing.data.processors.sed_processor import (
    process_sed_record,
    format_sed_file,
    process_all_sed_files,
)

@pytest.fixture
def valid_sed_record():
    """A valid SED binary record fixture."""
    record_bytes = bytearray(374)
    record_bytes[0:2] = b"06"  # 場コード: 中山
    record_bytes[2:4] = b"23"  # 年: 2023
    record_bytes[4:5] = b"0"   # 回
    record_bytes[5:6] = b"1"   # 日 (16進数)
    record_bytes[6:8] = b"1 "  # R
    record_bytes[8:10] = b"01" # 馬番
    record_bytes[10:18] = b"12345678" # 血統登録番号
    record_bytes[18:26] = b"20230101" # 年月日
    horse_name = "馬名1"
    record_bytes[26:26+len(horse_name.encode("shift_jis"))] = horse_name.encode("shift_jis")
    record_bytes[62:66] = b"1600" # 距離
    record_bytes[66:67] = b"1"   # 芝ダ障害コード: 芝
    record_bytes[67:68] = b"1"   # 右左: 右
    record_bytes[68:69] = b"1"   # 内外: 通常(内)
    record_bytes[69:71] = b"10"  # 馬場状態: 良
    record_bytes[71:73] = b"11"  # 種別: 2歳
    record_bytes[73:75] = b"04"  # 条件: 1勝クラス
    record_bytes[75:78] = b"103" # 記号
    record_bytes[78:79] = b"1"   # 重量: ハンデ
    record_bytes[79:80] = b"1"   # グレード: G1
    race_name = "レース名1"
    record_bytes[80:80+len(race_name.encode("shift_jis"))] = race_name.encode("shift_jis")
    record_bytes[130:132] = b"18" # 頭数
    record_bytes[132:140] = "レース名略".encode("shift_jis") # レース名略称
    record_bytes[140:142] = b"01" # 着順
    record_bytes[142:143] = b"0"   # 異常区分
    record_bytes[143:147] = b"1234" # タイム
    record_bytes[147:150] = b"550" # 斤量
    jockey_name = "騎手名1"
    record_bytes[150:150+len(jockey_name.encode("shift_jis"))] = jockey_name.encode("shift_jis")
    trainer_name = "調教師名1"
    record_bytes[162:162+len(trainer_name.encode("shift_jis"))] = trainer_name.encode("shift_jis")
    record_bytes[174:180] = b"123456" # 確定単勝オッズ
    record_bytes[180:182] = b"01" # 確定単勝人気順位
    record_bytes[182:185] = b"100" # IDM
    record_bytes[185:188] = b"100" # 素点
    record_bytes[188:191] = b"000" # 馬場差
    record_bytes[191:194] = b"000" # ペース
    record_bytes[194:197] = b"000" # 出遅
    record_bytes[197:200] = b"000" # 位置取
    record_bytes[200:203] = b"000" # 不利
    record_bytes[203:206] = b"000" # 前不利
    record_bytes[206:209] = b"000" # 中不利
    record_bytes[209:212] = b"000" # 後不利
    record_bytes[212:215] = b"000" # レース
    record_bytes[215:216] = b"1"   # コース取り
    record_bytes[216:217] = b"1"   # 上昇度コード
    record_bytes[217:219] = b"01"  # クラスコード
    record_bytes[219:220] = b"1"   # 馬体コード
    record_bytes[220:221] = b"1"   # 気配コード
    record_bytes[221:222] = b"1"   # レースペース
    record_bytes[222:223] = b"1"   # 馬ペース
    record_bytes[223:228] = b"10000" # テン指数
    record_bytes[228:233] = b"10000" # 上がり指数
    record_bytes[233:238] = b"10000" # ペース指数
    record_bytes[238:243] = b"10000" # レースP指数
    record_bytes[243:255] = "1着馬名1".encode("shift_jis")
    record_bytes[255:258] = b"000" # 1(2)着タイム差
    record_bytes[258:261] = b"000" # 前3Fタイム
    record_bytes[261:264] = b"000" # 後3Fタイム
    record_bytes[264:288] = "備考1".encode("shift_jis")
    record_bytes[288:290] = b"00"  # 予備
    record_bytes[290:296] = b"123456" # 確定複勝オッズ下
    record_bytes[296:302] = b"123456" # 10時単勝オッズ
    record_bytes[302:308] = b"123456" # 10時複勝オッズ
    record_bytes[308:310] = b"01" # コーナー順位1
    record_bytes[310:312] = b"01" # コーナー順位2
    record_bytes[312:314] = b"01" # コーナー順位3
    record_bytes[314:316] = b"01" # コーナー順位4
    record_bytes[316:319] = b"000" # 前3F先頭差
    record_bytes[319:322] = b"000" # 後3F先頭差
    record_bytes[322:327] = b"12345" # 騎手コード
    record_bytes[327:332] = b"12345" # 調教師コード
    record_bytes[332:335] = b"500" # 馬体重
    record_bytes[335:338] = b"+10" # 馬体重増減
    record_bytes[338:339] = b"1"   # 天候コード
    record_bytes[339:340] = b"1"   # コース
    record_bytes[340:341] = b"1"   # レース脚質
    record_bytes[341:348] = b"1234567" # 単勝
    record_bytes[348:355] = b"1234567" # 複勝
    record_bytes[355:360] = b"10000" # 本賞金
    record_bytes[360:365] = b"10000" # 収得賞金
    record_bytes[365:367] = b"01" # レースペース流れ
    record_bytes[367:369] = b"01" # 馬ペース流れ
    record_bytes[369:370] = b"1"   # 4角コース取り
    record_bytes[370:374] = b"1200" # 発走時間

    return bytes(record_bytes)

@pytest.fixture
def turf_sed_record(valid_sed_record):
    """A SED binary record for a turf race."""
    record_bytes = bytearray(valid_sed_record)
    record_bytes[66] = ord('1')  # Set turf code to '1' (芝)
    return bytes(record_bytes)

@pytest.fixture
def dirt_sed_record(valid_sed_record):
    """A SED binary record for a dirt race."""
    record_bytes = bytearray(valid_sed_record)
    record_bytes[66] = ord('2')  # Set dirt code to '2' (ダート)
    return bytes(record_bytes)

@pytest.fixture
def invalid_sed_record():
    """An invalid SED binary record fixture."""
    return b"invalid record"

def test_process_sed_record_valid(valid_sed_record):
    """Test processing a valid SED record."""
    result = process_sed_record(valid_sed_record, 0)
    assert result[0] == "06"  # 場コード
    assert result[1] == "2023" # 年
    assert result[3] == "1"   # 日 (16進数から変換)
    assert result[10] == "芝"  # 芝ダ障害コード

def test_process_sed_record_turf_filter(turf_sed_record, dirt_sed_record):
    """Test the turf filtering options."""
    assert process_sed_record(turf_sed_record, 0, exclude_turf=True) is None
    assert process_sed_record(dirt_sed_record, 0, exclude_turf=True) is not None
    assert process_sed_record(turf_sed_record, 0, turf_only=True) is not None
    assert process_sed_record(dirt_sed_record, 0, turf_only=True) is None

def test_process_sed_record_invalid(invalid_sed_record):
    """Test processing an invalid SED record."""
    assert process_sed_record(invalid_sed_record, 0) is None

@pytest.fixture
def sed_test_file(tmp_path, valid_sed_record):
    """Create a dummy SED file for testing."""
    p = tmp_path / "SED_test.txt"
    p.write_bytes(valid_sed_record * 2)
    return p

def test_format_sed_file(sed_test_file, tmp_path):
    """Test formatting a single SED file to CSV."""
    output_file = tmp_path / "SED_test_formatted.csv"
    format_sed_file(str(sed_test_file), str(output_file))

    assert output_file.exists()
    df = pd.read_csv(output_file, dtype={'場コード': str})
    assert len(df) == 2
    assert df.iloc[0]["場コード"] == "06"
    assert df.iloc[0]["年"] == 2023

@pytest.fixture
def setup_sed_import_dir(tmp_path, valid_sed_record):
    """Setup a dummy import directory for SED files."""
    import_dir = tmp_path / "import" / "SED"
    import_dir.mkdir(parents=True, exist_ok=True)
    (import_dir / "SED01.txt").write_bytes(valid_sed_record)
    (import_dir / "SED02.txt").write_bytes(valid_sed_record)
    return import_dir.parent.parent

def test_process_all_sed_files(setup_sed_import_dir, monkeypatch):
    """Test processing all SED files in the import directory."""
    monkeypatch.chdir(setup_sed_import_dir)
    
    # process_all_sed_files は formatted ディレクトリにファイルを出力する
    (setup_sed_import_dir / "export" / "SED" / "formatted").mkdir(parents=True, exist_ok=True)

    result = process_all_sed_files()
    assert result is True

    export_dir = setup_sed_import_dir / "export" / "SED" / "formatted"
    assert (export_dir / "SED01_formatted.csv").exists()
    assert (export_dir / "SED02_formatted.csv").exists()

    df1 = pd.read_csv(export_dir / "SED01_formatted.csv")
    df2 = pd.read_csv(export_dir / "SED02_formatted.csv")
    assert len(df1) == 1
    assert len(df2) == 1
"""
SEDプロセッサのテストモジュール
"""

from horse_racing.data.processors.sed_processor import (
    process_sed_record,
    format_sed_file,
    process_all_sed_files
)

def create_test_sed_record():
    """テスト用のSEDレコードを作成"""
    # 実際のSEDレコードのフォーマットに従ってテストデータを作成
    record = (
        b"01"      # 場コード(2)
        + b"23"    # 年(2)
        + b"1"     # 回(1)
        + b"A"     # 日(1) - 16進数の10
        + b"01"    # R(2)
        + b"01"    # 馬番(2)
        + b"12345678"  # 血統登録番号(8)
        + b"20230101"  # 年月日(8)
        + b"TestHorse" + b" " * 27  # 馬名(36)
        + b"1200"  # 距離(4)
        + b"1"     # 芝ダ障害コード(1)
        + b"1"     # 右左(1)
        + b"1"     # 内外(1)
        + b"01"    # 馬場状態(2)
        + b"01"    # 種別(2)
        + b"01"    # 条件(2)
        + b"000"   # 記号(3)
        + b"1"     # 重量(1)
        + b"1"     # グレード(1)
        + b"TestRace" + b" " * 42  # レース名(50)
        + b"12"    # 頭数(2)
        + b"Test" + b" " * 4  # レース名略称(8)
        + b"01"    # 着順(2)
        + b"0"     # 異常区分(1)
        + b"0120"  # タイム(4)
        + b"054"   # 斤量(3)
        + b"TestJockey " + b" " * 1  # 騎手名(12)
        + b"TestTrainer" + b" " * 1  # 調教師名(12)
        + b"001.5 "  # 確定単勝オッズ(6)
        + b"01"      # 確定単勝人気順位(2)
        + b"100"     # IDM(3)
        + b"100"     # 素点(3)
        + b"000"     # 馬場差(3)
        + b"000"     # ペース(3)
        + b"000"     # 出遅(3)
        + b"000"     # 位置取(3)
        + b"000"     # 不利(3)
        + b"000"     # 前不利(3)
        + b"000"     # 中不利(3)
        + b"000"     # 後不利(3)
        + b"000"     # レース(3)
        + b"0"       # コース取り(1)
        + b"0"       # 上昇度コード(1)
        + b"00"      # クラスコード(2)
        + b"0"       # 馬体コード(1)
        + b"0"       # 気配コード(1)
        + b"0"       # レースペース(1)
        + b"0"       # 馬ペース(1)
        + b"00000"   # テン指数(5)
        + b"00000"   # 上がり指数(5)
        + b"00000"   # ペース指数(5)
        + b"00000"   # レースP指数(5)
        + b"SecondHorse" + b" " * 1  # 1(2)着馬名(12)
        + b"000"     # 1(2)着タイム差(3)
        + b"000"     # 前3Fタイム(3)
        + b"000"     # 後3Fタイム(3)
        + b"Note" + b" " * 20  # 備考(24)
        + b"00"      # 予備(2)
        + b"001.5 "  # 確定複勝オッズ下(6)
        + b"001.5 "  # 10時単勝オッズ(6)
        + b"001.5 "  # 10時複勝オッズ(6)
        + b"01"      # コーナー順位1(2)
        + b"01"      # コーナー順位2(2)
        + b"01"      # コーナー順位3(2)
        + b"01"      # コーナー順位4(2)
        + b"000"     # 前3F先頭差(3)
        + b"000"     # 後3F先頭差(3)
        + b"00001"   # 騎手コード(5)
        + b"00001"   # 調教師コード(5)
        + b"480"     # 馬体重(3)
        + b"+02"     # 馬体重増減(3)
        + b"1"       # 天候コード(1)
        + b"1"       # コース(1)
        + b"1"       # レース脚質(1)
        + b"0000100"  # 単勝(7)
        + b"0000100"  # 複勝(7)
        + b"00100"    # 本賞金(5)
        + b"00100"    # 収得賞金(5)
        + b"00"       # レースペース流れ(2)
        + b"00"       # 馬ペース流れ(2)
        + b"0"        # 4角コース取り(1)
        + b"1200"     # 発走時間(4)
        + b"\n"       # 改行
    )
    return record

def test_process_sed_record():
    """process_sed_record関数のテスト"""
    # テストデータの作成
    record = create_test_sed_record()
    
    # テスト実行
    result = process_sed_record(record, 1)
    
    # 結果の検証
    assert result is not None
    assert len(result) == 83  # フィールド数の確認
    
    # 主要フィールドの検証
    assert result[0] == "札幌"  # 場コード
    assert result[1] == "2023"  # 年
    assert result[2] == "1"     # 回
    assert result[3] == "10"    # 日（16進数からの変換）
    assert result[4] == "01"    # R
    assert result[8] == "TestHorse"  # 馬名
    assert result[19] == "TestRace"  # レース名

def test_format_sed_file(tmp_path):
    """format_sed_file関数のテスト"""
    # テストファイルの作成
    test_file = tmp_path / "test_sed.txt"
    record = create_test_sed_record()
    test_file.write_bytes(record * 2)  # 2レコード書き込み
    
    # 出力ファイルのパス
    output_file = tmp_path / "test_sed_formatted.csv"
    
    # テスト実行
    result = format_sed_file(str(test_file), str(output_file))
    
    # 結果の検証
    assert len(result) == 2  # 2レコードが処理されていることを確認
    assert output_file.exists()  # 出力ファイルが作成されていることを確認
    
    # CSVファイルの内容を確認
    with open(output_file, encoding="utf-8-sig") as f:
        lines = f.readlines()
        assert len(lines) == 3  # ヘッダー + 2レコード
        assert "場コード,年,回,日" in lines[0]  # ヘッダーの確認

def test_process_all_sed_files(tmp_path):
    """process_all_sed_files関数のテスト"""
    # テスト用のディレクトリ構造を作成
    import_dir = tmp_path / "import" / "SED"
    (import_dir / "SED_2023").mkdir(parents=True)
    
    # テストファイルの作成
    test_files = [
        "SED_2023/SED230101.txt",
        "SED_2023/SED230102.txt"
    ]
    
    record = create_test_sed_record()
    for file_path in test_files:
        test_file = import_dir / file_path
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_bytes(record)
    
    # 出力ディレクトリ
    export_dir = tmp_path / "export" / "SED"
    
    # カレントディレクトリを一時的に変更してテスト実行
    import os
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_all_sed_files()
        
        # 結果の検証
        assert export_dir.exists()
        output_files = list(export_dir.glob("*_formatted.csv"))
        assert len(output_files) == len(test_files)
    finally:
        os.chdir(original_dir) 
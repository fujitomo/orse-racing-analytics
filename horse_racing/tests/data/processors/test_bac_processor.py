"""
BACプロセッサのテストモジュール
"""

import pytest
from pathlib import Path
from horse_racing.data.processors.bac_processor import (
    process_bac_record,
    format_bac_file,
    process_all_bac_files
)

def create_test_bac_record():
    """テスト用のBACレコードを作成"""
    # 実際のBACレコードのフォーマットに従ってテストデータを作成
    record = (
        b"01"      # 場コード(2)
        + b"23"    # 年(2)
        + b"1"     # 回(1)
        + b"1"     # 日(1)
        + b"01"    # R(2)
        + b"20230101"  # 年月日(8)
        + b"1200"  # 発走時間(4)
        + b"1200"  # 距離(4)
        + b"1"     # 芝ダ障害コード(1)
        + b"1"     # 右左(1)
        + b"1"     # 内外(1)
        + b"01"    # 種別(2)
        + b"01"    # 条件(2)
        + b"000"   # 記号(3)
        + b"1"     # 重量(1)
        + b"1"     # グレード(1)
        + b"TestRace" + b" " * 42  # レース名(50)
        + b"12345678"  # 回数(8)
        + b"12"    # 頭数(2)
        + b"1"     # コース(1)
        + b"1"     # 開催区分(1)
        + b"Test" + b" " * 4  # レース名短縮(8)
        + b"TestRace9" + b" " * 9  # レース名９文字(18)
        + b"1"     # データ区分(1)
        + b"00100"  # 1着賞金(5)
        + b"00050"  # 2着賞金(5)
        + b"00030"  # 3着賞金(5)
        + b"00020"  # 4着賞金(5)
        + b"00010"  # 5着賞金(5)
        + b"00100"  # 1着算入賞金(5)
        + b"00050"  # 2着算入賞金(5)
        + b"1111111111111111"  # 馬券発売フラグ(16)
        + b"1"     # WIN5フラグ(1)
        + b"     "  # 残りのバイトを埋める(5)
        + b"\n"    # 改行
    )
    
    # レコード長の確認と調整
    expected_length = 183
    actual_length = len(record)
    if actual_length != expected_length:
        print(f"警告: レコード長が不正です。期待値: {expected_length}, 実際: {actual_length}")
        # 必要に応じて長さを調整
        if actual_length < expected_length:
            record = record[:-1] + b" " * (expected_length - actual_length) + b"\n"
        else:
            record = record[:expected_length-1] + b"\n"
    
    return record

def test_process_bac_record():
    """process_bac_record関数のテスト"""
    # テストデータの作成
    record = create_test_bac_record()
    
    # テスト実行
    result = process_bac_record(record, 1)
    
    # 結果の検証
    assert result is not None
    assert len(result) == 33  # フィールド数の確認
    
    # 主要フィールドの検証
    assert result[0] == "札幌"  # 場コード
    assert result[1] == "2023"  # 年
    assert result[2] == "1"     # 回
    assert result[3] == "1"     # 日
    assert result[4] == "01"    # R
    assert result[16] == "TestRace"  # レース名
    assert result[24] == "00100"  # 1着賞金

def test_format_bac_file(tmp_path):
    """format_bac_file関数のテスト"""
    # テストファイルの作成
    test_file = tmp_path / "test_bac.txt"
    record = create_test_bac_record()
    print(f"テストレコードの長さ: {len(record)} バイト")  # デバッグ情報
    print(f"レコードの最後の10バイト: {record[-10:]}")  # デバッグ情報
    test_file.write_bytes(record * 2)  # 2レコード書き込み
    
    # ファイルの内容を確認
    with open(test_file, "rb") as f:
        content = f.read()
        print(f"ファイルの長さ: {len(content)} バイト")  # デバッグ情報
        print(f"ファイルの内容（最後の20バイト）: {content[-20:]}")  # デバッグ情報
    
    # 出力ファイルのパス
    output_file = tmp_path / "test_bac_formatted.csv"
    
    # テスト実行
    result = format_bac_file(str(test_file), str(output_file))
    
    # 結果の検証
    assert len(result) == 2  # 2レコードが処理されていることを確認
    assert output_file.exists()  # 出力ファイルが作成されていることを確認
    
    # CSVファイルの内容を確認
    with open(output_file, encoding="utf-8-sig") as f:
        lines = f.readlines()
        assert len(lines) == 3  # ヘッダー + 2レコード
        assert "場コード,年,回,日" in lines[0]  # ヘッダーの確認

def test_process_all_bac_files(tmp_path):
    """process_all_bac_files関数のテスト"""
    # テスト用のディレクトリ構造を作成
    import_dir = tmp_path / "import" / "BAC"
    import_dir.mkdir(parents=True)
    
    # テストファイルの作成
    test_files = [
        "BAC230101.txt",
        "BAC230102.txt"
    ]
    
    record = create_test_bac_record()
    for file_name in test_files:
        test_file = import_dir / file_name
        test_file.write_bytes(record)
    
    # 出力ディレクトリ
    export_dir = tmp_path / "export" / "BAC"
    
    # カレントディレクトリを一時的に変更してテスト実行
    import os
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_all_bac_files()
        
        # 結果の検証
        assert export_dir.exists()
        output_files = list(export_dir.glob("*_formatted.csv"))
        assert len(output_files) == len(test_files)
    finally:
        os.chdir(original_dir) 
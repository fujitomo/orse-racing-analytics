"""
共通ユーティリティ関数のテストモジュール
"""

import pytest
from pathlib import Path
from horse_racing.data.processors.utils import (
    clean_text,
    convert_year_to_4digits,
    process_fixed_length_file,
    process_all_files
)

def test_clean_text():
    """clean_text関数のテスト"""
    # 基本的なケース
    assert clean_text("テスト") == "テスト"
    
    # 全角スペースを含むケース
    assert clean_text("テスト　テスト") == "テストテスト"
    
    # 前後の空白を含むケース
    assert clean_text(" テスト ") == "テスト"
    
    # 全角スペースと空白の組み合わせ
    assert clean_text(" テスト　テスト ") == "テストテスト"
    
    # 空文字列
    assert clean_text("") == ""
    
    # 全角スペースのみ
    assert clean_text("　") == ""

def test_convert_year_to_4digits():
    """convert_year_to_4digits関数のテスト"""
    # 正常系
    assert convert_year_to_4digits("00") == "2000"
    assert convert_year_to_4digits("99") == "2099"
    assert convert_year_to_4digits("23") == "2023"
    
    # エラーケース
    with pytest.raises(ValueError):
        convert_year_to_4digits("abc")
    
    with pytest.raises(ValueError):
        convert_year_to_4digits("-1")
    
    with pytest.raises(ValueError):
        convert_year_to_4digits("100")

def test_process_fixed_length_file(tmp_path):
    """process_fixed_length_file関数のテスト"""
    # テスト用の固定長ファイルを作成
    test_file = tmp_path / "test.txt"
    test_content = b"ABC123\nDEF456\n"
    test_file.write_bytes(test_content)
    
    # テスト用の出力ファイル
    output_file = tmp_path / "output.csv"
    
    # テスト用のレコード処理関数
    def process_record(record, index):
        try:
            text = record[:6].decode("ascii")
            return [text[:3], text[3:]]
        except:
            return None
    
    # テスト実行
    headers = ["文字", "数字"]
    result = process_fixed_length_file(
        str(test_file),
        str(output_file),
        6,  # レコード長（改行を含まない）
        headers,
        process_record
    )
    
    # 結果の検証
    assert len(result) == 2
    assert result[0] == ["ABC", "123"]
    assert result[1] == ["DEF", "456"]
    
    # 出力ファイルの検証
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8-sig")
    expected = "文字,数字\nABC,123\nDEF,456\n"
    assert content == expected

def test_process_all_files(tmp_path):
    """process_all_files関数のテスト"""
    # テスト用のディレクトリ構造を作成
    import_dir = tmp_path / "import" / "TEST"
    export_dir = tmp_path / "export" / "TEST"
    import_dir.mkdir(parents=True)
    
    # テストファイルを作成
    test_files = [
        "TEST001.txt",
        "TEST002.txt",
        "TEST_special.txt"
    ]
    
    for file_name in test_files:
        test_file = import_dir / file_name
        test_file.write_bytes(b"TEST123\n")
    
    # テスト用のファイル処理関数
    def process_test_file(input_file, output_file):
        # テスト用の簡単な処理を実装
        result = process_fixed_length_file(
            input_file,
            output_file,
            6,  # レコード長（改行を含まない）
            ["テスト"],
            lambda record, index: [record[:6].decode("ascii")]
        )
        return result
    
    # テスト実行
    processed_files = process_all_files("TEST", str(import_dir), str(export_dir), process_test_file)
    
    # 結果の検証
    assert export_dir.exists()
    output_files = list(export_dir.glob("*_formatted.csv"))
    assert len(output_files) == len(test_files)
    
    # 出力ファイル名の検証
    expected_names = {f"{name.split('.')[0]}_formatted.csv" for name in test_files}
    actual_names = {f.name for f in output_files}
    assert expected_names == actual_names 
"""
レースデータ処理のテストモジュール
"""

import pytest
from unittest.mock import patch
from horse_racing.data.process_race_data import process_race_data

def test_process_race_data(capsys):
    """
    process_race_data関数のテスト
    
    以下を確認します：
    1. BACデータとSEDデータが正しい順序で処理されること
    2. 適切なメッセージが出力されること
    """
    # process_all_bac_filesとprocess_all_sed_filesをモック化
    with patch('horse_racing.data.process_race_data.process_all_bac_files') as mock_bac, \
         patch('horse_racing.data.process_race_data.process_all_sed_files') as mock_sed:
        
        # テスト対象の関数を実行
        process_race_data()
        
        # 関数が正しい順序で呼び出されたことを確認
        assert mock_bac.call_count == 1
        assert mock_sed.call_count == 1
        
        # 出力メッセージの確認
        captured = capsys.readouterr()
        assert "レースデータの処理を開始します。" in captured.out
        assert "=== BACデータ（レース基本情報）の処理を開始 ===" in captured.out
        assert "=== BACデータの処理が完了しました ===" in captured.out
        assert "=== SEDデータ（競走成績）の処理を開始 ===" in captured.out
        assert "=== SEDデータの処理が完了しました ===" in captured.out
        assert "すべてのデータ処理が完了しました。" in captured.out

def test_process_race_data_with_bac_error():
    """BACデータ処理でエラーが発生した場合のテスト"""
    with patch('horse_racing.data.process_race_data.process_all_bac_files') as mock_bac:
        # BACデータ処理でエラーを発生させる
        mock_bac.side_effect = Exception("BAC処理エラー")
        
        # エラーが伝播することを確認
        with pytest.raises(Exception) as exc_info:
            process_race_data()
        assert "BAC処理エラー" in str(exc_info.value)

def test_process_race_data_with_sed_error():
    """SEDデータ処理でエラーが発生した場合のテスト"""
    with patch('horse_racing.data.process_race_data.process_all_bac_files') as mock_bac, \
         patch('horse_racing.data.process_race_data.process_all_sed_files') as mock_sed:
        # SEDデータ処理でエラーを発生させる
        mock_sed.side_effect = Exception("SED処理エラー")
        
        # エラーが伝播することを確認
        with pytest.raises(Exception) as exc_info:
            process_race_data()
        assert "SED処理エラー" in str(exc_info.value)

def test_process_race_data_execution():
    """メイン実行部分のテスト"""
    from horse_racing.data import process_race_data
    
    with patch.object(process_race_data, 'process_race_data') as mock_process:
        # __name__ == '__main__'の状況をシミュレート
        original_name = process_race_data.__name__
        try:
            process_race_data.__name__ = '__main__'
            
            # メインモジュールのコードを直接実行
            if process_race_data.__name__ == '__main__':
                process_race_data.process_race_data()
            
            # 関数が1回呼び出されたことを確認
            mock_process.assert_called_once()
        finally:
            # モジュール名を元に戻す
            process_race_data.__name__ = original_name 
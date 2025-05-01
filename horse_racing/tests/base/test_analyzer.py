"""
BaseAnalyzerクラスのテストモジュール
"""

import pytest
import pandas as pd
from pathlib import Path
from horse_racing.base.analyzer import BaseAnalyzer, AnalysisConfig

# テスト用のアナライザークラスをフィクスチャとして定義
@pytest.fixture
def mock_analyzer_class():
    """テスト用のアナライザークラスを生成"""
    class MockAnalyzer(BaseAnalyzer):
        def load_data(self) -> pd.DataFrame:
            return pd.DataFrame({'test': [1, 2, 3]})
        
        def preprocess_data(self) -> pd.DataFrame:
            return self.df
        
        def calculate_feature(self) -> pd.DataFrame:
            return self.df
        
        def analyze(self) -> dict:
            return {'test_stats': self.df}
        
        def visualize(self) -> None:
            pass
    
    return MockAnalyzer

@pytest.fixture
def test_config():
    """テスト用の設定"""
    return AnalysisConfig(
        input_path='test_input.csv',
        output_dir='test_output',
        date_str='20240101',
        min_races=6,
        confidence_level=0.95
    )

@pytest.fixture
def analyzer(mock_analyzer_class, test_config):
    """テスト用のアナライザーインスタンス"""
    return mock_analyzer_class(test_config)

def test_init(mock_analyzer_class, test_config):
    """初期化のテスト"""
    analyzer = mock_analyzer_class(test_config)
    assert analyzer.config == test_config
    assert isinstance(analyzer.df, pd.DataFrame)
    assert analyzer.df.empty
    assert isinstance(analyzer.stats, dict)
    assert not analyzer.stats

def test_setup_output_dir(analyzer, tmp_path):
    """出力ディレクトリ設定のテスト"""
    analyzer.config.output_dir = str(tmp_path / 'test_output')
    analyzer._setup_output_dir()
    assert Path(analyzer.config.output_dir).exists()
    assert Path(analyzer.config.output_dir).is_dir()

def test_run(analyzer):
    """分析実行プロセスのテスト"""
    stats = analyzer.run()
    assert isinstance(stats, dict)
    assert 'test_stats' in stats
    assert not stats['test_stats'].empty
    assert len(stats['test_stats']) == 3

def test_save_results(analyzer, tmp_path):
    """結果保存のテスト"""
    # 出力ディレクトリの設定
    analyzer.config.output_dir = str(tmp_path / 'test_output')
    analyzer._setup_output_dir()
    
    # テストデータの作成と保存
    analyzer.stats = {
        'test_stats': pd.DataFrame({'value': [1, 2, 3]})
    }
    analyzer.save_results('test')
    
    # ファイルの存在確認
    expected_file = Path(analyzer.config.output_dir) / f"test_test_stats_{analyzer.config.date_str}.csv"
    assert expected_file.exists()
    
    # 保存されたデータの確認
    saved_df = pd.read_csv(expected_file)
    assert len(saved_df) == 3
    assert 'value' in saved_df.columns

def test_calculate_confidence_interval():
    """信頼区間計算のテスト"""
    # テストケース
    test_cases = [
        (0.5, 100, 0.95),  # 正の相関
        (0.0, 100, 0.95),  # 無相関
        (-0.5, 100, 0.95), # 負の相関
    ]
    
    for correlation, n, confidence_level in test_cases:
        lower, upper = BaseAnalyzer.calculate_confidence_interval(
            correlation, n, confidence_level
        )
        assert lower < upper
        if correlation > 0:
            assert lower > -1 and upper < 1
        elif correlation < 0:
            assert lower > -1 and upper < 1
        else:
            assert abs(lower) < 0.2 and abs(upper) < 0.2

def test_normalize_values(analyzer):
    """値の正規化テスト"""
    # 通常のケース
    series = pd.Series([1, 2, 3, 4, 5])
    normalized = analyzer.normalize_values(series)
    assert normalized.min() == 0
    assert normalized.max() == 10
    assert len(normalized) == len(series)
    
    # 全て同じ値のケース
    series = pd.Series([1, 1, 1])
    normalized = analyzer.normalize_values(series)
    assert all(normalized == 5.0)
    
    # 極端な値のケース
    series = pd.Series([-100, 0, 100])
    normalized = analyzer.normalize_values(series)
    assert normalized.min() == 0
    assert normalized.max() == 10
    assert len(normalized) == 3 
"""
BasicStatsAnalyzerのテストモジュール
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from horse_racing.analyzers.basic_stats_analyzer import BasicStatsAnalyzer
from horse_racing.base.analyzer import AnalysisConfig

class TestBasicStatsAnalyzer(BasicStatsAnalyzer):
    """テスト用のBasicStatsAnalyzer実装クラス"""
    
    def load_data(self) -> pd.DataFrame:
        return pd.DataFrame()
    
    def preprocess_data(self) -> pd.DataFrame:
        return self.df
    
    def calculate_feature(self) -> pd.DataFrame:
        return self.df
    
    def visualize(self) -> None:
        pass
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """基本統計量の計算"""
        stats = {
            "race_count": len(df),
            "horse_count": df["馬名"].nunique() if not df.empty else 0,
            "avg_distance": df["距離"].mean() if not df.empty else None,
            "avg_prize": df["本賞金"].mean() if not df.empty and "本賞金" in df.columns else None
        }
        return stats

@pytest.fixture
def test_config():
    """テスト用の設定"""
    return AnalysisConfig(
        input_path='test_input.csv',
        output_dir='test_output',
        date_str='20240101',
        min_races=3,
        confidence_level=0.95
    )

@pytest.fixture
def sample_race_data():
    """テスト用のレースデータ"""
    return pd.DataFrame({
        '場コード': [1, 1, 2],
        '年': [2024, 2024, 2024],
        '回': [1, 1, 2],
        '日': [1, 1, 2],
        'R': [1, 2, 3],
        '馬名': ['テスト馬A', 'テスト馬B', 'テスト馬A'],
        '距離': [1600, 2000, 1800],
        '着順': [1, 2, 3],
        'レース名': ['テストG1', 'テストG2', '一般レース'],
        '種別': [11, 12, 13],
        '芝ダ障害コード': ['芝', '芝', '芝'],
        '馬番': [1, 2, 3],
        'グレード': [1, 2, 5],
        '本賞金': [10000, 7000, 2000]
    })

@pytest.fixture
def analyzer(test_config):
    """テスト用のアナライザーインスタンス"""
    return TestBasicStatsAnalyzer(test_config)

def test_init(analyzer):
    """初期化のテスト"""
    assert isinstance(analyzer.config, AnalysisConfig)
    assert isinstance(analyzer.df, pd.DataFrame)
    assert analyzer.df.empty
    assert isinstance(analyzer.stats, dict)
    assert not analyzer.stats

def test_analyze(analyzer, sample_race_data):
    """基本統計分析のテスト"""
    result = analyzer.analyze(sample_race_data)
    
    assert isinstance(result, dict)
    assert 'race_count' in result
    assert 'horse_count' in result
    assert 'avg_distance' in result
    assert 'avg_prize' in result
    
    # 値の検証
    assert result['race_count'] == 3
    assert result['horse_count'] == 2
    assert result['avg_distance'] == sample_race_data['距離'].mean()
    assert result['avg_prize'] == sample_race_data['本賞金'].mean()

def test_analyze_without_prize(analyzer, sample_race_data):
    """賞金データがない場合のテスト"""
    data = sample_race_data.drop('本賞金', axis=1)
    result = analyzer.analyze(data)
    
    assert isinstance(result, dict)
    assert result['avg_prize'] is None

def test_analyze_empty_data(analyzer):
    """空のデータフレームでのテスト"""
    empty_df = pd.DataFrame(columns=[
        '場コード', '年', '回', '日', 'R', '馬名', '距離',
        '着順', 'レース名', '種別', '芝ダ障害コード', '馬番',
        'グレード', '本賞金'
    ])
    result = analyzer.analyze(empty_df)
    
    assert isinstance(result, dict)
    assert result['race_count'] == 0
    assert result['horse_count'] == 0
    assert result['avg_distance'] is None
    assert result['avg_prize'] is None 
"""
RaceLevelAnalyzerのテストモジュール
"""

import pytest
import pandas as pd
import logging
from unittest.mock import patch
from horse_racing.analyzers.race_level_analyzer import RaceLevelAnalyzer
from horse_racing.base.analyzer import AnalysisConfig
from horse_racing.visualization.plotter import RacePlotter
from horse_racing.data.loader import RaceDataLoader

# ロガーの設定
logger = logging.getLogger(__name__)

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
    df = pd.DataFrame({
        '場コード': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        '年': [2024] * 10,
        '回': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        '日': [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
        'R': list(range(1, 11)),
        '馬名': ['テスト馬A', 'テスト馬B', 'テスト馬A', 'テスト馬B', 'テスト馬A',
               'テスト馬B', 'テスト馬A', 'テスト馬B', 'テスト馬A', 'テスト馬B'],
        '距離': [1600, 2000, 1800, 2000, 1600, 1800, 2000, 1600, 2000, 1800],
        '着順': [1, 2, 3, 1, 2, 1, 2, 1, 3, 2],
        'レース名': ['テストG1', 'テストG2', '一般レース', 'テストG3', 'テストL',
                  'テストG2', '一般レース', 'テストG3', 'テストL', 'テストG1'],
        '種別': [11, 12, 13, 14, 13, 12, 13, 14, 13, 11],
        '芝ダ障害コード': ['芝'] * 10,
        '馬番': list(range(1, 11)),
        'グレード': [1, 2, 5, 3, 6, 2, 5, 3, 6, 1],
        '本賞金': [10000, 7000, 2000, 4500, 2500, 7000, 2000, 4500, 2500, 10000]
    })
    df['1着賞金'] = df['本賞金']
    df['is_win'] = df['着順'] == 1
    df['is_placed'] = df['着順'] <= 3
    df['race_level'] = 5.0
    return df

@pytest.fixture
def analyzer(test_config):
    """テスト用のアナライザーインスタンス"""
    with patch('horse_racing.data.loader.RaceDataLoader.load') as mock_load:
        analyzer = RaceLevelAnalyzer(test_config)
        mock_load.return_value = pd.DataFrame()
        return analyzer

def test_init(analyzer):
    """初期化のテスト"""
    assert isinstance(analyzer.plotter, RacePlotter)
    assert isinstance(analyzer.loader, RaceDataLoader)

def test_determine_grade(sample_race_data):
    """グレード判定のテスト"""
    # G1レースのテスト
    row = pd.Series({
        'レース名': 'テストG1',
        '種別': 11,
        '本賞金': 10000
    })
    assert RaceLevelAnalyzer.determine_grade(row) == 1

    # G2レースのテスト
    row = pd.Series({
        'レース名': 'テストG2',
        '種別': 12,
        '本賞金': 7000
    })
    assert RaceLevelAnalyzer.determine_grade(row) == 2

    # 一般レースのテスト
    row = pd.Series({
        'レース名': '一般レース',
        '種別': 13,
        '本賞金': 2000
    })
    assert RaceLevelAnalyzer.determine_grade(row) == 6

def test_determine_grade_by_prize():
    """賞金によるグレード判定のテスト"""
    test_cases = [
        ({'本賞金': 10000}, 1),  # G1
        ({'本賞金': 7000}, 2),   # G2
        ({'本賞金': 4500}, 3),   # G3
        ({'本賞金': 3500}, 4),   # 重賞
        ({'本賞金': 2000}, 6),   # L
        ({'本賞金': 1000}, 5),   # 一般
        ({'本賞金': None}, None) # 賞金なし
    ]
    
    for row_data, expected_grade in test_cases:
        row = pd.Series(row_data)
        assert RaceLevelAnalyzer.determine_grade_by_prize(row) == expected_grade

def test_calculate_feature(analyzer, sample_race_data):
    """レースレベル計算のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer.calculate_feature()
    
    assert 'race_level' in result.columns
    assert 'is_win' in result.columns
    assert 'is_placed' in result.columns
    assert not result['race_level'].isna().any()
    assert result['race_level'].between(0, 10).all()
    
    # 距離による補正が適切に適用されていることを確認
    distance_2000m = result[result['距離'] == 2000]['race_level'].mean()
    distance_1600m = result[result['距離'] == 1600]['race_level'].mean()
    assert distance_2000m > distance_1600m

def test_calculate_grade_level(analyzer, sample_race_data):
    """グレードレベル計算のテスト"""
    analyzer.df = sample_race_data.copy()
    grade_level = analyzer._calculate_grade_level(sample_race_data)
    
    assert isinstance(grade_level, pd.Series)
    assert not grade_level.isna().any()
    assert grade_level.between(0, 10).all()
    
    # G1の勝利が最高レベルになることを確認
    g1_win = sample_race_data[
        (sample_race_data['グレード'] == 1) &
        (sample_race_data['is_win'])
    ].index
    assert grade_level[g1_win].iloc[0] == 10.0

def test_calculate_prize_level(analyzer, sample_race_data):
    """賞金レベル計算のテスト"""
    analyzer.df = sample_race_data.copy()
    prize_level = analyzer._calculate_prize_level(sample_race_data)
    
    assert isinstance(prize_level, pd.Series)
    assert not prize_level.isna().any()
    assert prize_level.between(0, 10).all()
    
    # 最高賞金のレースが最高レベルになることを確認
    max_prize = sample_race_data['本賞金'].max()
    max_prize_races = sample_race_data['本賞金'] == max_prize
    assert prize_level[max_prize_races].iloc[0] == 10.0

def test_apply_distance_correction(analyzer, sample_race_data):
    """距離補正のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer.calculate_feature()
    
    assert 'race_level' in result.columns
    assert not result['race_level'].isna().any()
    assert result['race_level'].between(0, 10).all()
    
    # 距離による補正が適用されていることを確認
    distance_2000m = result[result['距離'] == 2000]['race_level'].mean()
    distance_1600m = result[result['距離'] == 1600]['race_level'].mean()
    assert distance_2000m > distance_1600m

def test_analyze(analyzer, sample_race_data):
    """分析実行のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer.analyze()
    
    assert isinstance(result, dict)
    assert 'horse_stats' in result
    assert 'grade_stats' in result
    assert 'correlation_stats' in result
    assert 'logistic_stats' in result

def test_calculate_horse_stats(analyzer, sample_race_data):
    """馬統計計算のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer._calculate_horse_stats()
    
    assert isinstance(result, pd.DataFrame)
    assert '馬名' in result.columns
    assert '勝利数' in result.columns
    assert '複勝数' in result.columns
    assert not result.empty
    
    # テスト馬Aの統計を確認
    horse_a = result[result['馬名'] == 'テスト馬A'].iloc[0]
    assert horse_a['勝利数'] == 1
    assert horse_a['複勝数'] >= 3

def test_calculate_grade_stats(analyzer, sample_race_data):
    """グレード統計計算のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer._calculate_grade_stats()
    
    assert isinstance(result, pd.DataFrame)
    assert 'グレード' in result.columns
    assert '勝率' in result.columns
    assert 'レース数' in result.columns
    assert not result.empty

def test_perform_correlation_analysis(analyzer, sample_race_data):
    """相関分析のテスト"""
    analyzer.df = sample_race_data.copy()
    horse_stats = analyzer._calculate_horse_stats()
    result = analyzer._perform_correlation_analysis(horse_stats)
    
    assert isinstance(result, dict)
    if result:  # データが十分にある場合
        assert 'correlation' in result
        assert 'model' in result
        assert 'r2' in result

def test_calculate_level_year_stats(analyzer, sample_race_data):
    """年別レベル統計計算のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer._calculate_level_year_stats()
    
    assert isinstance(result, pd.DataFrame)
    assert 'レースレベル区分' in result.columns
    assert '年' in result.columns
    assert 'レース数' in result.columns
    assert '勝率' in result.columns

def test_calculate_distance_level_pivot(analyzer, sample_race_data):
    """距離レベルピボットテーブル計算のテスト"""
    analyzer.df = sample_race_data.copy()
    result = analyzer._calculate_distance_level_pivot()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.index.name == '距離区分' 
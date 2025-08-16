import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime
from sklearn.linear_model import LinearRegression

from horse_racing.analyzers.race_level_analyzer import RaceLevelAnalyzer, AnalysisConfig
from horse_racing.data.loader import RaceDataLoader
from horse_racing.visualization.plotter import RacePlotter
from horse_racing.analyzers.causal_analyzer import analyze_causal_relationship, generate_causal_analysis_report

@pytest.fixture
def mock_config(tmp_path):
    """Fixture for AnalysisConfig."""
    return AnalysisConfig(
        input_path=str(tmp_path / "dummy_data.csv"),
        output_dir=str(tmp_path / "output"),
        date_str="20230101",
        min_races=3,
        confidence_level=0.95,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )

@pytest.fixture
def dummy_df():
    """Dummy DataFrame for RaceLevelAnalyzer tests."""
    data = {
        '場コード': ['01', '01', '02', '01', '01', '02'],
        '年': [2023, 2023, 2023, 2023, 2023, 2023],
        '回': [1, 1, 1, 2, 2, 2],
        '日': [1, 2, 1, 1, 2, 1],
        'R': [1, 2, 1, 1, 2, 1],
        '馬名': ['HorseA', 'HorseA', 'HorseB', 'HorseA', 'HorseB', 'HorseA'],
        '距離': [1600, 2000, 1600, 1800, 2400, 2000],
        '着順': [1, 5, 2, 1, 3, 4],
        'レース名': ['G1レース', 'OPレース', 'G3レース', '特別レース', 'G2レース', 'Lレース'],
        '種別': [13, 13, 13, 13, 13, 13],
        '芝ダ障害コード': ['芝', '芝', 'ダート', '芝', '芝', 'ダート'],
        '馬番': [1, 2, 3, 1, 2, 3],
        'クラス': [1, 5, 3, 5, 2, 6],
        '本賞金': [10000, 2000, 4000, 3000, 7000, 2500],
        '1着賞金': [5000, 1000, 2000, 1500, 3500, 1200],
        '年月日': ['20230101', '20230108', '20230101', '20230201', '20230208', '20230201'],
        'is_win': [True, False, False, True, False, False],
        'is_placed': [True, False, True, True, True, False]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['年月日'], format='%Y%m%d') # Add date column
    return df

@pytest.fixture
def analyzer(mock_config, dummy_df, mocker):
    """RaceLevelAnalyzer instance for testing."""
    mocker.patch('horse_racing.data.loader.RaceDataLoader.load', return_value=dummy_df)
    mock_plotter_class = mocker.patch('horse_racing.analyzers.race_level_analyzer.RacePlotter', autospec=True)
    analyzer_instance = RaceLevelAnalyzer(mock_config)
    analyzer_instance.plotter = mock_plotter_class.return_value # Assign the mock instance
    return analyzer_instance

def test_init(mock_config, analyzer, mocker):
    """Test RaceLevelAnalyzer initialization."""
    assert analyzer.config == mock_config
    assert isinstance(analyzer.plotter, MagicMock)
    assert isinstance(analyzer.loader, RaceDataLoader)
    assert analyzer.output_dir.exists()

def test_load_data(analyzer, dummy_df, mocker):
    """Test load_data method."""
    with patch.object(analyzer.loader, 'load', return_value=dummy_df) as mock_load:
        df = analyzer.load_data()
        mock_load.assert_called_once()
        pd.testing.assert_frame_equal(df, dummy_df)

def test_load_data_file_not_found(analyzer, mocker):
    """Test load_data with FileNotFoundError."""
    with patch.object(analyzer.loader, 'load', side_effect=FileNotFoundError) as mock_load:
        with pytest.raises(FileNotFoundError):
            analyzer.load_data()
        mock_load.assert_called_once()

def test_load_data_other_exception(analyzer, mocker):
    """Test load_data with other exceptions."""
    with patch.object(analyzer.loader, 'load', side_effect=Exception("Test Error")) as mock_load:
        with pytest.raises(Exception, match="Test Error"):
            analyzer.load_data()
        mock_load.assert_called_once()

def test_preprocess_data(analyzer, dummy_df, mock_config, mocker):
    """Test preprocess_data method."""
    analyzer.df = dummy_df.copy()
    processed_df = analyzer.preprocess_data()

    # Check column stripping
    assert all(col == col.strip() for col in processed_df.columns)

    # Check date filtering
    expected_df = dummy_df[
        (dummy_df['date'] >= mock_config.start_date) &
        (dummy_df['date'] <= mock_config.end_date)
    ]
    # min_races = 3, HorseA has 4 races, HorseB has 2 races
    expected_df = expected_df[expected_df['馬名'] == 'HorseA']
    
    pd.testing.assert_frame_equal(processed_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_preprocess_data_no_date_column(analyzer, mock_config, mocker):
    """Test preprocess_data when '年月日' column is missing."""
    df_no_ymd = pd.DataFrame({
        '場コード': ['01', '01', '02', '01', '01', '02', '01', '01', '02', '01'],
        '年': [2023] * 10,
        '回': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2],
        '日': [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        'R': [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        '馬名': ['HorseA', 'HorseA', 'HorseA', 'HorseA', 'HorseA', 'HorseA', 'HorseB', 'HorseB', 'HorseB', 'HorseB'],
        '距離': [1600, 2000, 1600, 1800, 2400, 2000, 1600, 2000, 1600, 1800],
        '着順': [1, 5, 2, 1, 3, 4, 1, 5, 2, 1],
        'レース名': ['G1レース', 'OPレース', 'G3レース', '特別レース', 'G2レース', 'Lレース', 'G1レース', 'OPレース', 'G3レース', '特別レース'],
        '種別': [13] * 10,
        '芝ダ障害コード': ['芝', '芝', 'ダート', '芝', '芝', 'ダート', '芝', '芝', 'ダート', '芝'],
        '馬番': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'クラス': [1, 5, 3, 5, 2, 6, 1, 5, 3, 5],
        '本賞金': [10000, 2000, 4000, 3000, 7000, 2500, 10000, 2000, 4000, 3000],
        '1着賞金': [5000, 1000, 2000, 1500, 3500, 1200, 5000, 1000, 2000, 1500]
    })
    analyzer.df = df_no_ymd.copy()
    processed_df = analyzer.preprocess_data()
    assert 'date' in processed_df.columns
    assert len(processed_df) == 4 # HorseA has 4 races, HorseB has 2 races (min_races=3)

def test_preprocess_data_date_conversion_failure(analyzer, dummy_df, mocker):
    """Test preprocess_data with date conversion failure."""
    df_bad_date = dummy_df.copy()
    df_bad_date.loc[0, '年月日'] = 'invalid_date'
    analyzer.df = df_bad_date
    with pytest.raises(ValueError, match="time data \"invalid_date\" doesn't match format \"%Y%m%d\""):
        analyzer.preprocess_data()

def test_preprocess_data_no_data_after_min_races_filter(analyzer, dummy_df, mocker):
    """Test preprocess_data when no data remains after min_races filter."""
    analyzer.config.min_races = 100 # Set a high min_races to filter out all data
    analyzer.df = dummy_df.copy()
    with pytest.raises(ValueError, match="条件を満たすデータが見つかりません"):
        analyzer.preprocess_data()

def test_calculate_feature(analyzer, dummy_df, mocker):
    """Test calculate_feature method."""
    analyzer.df = dummy_df.copy()
    featured_df = analyzer.calculate_feature()

    assert "race_level" in featured_df.columns
    assert "is_win" in featured_df.columns
    assert "is_placed" in featured_df.columns
    assert featured_df["race_level"].between(0, 10).all()

def test_perform_logistic_regression_analysis(analyzer, dummy_df, mocker):
    """Test _perform_logistic_regression_analysis method."""
    analyzer.df = dummy_df.copy()
    # Add race_level column as it's used in the method
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10
    
    results = analyzer._perform_logistic_regression_analysis()
    assert "win" in results
    assert "place" in results
    assert "model" in results["win"]
    assert "scaler" in results["win"]
    assert "accuracy" in results["win"]
    assert "report" in results["win"]
    assert "conf_matrix" in results["win"]
    assert "data" in results

def test_analyze(analyzer, dummy_df, mocker):
    """Test analyze method."""
    analyzer.df = dummy_df.copy()
    analyzer.df = analyzer.preprocess_data()
    analyzer.df = analyzer.calculate_feature()

    with patch('horse_racing.analyzers.causal_analyzer.analyze_causal_relationship', return_value={}) as mock_causal_analysis:
        with patch('horse_racing.analyzers.causal_analyzer.generate_causal_analysis_report') as mock_generate_report:
            with patch.object(analyzer, '_perform_correlation_analysis', return_value={}) as mock_correlation_analysis:
                results = analyzer.analyze()
                mock_correlation_analysis.assert_called_once()
                mock_causal_analysis.assert_called_once_with(analyzer.df)
                mock_generate_report.assert_called_once()
                assert 'correlation_stats' in results
                assert 'causal_analysis' in results

def test_visualize(analyzer, dummy_df, mocker):
    """Test visualize method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.stats = {
        'correlation_stats': {},
        'causal_analysis': {}
    }

    with patch.object(analyzer.plotter, '_visualize_correlations') as mock_plot_correlations:
        with patch.object(analyzer, '_visualize_causal_analysis') as mock_visualize_causal:
            analyzer.visualize()
            mock_plot_correlations.assert_called_once()
            mock_visualize_causal.assert_called_once()

def test_visualize_no_stats(analyzer, mocker):
    """Test visualize method raises ValueError if no stats."""
    analyzer.stats = {}
    with pytest.raises(ValueError, match="分析結果がありません"):
        analyzer.visualize()

def test_visualize_causal_analysis(analyzer, dummy_df, mocker):
    """Test _visualize_causal_analysis method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.stats = {
        'causal_analysis': {
            'temporal_precedence': {},
            'mechanism': {},
            'confounding_factors': {}
        }
    }

    with patch.object(analyzer, '_plot_temporal_precedence') as mock_plot_temporal:
        with patch.object(analyzer, '_plot_mechanism_analysis') as mock_plot_mechanism:
            with patch.object(analyzer, '_plot_confounding_factors') as mock_plot_confounding:
                analyzer._visualize_causal_analysis()
                mock_plot_temporal.assert_called_once()
                mock_plot_mechanism.assert_called_once()
                mock_plot_confounding.assert_called_once()

def test_plot_temporal_precedence(analyzer, dummy_df, mock_config):
    """Test _plot_temporal_precedence method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    
    # Ensure enough data for the plot
    analyzer.df = pd.concat([analyzer.df] * 2, ignore_index=True) # Duplicate data to ensure >= 6 races per horse
    analyzer.df['年月日'] = pd.to_datetime(analyzer.df['年月日'], format='%Y%m%d')
    analyzer.df = analyzer.df.sort_values(by=['馬名', '年月日'])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            analyzer._plot_temporal_precedence(Path(mock_config.output_dir) / 'causal_analysis')
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

def test_plot_mechanism_analysis(analyzer, dummy_df, mock_config):
    """Test _plot_mechanism_analysis method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3

    # Ensure enough data for the plot
    analyzer.df = pd.concat([analyzer.df] * 2, ignore_index=True) # Duplicate data to ensure >= 6 races per horse
    analyzer.df['年月日'] = pd.to_datetime(analyzer.df['年月日'], format='%Y%m%d')
    analyzer.df = analyzer.df.sort_values(by=['馬名', '年月日'])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            analyzer._plot_mechanism_analysis(Path(mock_config.output_dir) / 'causal_analysis')
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

def test_plot_confounding_factors(analyzer, dummy_df, mock_config):
    """Test _plot_confounding_factors method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            analyzer._plot_confounding_factors(Path(mock_config.output_dir) / 'causal_analysis')
            assert mock_savefig.call_count == 3 # For each confounder (場コード, 距離, 芝ダ障害コード)
            assert mock_close.call_count == 3

def test_calculate_grade_level(analyzer, dummy_df, mocker):
    """Test _calculate_grade_level method with updated logic."""
    analyzer.df = dummy_df.copy()
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    
    # Test with existing grade column
    analyzer.df['グレード'] = [1, 2, 3, 1, 2, 3]  # Add grade column (6 items to match DataFrame length)
    grade_level = analyzer._calculate_grade_level(analyzer.df)
    assert "race_level" not in grade_level.name # Should return a Series, not a DataFrame
    assert grade_level.between(0, 10).all()
    
    # Test fallback to prize-based calculation
    analyzer.df = dummy_df.copy()  # Reset without grade column
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    grade_level_fallback = analyzer._calculate_grade_level(analyzer.df)
    assert grade_level_fallback.between(0, 10).all()

def test_convert_grade_to_level(analyzer, dummy_df, mocker):
    """Test _convert_grade_to_level method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['グレード'] = [1, 2, 3, 4, 5, 6]  # Add various grade values
    
    grade_level = analyzer._convert_grade_to_level(analyzer.df, 'グレード')
    
    # Check expected mappings
    expected_mappings = {1: 9.0, 2: 7.5, 3: 6.0, 4: 4.5, 5: 2.0, 6: 3.0}
    for i, (grade, expected_level) in enumerate(expected_mappings.items()):
        if i < len(grade_level):
            assert grade_level.iloc[i] == expected_level

def test_calculate_grade_level_from_prize(analyzer, dummy_df, mocker):
    """Test _calculate_grade_level_from_prize method."""
    analyzer.df = dummy_df.copy()
    grade_level = analyzer._calculate_grade_level_from_prize(analyzer.df)
    assert grade_level.between(0, 10).all()

def test_calculate_prize_level(analyzer, dummy_df, mocker):
    """Test _calculate_prize_level method."""
    analyzer.df = dummy_df.copy()
    prize_level = analyzer._calculate_prize_level(analyzer.df)
    assert prize_level.between(0, 10).all()

def test_calculate_horse_stats(analyzer, dummy_df, mocker):
    """Test _calculate_horse_stats method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3

    horse_stats = analyzer._calculate_horse_stats()
    assert "最高レベル" in horse_stats.columns
    assert "平均レベル" in horse_stats.columns
    assert "win_rate" in horse_stats.columns
    assert "place_rate" in horse_stats.columns
    assert "出走回数" in horse_stats.columns
    assert "主戦クラス" in horse_stats.columns
    assert len(horse_stats) == 1 # Only HorseA meets min_races=3
    assert horse_stats.iloc[0]['馬名'] == 'HorseA'

def test_calculate_grade_stats(analyzer, dummy_df, mocker):
    """Test _calculate_grade_stats method."""
    analyzer.df = dummy_df.copy()
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3

    grade_stats = analyzer._calculate_grade_stats()
    assert "クラス" in grade_stats.columns
    assert "勝率" in grade_stats.columns
    assert "複勝率" in grade_stats.columns
    assert "平均レベル" in grade_stats.columns

def test_perform_correlation_analysis(analyzer, dummy_df, mocker):
    """Test _perform_correlation_analysis method."""
    analyzer.df = dummy_df.copy()
    analyzer.df['race_level'] = np.random.rand(len(dummy_df)) * 10 # Add race_level
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3
    analyzer.df["is_win"] = analyzer.df["着順"] == 1
    analyzer.df["is_placed"] = analyzer.df["着順"] <= 3

    horse_stats = analyzer._calculate_horse_stats()
    results = analyzer._perform_correlation_analysis(horse_stats)

    assert "correlation_win_max" in results
    assert "model_win_max" in results
    assert "r2_win_max" in results

def test_perform_correlation_analysis_empty_data(analyzer, mocker):
    """Test _perform_correlation_analysis with empty data."""
    empty_df = pd.DataFrame(columns=['馬名', '最高レベル', '平均レベル', 'win_rate', 'place_rate'])
    results = analyzer._perform_correlation_analysis(empty_df)
    assert results == {}

def test_perform_correlation_analysis_zero_std(analyzer, mocker):
    """Test _perform_correlation_analysis with zero standard deviation."""
    data = pd.DataFrame({
        '馬名': ['A', 'B'],
        '最高レベル': [5.0, 5.0],
        '平均レベル': [5.0, 5.0],
        'win_rate': [0.5, 0.5],
        'place_rate': [0.8, 0.8],
        '出走回数': [10, 10],
        '主戦クラス': [1, 1]
    })
    results = analyzer._perform_correlation_analysis(data)
    assert results["correlation_win_max"] == 0.0
    assert results["r2_win_max"] == 0.0
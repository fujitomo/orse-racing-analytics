import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from horse_racing.base.analyzer import AnalysisConfig, BaseAnalyzer

# Mock implementation of BaseAnalyzer for testing abstract methods
class MockAnalyzer(BaseAnalyzer):
    def load_data(self) -> pd.DataFrame:
        return pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    def preprocess_data(self) -> pd.DataFrame:
        return self.df.copy()

    def calculate_feature(self) -> pd.DataFrame:
        return self.df.copy()

    def analyze(self) -> dict:
        return {'test_stat': 0.5}

    def visualize(self) -> None:
        pass

@pytest.fixture
def mock_config(tmp_path):
    """Fixture for AnalysisConfig."""
    return AnalysisConfig(
        input_path="dummy_path",
        output_dir=str(tmp_path / "output"),
        date_str="20230101",
        min_races=5,
        confidence_level=0.95,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )

@pytest.fixture
def mock_analyzer(mock_config):
    """Fixture for MockAnalyzer instance."""
    return MockAnalyzer(mock_config)

def test_analysis_config_defaults():
    """Test AnalysisConfig default values."""
    config = AnalysisConfig(input_path="test_path")
    assert config.input_path == "test_path"
    assert config.output_dir == "export/analysis"
    assert config.date_str == ""
    assert config.min_races == 6
    assert config.confidence_level == 0.95
    assert config.start_date is None
    assert config.end_date is None

def test_analysis_config_custom_values(mock_config):
    """Test AnalysisConfig custom values."""
    assert mock_config.input_path == "dummy_path"
    assert mock_config.output_dir.endswith("output")
    assert mock_config.date_str == "20230101"
    assert mock_config.min_races == 5
    assert mock_config.confidence_level == 0.95
    assert mock_config.start_date == datetime(2023, 1, 1)
    assert mock_config.end_date == datetime(2023, 12, 31)

def test_base_analyzer_init(mock_analyzer, mock_config):
    """Test BaseAnalyzer initialization and output directory setup."""
    assert mock_analyzer.config == mock_config
    assert isinstance(mock_analyzer.df, pd.DataFrame)
    assert mock_analyzer.df.empty
    assert mock_analyzer.stats == {}
    assert mock_analyzer.output_dir.exists()
    assert mock_analyzer.output_dir == Path(mock_config.output_dir)

def test_abstract_methods_not_implemented():
    """Test that abstract methods prevent instantiation if not implemented."""
    class IncompleteAnalyzer(BaseAnalyzer):
        # Intentionally do NOT implement abstract methods
        pass

    config = AnalysisConfig(input_path="test")
    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteAnalyzer without an implementation for abstract methods 'analyze', 'calculate_feature', 'load_data', 'preprocess_data', 'visualize'"):
        IncompleteAnalyzer(config)

def test_run_method(mock_analyzer):
    """Test the run method calls abstract methods in correct order."""
    with patch.object(mock_analyzer, 'load_data', wraps=mock_analyzer.load_data) as mock_load_data:
        with patch.object(mock_analyzer, 'preprocess_data', wraps=mock_analyzer.preprocess_data) as mock_preprocess_data:
            with patch.object(mock_analyzer, 'calculate_feature', wraps=mock_analyzer.calculate_feature) as mock_calculate_feature:
                with patch.object(mock_analyzer, 'analyze', wraps=mock_analyzer.analyze) as mock_analyze:
                    with patch.object(mock_analyzer, 'visualize', wraps=mock_analyzer.visualize) as mock_visualize:

                        stats = mock_analyzer.run()

                        mock_load_data.assert_called_once()
                        mock_preprocess_data.assert_called_once()
                        mock_calculate_feature.assert_called_once()
                        mock_analyze.assert_called_once()
                        mock_visualize.assert_called_once()
                        assert stats == {'test_stat': 0.5}
                        assert mock_analyzer.stats == {'test_stat': 0.5}

def test_save_results(mock_analyzer, mock_config):
    """Test save_results method saves DataFrame to CSV."""
    mock_analyzer.stats = {'test_df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
    mock_analyzer.save_results("test_prefix")

    expected_path = Path(mock_config.output_dir) / "test_prefix_test_df_20230101.csv"
    assert expected_path.exists()
    df_loaded = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(df_loaded, mock_analyzer.stats['test_df'])

def test_save_results_empty_stats(mock_analyzer, mock_config):
    """Test save_results does nothing if stats is empty."""
    mock_analyzer.stats = {}
    mock_analyzer.save_results("test_prefix")
    # Check that no files were created in the output directory
    assert not any(Path(mock_config.output_dir).iterdir())

def test_calculate_confidence_interval():
    """Test calculate_confidence_interval static method."""
    # Example values for correlation and n
    correlation = 0.7
    n = 100
    lower, upper = BaseAnalyzer.calculate_confidence_interval(correlation, n)
    
    # Approximate expected values (can vary slightly due to floating point precision)
    assert lower == pytest.approx(0.584, abs=1e-2)
    assert upper == pytest.approx(0.788, abs=1e-2)

def test_normalize_values(mock_analyzer):
    """Test normalize_values method."""
    series = pd.Series([0, 5, 10, 15, 20])
    normalized_series = mock_analyzer.normalize_values(series)
    pd.testing.assert_series_equal(normalized_series, pd.Series([0.0, 2.5, 5.0, 7.5, 10.0]))

def test_normalize_values_single_value(mock_analyzer):
    """Test normalize_values with a single unique value."""
    series = pd.Series([10, 10, 10])
    normalized_series = mock_analyzer.normalize_values(series)
    pd.testing.assert_series_equal(normalized_series, pd.Series([5.0, 5.0, 5.0]))
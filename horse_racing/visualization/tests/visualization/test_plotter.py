
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from horse_racing.visualization.plotter import RacePlotter

@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for plots."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

@pytest.fixture
def plotter(output_dir):
    """RacePlotter instance for testing."""
    return RacePlotter(output_dir)

def test_init_and_setup_style(plotter, output_dir):
    """Test initialization and style setup."""
    assert plotter.output_dir == output_dir
    # Check if _setup_style was called (indirectly by checking rcParams)
    assert 'MS Gothic' in plt.rcParams['font.family']

def test_save_plot(plotter, output_dir):
    """Test saving a plot."""
    fig, ax = plt.subplots()
    filename = "test_plot.png"
    plotter.save_plot(fig, filename)
    assert (output_dir / filename).exists()

def test_plot_grade_stats(plotter, output_dir):
    """Test plotting grade statistics."""
    grade_stats = pd.DataFrame({
        "グレード": [1, 2, 3],
        "勝率": [0.3, 0.2, 0.1],
        "複勝率": [0.6, 0.5, 0.4]
    })
    grade_levels = {
        1: {'name': 'G1'},
        2: {'name': 'G2'},
        3: {'name': 'G3'}
    }
    plotter.plot_grade_stats(grade_stats, grade_levels)
    assert (output_dir / "grade_win_rate.png").exists()

def test_plot_correlation_analysis(plotter, output_dir):
    """Test plotting correlation analysis."""
    data = pd.DataFrame({
        "平均レベル": np.random.rand(10) * 100,
        "最高レベル": np.random.rand(10) * 100,
        "win_rate": np.random.rand(10),
        "出走回数": np.random.randint(10, 100, 10),
        "主戦クラス": np.random.randint(1, 5, 10)
    })
    model = LinearRegression()
    model.fit(data[["平均レベル"]], data["win_rate"])
    
    plotter.plot_correlation_analysis(
        data=data,
        correlation=0.5,
        model=model,
        r2=0.25,
        feature_name="平均レベルと勝率",
        x_column="平均レベル",
        y_column="win_rate"
    )
    assert (output_dir / "平均レベルと勝率_correlation.png").exists()
    assert (output_dir / "平均レベルと勝率_binned_correlation.png").exists()

def test_plot_distribution_analysis(plotter, output_dir):
    """Test plotting distribution analysis."""
    data = pd.DataFrame({"feature": np.random.randn(100)})
    plotter.plot_distribution_analysis(data, "feature", title="Feature Distribution")
    assert (output_dir / "feature_distribution.png").exists()

def test_plot_trend_analysis(plotter, output_dir):
    """Test plotting trend analysis."""
    data = pd.DataFrame({
        "年": [2020, 2021, 2022, 2020, 2021, 2022],
        "feature区分": ["A", "A", "A", "B", "B", "B"],
        "勝率": np.random.rand(6)
    })
    plotter.plot_trend_analysis(data, "feature")
    assert (output_dir / "feature_trends.png").exists()

def test_plot_heatmap(plotter, output_dir):
    """Test plotting heatmap."""
    pivot_table = pd.DataFrame(np.random.rand(5, 5), index=list('ABCDE'), columns=list('FGHIJ'))
    plotter.plot_heatmap(pivot_table, "Test Heatmap", "test_heatmap.png")
    assert (output_dir / "test_heatmap.png").exists()

def test_plot_logistic_regression_curve(plotter, output_dir):
    """Test plotting logistic regression curve."""
    X = np.random.rand(100)
    y = (X > 0.5).astype(int)
    y_pred_proba = X
    plotter.plot_logistic_regression_curve(X, y, y_pred_proba, "TestFeature")
    assert (output_dir / "TestFeature_logistic_regression_curve.png").exists()

def test_plot_confusion_matrix(plotter, output_dir):
    """Test plotting confusion matrix."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    plotter.plot_confusion_matrix(y_true, y_pred, "TestFeature")
    assert (output_dir / "TestFeature_confusion_matrix.png").exists()

def test_plot_continuous_regression(plotter, output_dir):
    """Test plotting continuous regression."""
    X = np.random.rand(100)
    y = X * 2 + np.random.randn(100) * 0.1
    y_pred = X * 2
    plotter.plot_continuous_regression(X, y, y_pred, "Feature", "Target", 0.9, 0.01)
    assert (output_dir / "Feature_Target_continuous_regression.png").exists()

def test_visualize_correlations(plotter, output_dir):
    """Test _visualize_correlations method."""
    horse_stats = pd.DataFrame({
        "平均レベル": np.random.rand(10) * 100,
        "最高レベル": np.random.rand(10) * 100,
        "win_rate": np.random.rand(10),
        "place_rate": np.random.rand(10),
        "出走回数": np.random.randint(10, 100, 10),
        "主戦クラス": np.random.randint(1, 5, 10)
    })
    correlation_stats = {
        "correlation_win_max": 0.5, "model_win_max": LinearRegression().fit(horse_stats[["最高レベル"]], horse_stats["win_rate"]),
        "r2_win_max": 0.25,
        "correlation_place_max": 0.6, "model_place_max": LinearRegression().fit(horse_stats[["最高レベル"]], horse_stats["place_rate"]),
        "r2_place_max": 0.36,
        "correlation_win_avg": 0.7, "model_win_avg": LinearRegression().fit(horse_stats[["平均レベル"]], horse_stats["win_rate"]),
        "r2_win_avg": 0.49,
        "correlation_place_avg": 0.8, "model_place_avg": LinearRegression().fit(horse_stats[["平均レベル"]], horse_stats["place_rate"]),
        "r2_place_avg": 0.64,
    }
    
    with patch.object(plotter, 'plot_correlation_analysis') as mock_plot_correlation_analysis:
        plotter._visualize_correlations(horse_stats, correlation_stats)
        assert mock_plot_correlation_analysis.call_count == 4

def test_plot_logistic_regression(plotter, output_dir):
    """Test plot_logistic_regression method."""
    X = np.random.rand(100)
    y = (X > 0.5).astype(int)
    y_pred_proba = X
    y_pred = (X > 0.5).astype(int)
    
    with patch.object(plotter, 'plot_logistic_regression_curve') as mock_plot_curve:
        with patch.object(plotter, 'plot_confusion_matrix') as mock_plot_confusion_matrix:
            plotter.plot_logistic_regression(X, y, y_pred_proba, y_pred, "TestFeature")
            mock_plot_curve.assert_called_once()
            mock_plot_confusion_matrix.assert_called_once()

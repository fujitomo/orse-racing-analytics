import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib as mpl

class RacePlotter:
    """レース分析の可視化クラス"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_style()

    def _setup_style(self) -> None:
        """プロットスタイルの設定"""
        plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
        mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """プロットの保存"""
        fig.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_grade_stats(self, grade_stats: pd.DataFrame, grade_levels: Dict[int, Dict[str, Any]]) -> None:
        """グレード別統計のプロット"""
        fig = plt.figure(figsize=(15, 8))
        x = np.arange(len(grade_stats))
        width = 0.35

        plt.bar(x - width/2, grade_stats["勝率"], width, label="単勝率", color="skyblue")
        plt.bar(x + width/2, grade_stats["複勝率"], width, label="複勝率", color="lightcoral")

        plt.title("グレード別 勝率・複勝率")
        plt.xlabel("グレード")
        plt.ylabel("確率")
        plt.xticks(x, [f"{grade_levels[g]['name']}" for g in grade_stats["グレード"]])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot(fig, "grade_win_rate.png")

    def plot_correlation_analysis(
        self,
        data: pd.DataFrame,
        correlation: float,
        model: Any,
        r2: float,
        feature_name: str
    ) -> None:
        """相関分析の可視化"""
        fig = plt.figure(figsize=(15, 8))
        scatter = plt.scatter(
            data["最高レベル"],
            data["win_rate"],
            s=data["出走回数"]*20,
            alpha=0.5,
            c=data["主戦グレード"],
            cmap='viridis'
        )

        X_plot = np.linspace(data["最高レベル"].min(), data["最高レベル"].max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', linestyle='--', label=f'回帰直線 (R² = {r2:.3f})')

        plt.title(f"{feature_name}と勝率の関係\n相関係数: {correlation:.3f}")
        plt.xlabel(f"最高{feature_name}")
        plt.ylabel("勝率")
        plt.colorbar(scatter, label="主戦グレード")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self.save_plot(fig, f"{feature_name}_correlation.png")

    def plot_distribution_analysis(
        self,
        data: pd.DataFrame,
        feature_name: str,
        bins: int = 30
    ) -> None:
        """分布分析の可視化"""
        fig = plt.figure(figsize=(15, 8))
        sns.histplot(data=data, x=feature_name, bins=bins, kde=True)
        plt.title(f"{feature_name}の分布")
        plt.xlabel(feature_name)
        plt.ylabel("頻度")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self.save_plot(fig, f"{feature_name}_distribution.png")

    def plot_trend_analysis(
        self,
        data: pd.DataFrame,
        feature_name: str,
        time_column: str = "年"
    ) -> None:
        """トレンド分析の可視化"""
        fig = plt.figure(figsize=(15, 8))
        
        for level in sorted(data[f"{feature_name}区分"].unique()):
            level_data = data[data[f"{feature_name}区分"] == level]
            plt.plot(level_data[time_column], level_data["勝率"],
                    marker='o', label=f"レベル{level}")

        plt.title(f"{feature_name}別の勝率推移")
        plt.xlabel(time_column)
        plt.ylabel("勝率")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self.save_plot(fig, f"{feature_name}_trends.png")

    def plot_heatmap(
        self,
        pivot_table: pd.DataFrame,
        title: str,
        filename: str
    ) -> None:
        """ヒートマップの可視化"""
        fig = plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title(title)
        plt.tight_layout()

        self.save_plot(fig, filename) 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report

class RacePlotter:
    """レース分析の可視化クラス"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_style()
        self.fig_size = (15, 10)

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

    def plot_logistic_regression_curve(
        self,
        X,
        y,
        y_pred_proba,
        feature_name: str,
        title: str = "ロジスティック回帰分析"
    ) -> None:
        """
        ロジスティック回帰の予測曲線を可視化します。
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y : array-like
            実際のクラスラベル
        y_pred_proba : array-like
            予測確率
        feature_name : str
            説明変数の名前（軸ラベルに使用）
        title : str, optional
            図のタイトル
        """
        # 散布図とロジスティック回帰曲線
        fig = plt.figure(figsize=(15, 8))
        plt.scatter(X, y, color='blue', alpha=0.5, label='実データ')
        
        # データをソートしてスムーズな曲線を描画
        sort_idx = np.argsort(X)
        plt.plot(X[sort_idx], y_pred_proba[sort_idx], color='red', label='回帰曲線')
        
        plt.xlabel(feature_name)
        plt.ylabel('確率')
        plt.title(f'{title}\n予測曲線')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # プロットを保存
        self.save_plot(fig, f"{feature_name}_logistic_regression_curve.png")

    def plot_confusion_matrix(
        self,
        y,
        y_pred,
        feature_name: str,
        title: str = "ロジスティック回帰分析"
    ) -> None:
        """
        混同行列とクラス分類レポートを可視化します。
        
        Parameters:
        -----------
        y : array-like
            実際のクラスラベル
        y_pred : array-like
            予測クラスラベル
        feature_name : str
            説明変数の名前（ファイル名に使用）
        title : str, optional
            図のタイトル
        """
        # 混同行列の計算
        cm = confusion_matrix(y, y_pred)
        
        # 混同行列の可視化
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        
        # 混同行列のヒートマップ
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel('予測クラス')
        ax1.set_ylabel('実際のクラス')
        ax1.set_title(f'{title}\n混同行列')
        
        # 分類レポート
        ax2 = fig.add_subplot(gs[1])
        report = classification_report(y, y_pred)
        ax2.text(0.1, 0.1, report, family='monospace', size=10)
        ax2.axis('off')
        ax2.set_title('分類レポート')
        
        plt.tight_layout()
        
        # プロットを保存
        self.save_plot(fig, f"{feature_name}_confusion_matrix.png")

    def plot_logistic_regression(
        self,
        X,
        y,
        y_pred_proba,
        y_pred,
        feature_name: str,
        title: str = "ロジスティック回帰分析"
    ) -> None:
        """
        ロジスティック回帰分析の結果を2つの別々の図として可視化します。
        1. 散布図とロジスティック回帰曲線
        2. 混同行列と分類レポート
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y : array-like
            実際のクラスラベル
        y_pred_proba : array-like
            予測確率
        y_pred : array-like
            予測クラスラベル
        feature_name : str
            説明変数の名前（軸ラベルに使用）
        title : str, optional
            図全体のタイトル
        """
        # 予測曲線の可視化
        self.plot_logistic_regression_curve(
            X=X,
            y=y,
            y_pred_proba=y_pred_proba,
            feature_name=feature_name,
            title=title
        )
        
        # 混同行列と分類レポートの可視化
        self.plot_confusion_matrix(
            y=y,
            y_pred=y_pred,
            feature_name=feature_name,
            title=title
        ) 
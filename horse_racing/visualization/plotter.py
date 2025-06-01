import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression

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
        # TODO:グレード別統計のプロットの可視化を調査予定
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
        feature_name: str,
        x_column: str,
        y_column: str = "win_rate"
    ) -> None:
        """相関分析の可視化"""
        fig = plt.figure(figsize=(15, 8))
        
        # X軸のデータを決定
        if x_column == "平均レベル":
            x_data = data["平均レベル"]
        else:
            x_data = data["最高レベル"]
        
        # レース回数に基づくサイズ設定（より明確に）
        min_size = 30
        max_size = 300
        race_counts = data["出走回数"]
        # レース回数を正規化してサイズに変換
        normalized_sizes = min_size + (race_counts - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
        
        # 散布図の描画
        scatter = plt.scatter(
            x_data,
            data[y_column],
            s=normalized_sizes,
            alpha=0.6,
            c=data["主戦グレード"],
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )

        # 回帰直線の描画
        # グラフの表示範囲全体に回帰直線を描画
        x_min, x_max = plt.xlim()  # 現在のX軸の表示範囲を取得
        X_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', linestyle='--', linewidth=2, label=f'回帰直線 (R² = {r2:.3f})')

        plt.title(f"{feature_name}\n相関係数: {correlation:.3f}")
        plt.xlabel(x_column)
        plt.ylabel("勝率" if y_column == "win_rate" else "複勝率")
        
        # Y軸の範囲を適切に設定
        if y_column in ["win_rate", "place_rate"]:
            plt.ylim(-0.05, 1.05)  # 確率なので0-1の範囲に制限
        
        # グリッドを追加して傾向を見やすくする
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # カラーバーの設定
        cbar = plt.colorbar(scatter)
        cbar.ax.set_title("")

        # 1つ目：タイトル（カラーバーの右外側、縦書き）
        cbar.ax.text(
            1.1, 0.5,
            "最も出走回数が多いグレード",
            va='center', ha='left', fontsize=12, fontweight='bold', rotation=90, transform=cbar.ax.transAxes
        )
        # 2つ目：例示（さらに右外側、縦書き・重ならないように間隔を広げる）
        cbar.ax.text(
            2, 0.5,  # さらに右へ
            "※例：G1: 3回、G2: 5回、G3: 2回 → 主戦グレードはG2",
            va='center', ha='left', fontsize=12, rotation=90, transform=cbar.ax.transAxes
        )
        
        # レース回数のサイズ凡例を追加
        sizes_for_legend = [race_counts.min(), race_counts.quantile(0.5), race_counts.max()]
        labels_for_legend = [f'{int(size)}回' for size in sizes_for_legend]
        
        # サイズ凡例用のマーカーサイズを計算
        legend_sizes = []
        for size in sizes_for_legend:
            normalized_size = min_size + (size - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
            legend_sizes.append(normalized_size)
        
        # サイズ凡例の作成
        legend_elements = []
        for size, label, marker_size in zip(sizes_for_legend, labels_for_legend, legend_sizes):
            legend_elements.append(plt.scatter([], [], s=marker_size, c='gray', alpha=0.6, 
                                             edgecolors='black', linewidth=0.5, label=label))
        
        # 既存の凡例と組み合わせ
        legend1 = plt.legend(handles=legend_elements, title="レース回数（点のサイズ）", 
                           loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True)
        plt.gca().add_artist(legend1)
        
        # 回帰直線の凡例
        legend2 = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # グラフ左下にレース回数の範囲情報を追加
        info_text = f"レース回数範囲: {int(race_counts.min())}～{int(race_counts.max())}回\n平均: {race_counts.mean():.1f}回"
        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom', fontsize=10)
        
        plt.tight_layout()

        self.save_plot(fig, f"{feature_name}_correlation.png")
        
        # ビニング版の散布図も作成（傾向をより明確に表示）
        self._plot_binned_correlation(data, x_data, y_column, feature_name, x_column)

    def _plot_binned_correlation(self, data, x_data, y_column, feature_name, x_column):
        """ビニング版の相関分析（傾向をより明確に表示）"""
        fig = plt.figure(figsize=(15, 8))
        
        # データをビンに分割
        n_bins = 10
        bins = np.linspace(x_data.min(), x_data.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 各ビンの平均値と標準誤差、レース回数統計を計算
        bin_means = []
        bin_stds = []
        bin_counts = []
        bin_race_counts_mean = []  # 各ビンの平均レース回数
        
        for i in range(len(bins) - 1):
            mask = (x_data >= bins[i]) & (x_data < bins[i + 1])
            if i == len(bins) - 2:  # 最後のビンは右端を含む
                mask = (x_data >= bins[i]) & (x_data <= bins[i + 1])
            
            if mask.sum() > 0:
                bin_data = data[mask][y_column]
                bin_race_data = data[mask]["出走回数"]
                bin_means.append(bin_data.mean())
                bin_stds.append(bin_data.std() / np.sqrt(len(bin_data)))  # 標準誤差
                bin_counts.append(len(bin_data))
                bin_race_counts_mean.append(bin_race_data.mean())  # 平均レース回数
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
                bin_counts.append(0)
                bin_race_counts_mean.append(np.nan)
        
        # 有効なデータのみを抽出
        valid_mask = ~np.isnan(bin_means)
        valid_centers = bin_centers[valid_mask]
        valid_means = np.array(bin_means)[valid_mask]
        valid_stds = np.array(bin_stds)[valid_mask]
        valid_counts = np.array(bin_counts)[valid_mask]
        valid_race_counts = np.array(bin_race_counts_mean)[valid_mask]
        
        # エラーバー付きの散布図
        plt.errorbar(valid_centers, valid_means, yerr=valid_stds, 
                    fmt='o', markersize=8, capsize=5, capthick=2, 
                    color='blue', alpha=0.7, label='区間平均値')
        
        # バブルサイズでデータ数を表現（サイズをレース回数平均に比例させる）
        max_race_count = valid_race_counts.max() if len(valid_race_counts) > 0 else 1
        min_race_count = valid_race_counts.min() if len(valid_race_counts) > 0 else 1
        bubble_sizes = 50 + (valid_race_counts - min_race_count) / (max_race_count - min_race_count) * 200
        
        scatter = plt.scatter(valid_centers, valid_means, s=bubble_sizes, 
                   alpha=0.4, c=valid_race_counts, cmap='Reds', 
                   edgecolors='black', linewidth=1, label='平均レース回数（バブルサイズ・色）')
        
        # カラーバーの追加
        cbar = plt.colorbar(scatter)
        cbar.set_label('平均レース回数', rotation=270, labelpad=20)
        
        # 線形回帰直線
        if len(valid_centers) > 1:
            z = np.polyfit(valid_centers, valid_means, 1)
            p = np.poly1d(z)
            plt.plot(valid_centers, p(valid_centers), "r--", alpha=0.8, linewidth=2, 
                    label=f'回帰直線 (傾き: {z[0]:.3f})')
        
        plt.title(f"{feature_name}（区間平均版）\n各区間の平均値とトレンド")
        plt.xlabel(x_column)
        plt.ylabel(f"平均{('勝率' if y_column == 'win_rate' else '複勝率')}")
        
        # Y軸の範囲を適切に設定
        if y_column in ["win_rate", "place_rate"]:
            plt.ylim(-0.05, 1.05)  # 確率なので0-1の範囲に制限
        
        # グラフ右上にレース回数統計情報を追加
        race_stats_text = f"レース回数統計（区間別）:\n最小: {min_race_count:.1f}回\n最大: {max_race_count:.1f}回\n全体平均: {data['出走回数'].mean():.1f}回"
        plt.text(0.98, 0.98, race_stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', horizontalalignment='right', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        self.save_plot(fig, f"{feature_name}_binned_correlation.png")

    def plot_distribution_analysis(
        self,
        data: pd.DataFrame,
        feature_name: str,
        bins: int = 30,
        title: str = None
    ) -> None:
        """分布分析の可視化"""
        fig = plt.figure(figsize=(15, 8))
        sns.histplot(data=data, x=feature_name, bins=bins, kde=True)
        plt.title(title if title else f"{feature_name}の分布")
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
        title: str = "ロジスティック回帰分析",
        threshold_info: dict = None
    ) -> None:
        """
        改善されたロジスティック回帰の予測曲線を可視化します。
        
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
        threshold_info : dict, optional
            閾値情報
        """
        # 散布図とロジスティック回帰曲線
        fig = plt.figure(figsize=(15, 10))
        
        # 実データの散布図（透明度とサイズを調整）
        colors = ['red' if label == 0 else 'blue' for label in y]
        plt.scatter(X, y, c=colors, alpha=0.6, s=30, 
                   label='実データ (赤:低, 青:高)')
        
        # データをソートしてスムーズな曲線を描画
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_proba_sorted = y_pred_proba[sort_idx]
        
        # 回帰曲線を太く、目立つ色で描画
        plt.plot(X_sorted, y_proba_sorted, color='darkgreen', linewidth=3, 
                label='ロジスティック回帰曲線')
        
        # 50%ラインを追加
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                   label='50%ライン')
        
        # 軸ラベルとタイトル
        plt.xlabel(feature_name.split('（')[0], fontsize=12)
        plt.ylabel('高成績確率', fontsize=12)
        
        # タイトルに閾値情報を追加
        title_text = f'{title}\n予測曲線'
        if threshold_info:
            if 'place_threshold' in threshold_info:
                title_text += f' (複勝率閾値: {threshold_info["place_threshold"]:.2f})'
            elif 'win_threshold' in threshold_info:
                title_text += f' (勝率閾値: {threshold_info["win_threshold"]:.2f})'
        
        plt.title(title_text, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Y軸を0-1に制限
        plt.ylim(-0.05, 1.05)
        
        # X軸の範囲を適切に設定
        x_min, x_max = X.min(), X.max()
        x_range = x_max - x_min
        plt.xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
        
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

    def plot_continuous_regression(
        self,
        X,
        y,
        y_pred,
        feature_name: str,
        target_name: str,
        r2: float,
        mse: float,
        outliers_removed: int = 0
    ) -> None:
        """
        連続値回帰の結果を可視化します。
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y : array-like
            実際の値
        y_pred : array-like
            予測値
        feature_name : str
            説明変数の名前
        target_name : str
            目的変数の名前
        r2 : float
            決定係数
        mse : float
            平均二乗誤差
        outliers_removed : int
            除去された外れ値の数
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 散布図
        plt.scatter(X, y, alpha=0.6, s=30, color='blue', label='実データ')
        
        # 回帰直線
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=3, 
                label=f'回帰直線 (R²={r2:.3f})')
        
        # 軸ラベルとタイトル
        plt.xlabel(feature_name.split('（')[0], fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        
        title_text = f'{feature_name}と{target_name}の関係（線形回帰）'
        if outliers_removed > 0:
            title_text += f'\n外れ値除去: {outliers_removed}件'
        title_text += f'\nR² = {r2:.3f}, MSE = {mse:.4f}'
        
        plt.title(title_text, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Y軸を0-1に制限（確率の場合）
        if target_name in ['勝率', 'win_rate', '複勝率', 'place_rate']:
            plt.ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # プロットを保存
        safe_target_name = target_name.replace('/', '_')
        self.save_plot(fig, f"{feature_name}_{safe_target_name}_continuous_regression.png")

    def plot_logistic_regression(
        self,
        X,
        y,
        y_pred_proba,
        y_pred,
        feature_name: str,
        title: str = "ロジスティック回帰分析",
        threshold_info: dict = None
    ) -> None:
        """
        改善されたロジスティック回帰分析の結果を2つの別々の図として可視化します。
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
        threshold_info : dict, optional
            閾値情報
        """
        # 予測曲線の可視化
        self.plot_logistic_regression_curve(
            X=X,
            y=y,
            y_pred_proba=y_pred_proba,
            feature_name=feature_name,
            title=title,
            threshold_info=threshold_info
        )
        
        # 混同行列と分類レポートの可視化
        self.plot_confusion_matrix(
            y=y,
            y_pred=y_pred,
            feature_name=feature_name,
            title=title
        ) 
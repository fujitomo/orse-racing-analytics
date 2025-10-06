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
        self.output_dir = Path(output_dir)  # Pathオブジェクトに変換
        self._setup_style()
        self.fig_size = (15, 10)

    def _setup_style(self) -> None:
        """プロットスタイルの設定"""
        from horse_racing.utils.font_config import setup_japanese_fonts, apply_plot_style
        setup_japanese_fonts(suppress_warnings=True)
        apply_plot_style(fig_size=(15, 10))
        
        # 追加のカスタマイズ
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
            c=data["主戦クラス"] if "主戦クラス" in data.columns else None,
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
        if "主戦クラス" in data.columns:
            cbar = plt.colorbar(scatter)
            cbar.ax.set_title("")

            # 1つ目：タイトル（カラーバーの右外側、縦書き）
            cbar.ax.text(
                1.1, 0.5,
                "最も出走回数が多いクラス",
                va='center', ha='left', fontsize=12, fontweight='bold', rotation=90, transform=cbar.ax.transAxes
            )
            # 2つ目：例示（さらに右外側、縦書き・重ならないように間隔を広げる）
            cbar.ax.text(
                2, 0.5,  # さらに右へ
                "※例：G1: 3回、G2: 5回、G3: 2回 → 主戦クラスはG2",
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

    def _visualize_correlations(self, horse_stats: pd.DataFrame, correlation_stats: Dict[str, Any]) -> None:
        """
        相関分析の結果を可視化します。
        """
        # 最高レベルと勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_win_max"],
            model=correlation_stats["model_win_max"],
            r2=correlation_stats["r2_win_max"],
            feature_name="最高競走経験質指数（REQI）と勝率",
            x_column="最高レベル",
            y_column="win_rate"
        )

        # 最高レベルと複勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_place_max"],
            model=correlation_stats["model_place_max"],
            r2=correlation_stats["r2_place_max"],
            feature_name="最高競走経験質指数（REQI）と複勝率",
            x_column="最高レベル",
            y_column="place_rate"
        )

        # 平均レベルと勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_win_avg"],
            model=correlation_stats["model_win_avg"],
            r2=correlation_stats["r2_win_avg"],
            feature_name="平均競走経験質指数（REQI）と勝率",
            x_column="平均レベル",
            y_column="win_rate"
        )

        # 平均レベルと複勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_place_avg"],
            model=correlation_stats["model_place_avg"],
            r2=correlation_stats["r2_place_avg"],
            feature_name="平均競走経験質指数（REQI）と複勝率",
            x_column="平均レベル",
            y_column="place_rate"
        )

        # 最高場所レベルと複勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_place_venue_max"],
            model=correlation_stats["model_place_venue_max"],
            r2=correlation_stats["r2_place_venue_max"],
            feature_name="最高場所レベルと複勝率",
            x_column="最高場所レベル",
            y_column="place_rate"
        )

        # 平均場所レベルと複勝率の相関
        self.plot_correlation_analysis(
            data=horse_stats,
            correlation=correlation_stats["correlation_place_venue_avg"],
            model=correlation_stats["model_place_venue_avg"],
            r2=correlation_stats["r2_place_venue_avg"],
            feature_name="平均場所レベルと複勝率",
            x_column="平均場所レベル",
            y_column="place_rate"
        )

        # 馬ごとの統計データを使った箱ひげ図分析
        # 主戦クラス別の分析（カラムが存在する場合のみ）
        if '主戦クラス' in horse_stats.columns:
            self.plot_boxplot_analysis(
                data=horse_stats,
                groupby_column='主戦クラス',
                value_columns=['平均レベル', '最高レベル', 'win_rate', 'place_rate'],
                title_prefix="馬の主戦クラス別分析（馬統計）",
                filename_prefix="horse_main_class_boxplot"
            )
        else:
            logger.info("📊 '主戦クラス'カラムが存在しないため、主戦クラス別分析をスキップします")

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

    def plot_boxplot_analysis(
        self,
        data: pd.DataFrame,
        groupby_column: str,
        value_columns: list,
        title_prefix: str = "箱ひげ図分析",
        filename_prefix: str = "boxplot"
    ) -> None:
        """
        箱ひげ図による分析結果の可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            分析対象データ
        groupby_column : str
            グループ化するカラム名
        value_columns : list
            分析対象の値カラム名のリスト
        title_prefix : str
            図のタイトルのプレフィックス
        filename_prefix : str
            保存ファイル名のプレフィックス
        """
        for value_col in value_columns:
            if value_col not in data.columns:
                continue
                
            # データの準備
            plot_data = data[[groupby_column, value_col]].dropna()
            if len(plot_data) == 0:
                continue
            
            # 図の作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # 左側: 基本的な箱ひげ図
            sns.boxplot(data=plot_data, x=groupby_column, y=value_col, ax=ax1)
            ax1.set_title(f'{title_prefix}: {value_col}の分布')
            ax1.set_xlabel(groupby_column)
            ax1.set_ylabel(value_col)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 右側: バイオリンプロット（分布の詳細確認）
            sns.violinplot(data=plot_data, x=groupby_column, y=value_col, ax=ax2)
            ax2.set_title(f'{title_prefix}: {value_col}の分布密度')
            ax2.set_xlabel(groupby_column)
            ax2.set_ylabel(value_col)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 統計情報の追加
            group_stats = plot_data.groupby(groupby_column)[value_col].agg(['count', 'mean', 'std', 'median'])
            stats_text = "統計情報:\n"
            for group, stats in group_stats.iterrows():
                stats_text += f"{group}: N={stats['count']}, 平均={stats['mean']:.3f}, 中央値={stats['median']:.3f}\n"
            
            # 統計情報をプロットに追加
            fig.text(0.02, 0.02, stats_text, fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    verticalalignment='bottom')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # 統計情報の表示スペースを確保
            
            # ファイル保存
            safe_value_col = value_col.replace('/', '_').replace(' ', '_')
            safe_groupby_col = groupby_column.replace('/', '_').replace(' ', '_')
            filename = f"{filename_prefix}_{safe_groupby_col}_{safe_value_col}.png"
            self.save_plot(fig, filename)

    def plot_race_grade_distance_boxplot(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        レース格別・距離別の箱ひげ図分析
        論文の「レース格別・距離別の基本統計量の比較」要求に対応
        
        Parameters:
        -----------
        df : pd.DataFrame
            レースデータ（馬単位ではなくレース単位）
        """
        # 距離カテゴリの作成
        df_analysis = df.copy()
        
        # 距離カテゴリ化
        def categorize_distance(distance):
            if distance <= 1400:
                return "短距離(≤1400m)"
            elif distance <= 1800:
                return "マイル(1401-1800m)"
            elif distance <= 2000:
                return "中距離(1801-2000m)"
            elif distance <= 2400:
                return "中長距離(2001-2400m)"
            else:
                return "長距離(≥2401m)"
        
        df_analysis['距離カテゴリ'] = df_analysis['距離'].apply(categorize_distance)
        
        # レース格のカテゴリ化（数値からテキストへ）
        grade_mapping = {
            1: "G1", 2: "G2", 3: "G3", 4: "重賞", 
            5: "特別戦", 6: "L", 99: "その他"
        }
        
        # クラスカラムを特定
        class_column = None
        for col in ['クラス', 'クラスコード', '条件']:
            if col in df_analysis.columns:
                class_column = col
                break
        
        if class_column:
            df_analysis['レース格'] = df_analysis[class_column].map(grade_mapping).fillna("その他")
        else:
            # クラス情報がない場合は競走経験質指数（REQI）で代用
            def level_to_grade(level):
                if level >= 8.5:
                    return "G1相当"
                elif level >= 7.5:
                    return "G2相当"
                elif level >= 6.5:
                    return "G3相当"
                elif level >= 5.5:
                    return "重賞相当"
                else:
                    return "一般戦"
            df_analysis['レース格'] = df_analysis['race_level'].apply(level_to_grade)
        
        # 成績カラムの作成
        df_analysis['複勝'] = (df_analysis['着順'] <= 3).astype(int)
        df_analysis['勝利'] = (df_analysis['着順'] == 1).astype(int)
        
        # 1. レース格別の箱ひげ図
        self.plot_boxplot_analysis(
            data=df_analysis,
            groupby_column='レース格',
            value_columns=['race_level'],
            title_prefix="レース格別分析",
            filename_prefix="grade_boxplot"
        )
        
        # 2. 距離カテゴリ別の箱ひげ図
        self.plot_boxplot_analysis(
            data=df_analysis,
            groupby_column='距離カテゴリ',
            value_columns=['race_level'],
            title_prefix="距離カテゴリ別分析",
            filename_prefix="distance_boxplot"
        )
        
        # 3. レース格×距離の組み合わせ分析
        # データ量を考慮して主要な組み合わせのみ
        major_grades = ["G1", "G2", "G3", "重賞", "特別戦"]
        major_distances = ["短距離(≤1400m)", "マイル(1401-1800m)", "中距離(1801-2000m)"]
        
        filtered_data = df_analysis[
            (df_analysis['レース格'].isin(major_grades)) & 
            (df_analysis['距離カテゴリ'].isin(major_distances))
        ]
        
        if len(filtered_data) > 0:
            # レース格×距離の組み合わせカラムを作成
            filtered_data['格×距離'] = filtered_data['レース格'] + " × " + filtered_data['距離カテゴリ']
            
            self.plot_boxplot_analysis(
                data=filtered_data,
                groupby_column='格×距離',
                value_columns=['race_level'],
                title_prefix="レース格×距離組み合わせ分析",
                filename_prefix="grade_distance_boxplot"
            )
        
        # 4. 馬単位での成績分析（馬ごとの統計を計算）
        horse_stats = df_analysis.groupby('馬名').agg({
            'race_level': ['mean', 'max'],
            '複勝': 'mean',
            '勝利': 'mean',
            'レース格': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "その他",
            '距離カテゴリ': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "その他"
        }).reset_index()
        
        # カラム名を平坦化
        horse_stats.columns = ['馬名', '平均レベル', '最高レベル', '複勝率', '勝率', '主戦格', '主戦距離']
        
        # 5. 馬ごとの主戦格別分析
        self.plot_boxplot_analysis(
            data=horse_stats,
            groupby_column='主戦格',
            value_columns=['平均レベル', '最高レベル', '複勝率', '勝率'],
            title_prefix="馬の主戦格別分析",
            filename_prefix="horse_grade_boxplot"
        )
        
        # 6. 馬ごとの主戦距離別分析
        self.plot_boxplot_analysis(
            data=horse_stats,
            groupby_column='主戦距離',
            value_columns=['平均レベル', '最高レベル', '複勝率', '勝率'],
            title_prefix="馬の主戦距離別分析",
            filename_prefix="horse_distance_boxplot"
        ) 
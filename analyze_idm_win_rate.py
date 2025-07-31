#!/usr/bin/env python
"""
IDMと勝率の相関分析を行うスクリプト
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import japanize_matplotlib
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging
import os
import glob
from datetime import datetime

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def calculate_horse_idm_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとのIDMと成績の統計を計算
    """
    try:
        # IDMの欠損値を処理
        df = df.copy()
        df['IDM'] = pd.to_numeric(df['IDM'], errors='coerce')
        logger.info(f"IDM欠損値数: {df['IDM'].isna().sum()}")
        
        # IDMが欠損していないデータのみを使用
        df = df.dropna(subset=['IDM'])
        logger.info(f"有効なIDMデータ数: {len(df)}")
        
        # 基本的な統計量を計算
        stats_df = df.groupby('馬名').agg({
            'IDM': ['mean', 'std', 'count'],
            '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        
        # カラム名の設定
        stats_df.columns = ['馬名', 'IDM平均', 'IDM標準偏差', 'IDM計測回数',
                          'レース数', '1着回数', '複勝回数']
        
        # 勝率と複勝率の計算
        stats_df['勝率'] = stats_df['1着回数'] / stats_df['レース数']
        stats_df['複勝率'] = stats_df['複勝回数'] / stats_df['レース数']
        
        # IDMの基本統計情報を表示
        print("\n=== IDMの基本統計 ===")
        print(f"平均: {stats_df['IDM平均'].mean():.2f}")
        print(f"標準偏差: {stats_df['IDM平均'].std():.2f}")
        print(f"最小値: {stats_df['IDM平均'].min():.2f}")
        print(f"最大値: {stats_df['IDM平均'].max():.2f}")
        
        return stats_df
        
    except Exception as e:
        logger.error(f"統計計算中にエラーが発生しました: {str(e)}")
        raise

def create_idm_win_rate_visualization(stats_df: pd.DataFrame, output_path: Path, min_races: int = 3, period_suffix: str = '') -> dict:
    """
    IDMと勝率・複勝率の関係を可視化（改善版）
    
    Returns
    -------
    dict
        分析結果の統計情報
    """
    try:
        # レース数でフィルタリング
        filtered_df = stats_df[stats_df['レース数'] >= min_races].copy()
        logger.info(f"分析対象馬数: {len(filtered_df)}")
        
        if len(filtered_df) < 2:
            logger.error(f"分析対象データが不足しています（{len(filtered_df)}件）")
            return {}
        
        # 勝率と複勝率の相関分析
        win_correlation, win_p_value = stats.pearsonr(filtered_df['IDM平均'], filtered_df['勝率'])
        place_correlation, place_p_value = stats.pearsonr(filtered_df['IDM平均'], filtered_df['複勝率'])
        
        # 勝率と複勝率の回帰分析
        X = filtered_df['IDM平均'].values.reshape(-1, 1)
        y_win = filtered_df['勝率'].values
        y_place = filtered_df['複勝率'].values
        
        reg_win = LinearRegression().fit(X, y_win)
        reg_place = LinearRegression().fit(X, y_place)
        r2_win = reg_win.score(X, y_win)
        r2_place = reg_place.score(X, y_place)
        
        # サブプロットの作成（2x2レイアウト）
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        
        # 共通の背景設定
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('IDM平均')
        
        # 勝率の散布図（左上）
        ax1.set_ylabel('勝率')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        sizes = np.interp(filtered_df['レース数'], 
                         (filtered_df['レース数'].min(), filtered_df['レース数'].max()), 
                         (20, 100))
        scatter1 = ax1.scatter(filtered_df['IDM平均'], filtered_df['勝率'],
                             s=sizes, c=filtered_df['レース数'],
                             cmap='viridis', alpha=0.3)
        
        plt.colorbar(scatter1, ax=ax1, label='レース数')
        
        # 勝率の回帰直線
        x_range = np.linspace(filtered_df['IDM平均'].min(), filtered_df['IDM平均'].max(), 100)
        y_pred_win = reg_win.predict(x_range.reshape(-1, 1))
        ax1.plot(x_range, y_pred_win, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2_win:.3f})')
        
        # 勝率の移動平均線
        window_size = min(50, len(filtered_df) // 4)
        if window_size >= 3:
            moving_avg_df = filtered_df[['IDM平均', '勝率']].sort_values('IDM平均')
            rolling = moving_avg_df.rolling(window=window_size, center=True, min_periods=1)
            rolling_mean = rolling.mean()
            
            ax1.plot(rolling_mean['IDM平均'], rolling_mean['勝率'],
                    color='green', label=f'移動平均（窓幅: {window_size}）')
        
        ax1.set_title('IDM平均と勝率の散布図')
        ax1.legend()
        
        # 複勝率の散布図（右上）
        ax2.set_ylabel('複勝率')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        scatter2 = ax2.scatter(filtered_df['IDM平均'], filtered_df['複勝率'],
                             s=sizes, c=filtered_df['レース数'],
                             cmap='plasma', alpha=0.3)
        
        plt.colorbar(scatter2, ax=ax2, label='レース数')
        
        # 複勝率の回帰直線
        y_pred_place = reg_place.predict(x_range.reshape(-1, 1))
        ax2.plot(x_range, y_pred_place, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2_place:.3f})')
        
        # 複勝率の移動平均線
        if window_size >= 3:
            moving_avg_df_place = filtered_df[['IDM平均', '複勝率']].sort_values('IDM平均')
            rolling_place = moving_avg_df_place.rolling(window=window_size, center=True, min_periods=1)
            rolling_mean_place = rolling_place.mean()
            
            ax2.plot(rolling_mean_place['IDM平均'], rolling_mean_place['複勝率'],
                    color='green', label=f'移動平均（窓幅: {window_size}）')
        
        ax2.set_title('IDM平均と複勝率の散布図')
        ax2.legend()
        
        # 勝率の六角形ビンプロット（左下）
        ax3.set_ylabel('勝率')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        hb1 = ax3.hexbin(filtered_df['IDM平均'], filtered_df['勝率'],
                       gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb1, ax=ax3, label='データ密度')
        
        ax3.plot(x_range, y_pred_win, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2_win:.3f})')
        
        ax3.set_title('IDM平均と勝率の密度プロット')
        ax3.legend()
        
        # 複勝率の六角形ビンプロット（右下）
        ax4.set_ylabel('複勝率')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        hb2 = ax4.hexbin(filtered_df['IDM平均'], filtered_df['複勝率'],
                       gridsize=30, cmap='YlGnBu', mincnt=1)
        plt.colorbar(hb2, ax=ax4, label='データ密度')
        
        ax4.plot(x_range, y_pred_place, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2_place:.3f})')
        
        ax4.set_title('IDM平均と複勝率の密度プロット')
        ax4.legend()
        
        # 全体のタイトル
        title_suffix = f" ({period_suffix})" if period_suffix else ""
        fig.suptitle(f'IDM平均と勝率・複勝率の関係{title_suffix}\n'
                    f'勝率相関: {win_correlation:.3f} | 複勝率相関: {place_correlation:.3f}',
                    fontsize=16, y=0.98)
        
        # レイアウトの調整
        plt.tight_layout()
        
        # グラフの保存
        filename = f'idm_win_rate_correlation{period_suffix}.png'
        output_file = output_path / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相関分析グラフを保存しました: {output_file}")
        
        # 統計情報の返却
        result_stats = {
            'win_correlation': win_correlation,
            'win_p_value': win_p_value,
            'win_r2': r2_win,
            'win_coefficient': reg_win.coef_[0],
            'win_intercept': reg_win.intercept_,
            'place_correlation': place_correlation,
            'place_p_value': place_p_value,
            'place_r2': r2_place,
            'place_coefficient': reg_place.coef_[0],
            'place_intercept': reg_place.intercept_,
            'sample_size': len(filtered_df)
        }
        
        # 詳細な統計情報の表示
        title_suffix = f" ({period_suffix})" if period_suffix else ""
        print(f"\n=== IDMと勝率・複勝率の相関分析{title_suffix} ===")
        print(f"勝率相関係数: {win_correlation:.3f} (p値: {win_p_value:.3e})")
        print(f"勝率決定係数 (R²): {r2_win:.3f}")
        print(f"勝率回帰係数: {reg_win.coef_[0]:.3f}")
        print(f"勝率切片: {reg_win.intercept_:.3f}")
        print()
        print(f"複勝率相関係数: {place_correlation:.3f} (p値: {place_p_value:.3e})")
        print(f"複勝率決定係数 (R²): {r2_place:.3f}")
        print(f"複勝率回帰係数: {reg_place.coef_[0]:.3f}")
        print(f"複勝率切片: {reg_place.intercept_:.3f}")
        print(f"サンプル数: {len(filtered_df)}")
        
        # レース数による層別分析
        print(f"\n=== レース数による層別分析{title_suffix} ===")
        race_bins = [3, 5, 10, 20, float('inf')]
        prev_bin = min_races
        for race_bin in race_bins:
            mask = (filtered_df['レース数'] >= prev_bin) & (filtered_df['レース数'] < race_bin)
            group_data = filtered_df[mask]
            
            if len(group_data) >= 2:
                try:
                    bin_win_corr, bin_win_p = stats.pearsonr(group_data['IDM平均'], group_data['勝率'])
                    bin_place_corr, bin_place_p = stats.pearsonr(group_data['IDM平均'], group_data['複勝率'])
                    print(f"レース数 {prev_bin}-{race_bin if race_bin != float('inf') else '∞'}:")
                    print(f"  データ数: {len(group_data)}")
                    print(f"  勝率相関: {bin_win_corr:.3f} (p値: {bin_win_p:.3e})")
                    print(f"  複勝率相関: {bin_place_corr:.3f} (p値: {bin_place_p:.3e})")
                except Exception as e:
                    logger.warning(f"レース数 {prev_bin}-{race_bin} の分析でエラーが発生: {str(e)}")
            else:
                logger.warning(f"レース数 {prev_bin}-{race_bin} のデータが不足しています（{len(group_data)}件）")
            
            prev_bin = race_bin
        
        return result_stats
        
    except Exception as e:
        logger.error(f"可視化の作成中にエラーが発生しました: {str(e)}")
        raise

def analyze_by_periods(df: pd.DataFrame, output_path: Path, min_races: int = 3) -> dict:
    """
    3年間隔での期間別分析を実行
    
    Parameters
    ----------
    df : pd.DataFrame
        分析対象データ
    output_path : Path
        出力ディレクトリのパス
    min_races : int
        最小レース数
        
    Returns
    -------
    dict
        期間別分析結果
    """
    try:
        # 年情報の抽出
        if '年' in df.columns:
            df['年'] = pd.to_numeric(df['年'], errors='coerce')
        else:
            logger.error("'年'カラムが見つかりません")
            return {}
        
        # 年の範囲を取得
        min_year = int(df['年'].min())
        max_year = int(df['年'].max())
        logger.info(f"データ年範囲: {min_year}年 - {max_year}年")
        
        # 3年間隔で期間を設定
        periods = []
        start_year = min_year
        while start_year <= max_year:
            end_year = min(start_year + 2, max_year)
            periods.append((start_year, end_year))
            start_year += 3
        
        period_results = {}
        
        print(f"\n{'='*60}")
        print(f"3年間隔分析を開始します: {len(periods)}期間")
        print(f"{'='*60}")
        
        for i, (start_year, end_year) in enumerate(periods, 1):
            period_name = f"{start_year}-{end_year}"
            print(f"\n期間 {i}/{len(periods)}: {period_name}年")
            
            # 期間でフィルタリング
            period_df = df[(df['年'] >= start_year) & (df['年'] <= end_year)].copy()
            
            if len(period_df) == 0:
                logger.warning(f"期間 {period_name} にデータがありません")
                continue
                
            logger.info(f"期間 {period_name}: {len(period_df)}件のデータ")
            
            try:
                # IDMと成績の統計計算
                stats_df = calculate_horse_idm_stats(period_df)
                
                if len(stats_df) == 0:
                    logger.warning(f"期間 {period_name}: 統計データが不足しています")
                    continue
                
                # 可視化と分析の実行
                period_suffix = f"_{period_name}"
                result_stats = create_idm_win_rate_visualization(stats_df, output_path, min_races, period_suffix)
                
                if result_stats:
                    result_stats['period'] = period_name
                    result_stats['data_count'] = len(period_df)
                    result_stats['horse_count'] = len(stats_df)
                    period_results[period_name] = result_stats
                    
                    # 上位馬の表示
                    print(f"\n=== IDM平均上位馬 ({period_name}年、最小レース数: {min_races}) ===")
                    top_horses = stats_df[stats_df['レース数'] >= min_races].sort_values('IDM平均', ascending=False).head(5)
                    if not top_horses.empty:
                        print(top_horses[['馬名', 'IDM平均', 'IDM標準偏差', 'レース数', '勝率', '複勝率']].to_string())
                    else:
                        print("該当する馬がいません")
                        
            except Exception as e:
                logger.error(f"期間 {period_name} の分析中にエラーが発生: {str(e)}")
                continue
        
        return period_results
        
    except Exception as e:
        logger.error(f"期間別分析中にエラーが発生しました: {str(e)}")
        raise

def generate_period_summary_report(period_results: dict, output_path: Path) -> None:
    """
    期間別分析の総合レポートを生成
    
    Parameters
    ----------
    period_results : dict
        期間別分析結果
    output_path : Path
        出力ディレクトリのパス
    """
    try:
        if not period_results:
            logger.warning("期間別分析結果がありません")
            return
            
        # レポートファイルの作成
        report_file = output_path / 'idm_correlation_period_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("IDMと勝率・複勝率の相関分析 - 期間別総合レポート\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 期間別結果一覧（勝率）
            f.write("期間別分析結果一覧（勝率）:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'期間':<12} {'相関係数':<8} {'R²':<8} {'p値':<12} {'サンプル数':<8}\n")
            f.write("-" * 60 + "\n")
            
            for period, stats in sorted(period_results.items()):
                f.write(f"{period:<12} {stats['win_correlation']:>7.3f} {stats['win_r2']:>7.3f} {stats['win_p_value']:>11.3e} {stats['sample_size']:>8}\n")
            
            # 期間別結果一覧（複勝率）
            f.write(f"\n期間別分析結果一覧（複勝率）:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'期間':<12} {'相関係数':<8} {'R²':<8} {'p値':<12} {'サンプル数':<8}\n")
            f.write("-" * 60 + "\n")
            
            for period, stats in sorted(period_results.items()):
                f.write(f"{period:<12} {stats['place_correlation']:>7.3f} {stats['place_r2']:>7.3f} {stats['place_p_value']:>11.3e} {stats['sample_size']:>8}\n")
            
            # 統計サマリー
            win_correlations = [stats['win_correlation'] for stats in period_results.values()]
            win_r2_values = [stats['win_r2'] for stats in period_results.values()]
            place_correlations = [stats['place_correlation'] for stats in period_results.values()]
            place_r2_values = [stats['place_r2'] for stats in period_results.values()]
            
            f.write(f"\n統計サマリー:\n")
            f.write("-" * 30 + "\n")
            f.write(f"勝率相関係数 - 平均: {np.mean(win_correlations):.3f}, 標準偏差: {np.std(win_correlations):.3f}\n")
            f.write(f"勝率決定係数 (R²) - 平均: {np.mean(win_r2_values):.3f}, 標準偏差: {np.std(win_r2_values):.3f}\n")
            f.write(f"勝率最高相関: {max(win_correlations):.3f}, 勝率最低相関: {min(win_correlations):.3f}\n")
            f.write(f"\n")
            f.write(f"複勝率相関係数 - 平均: {np.mean(place_correlations):.3f}, 標準偏差: {np.std(place_correlations):.3f}\n")
            f.write(f"複勝率決定係数 (R²) - 平均: {np.mean(place_r2_values):.3f}, 標準偏差: {np.std(place_r2_values):.3f}\n")
            f.write(f"複勝率最高相関: {max(place_correlations):.3f}, 複勝率最低相関: {min(place_correlations):.3f}\n")
            
            # 期間別詳細
            f.write(f"\n期間別詳細分析:\n")
            f.write("=" * 40 + "\n")
            
            for period, stats in sorted(period_results.items()):
                f.write(f"\n期間: {period}年\n")
                f.write("-" * 20 + "\n")
                f.write(f"データ数: {stats['data_count']:,}件\n")
                f.write(f"馬数: {stats['horse_count']:,}頭\n")
                f.write(f"\n【勝率分析結果】\n")
                f.write(f"勝率相関係数: {stats['win_correlation']:.3f}\n")
                f.write(f"勝率決定係数 (R²): {stats['win_r2']:.3f}\n")
                f.write(f"勝率p値: {stats['win_p_value']:.3e}\n")
                f.write(f"勝率回帰係数: {stats['win_coefficient']:.3f}\n")
                f.write(f"勝率切片: {stats['win_intercept']:.3f}\n")
                f.write(f"\n【複勝率分析結果】\n")
                f.write(f"複勝率相関係数: {stats['place_correlation']:.3f}\n")
                f.write(f"複勝率決定係数 (R²): {stats['place_r2']:.3f}\n")
                f.write(f"複勝率p値: {stats['place_p_value']:.3e}\n")
                f.write(f"複勝率回帰係数: {stats['place_coefficient']:.3f}\n")
                f.write(f"複勝率切片: {stats['place_intercept']:.3f}\n")
        
        logger.info(f"期間別総合レポートを保存しました: {report_file}")
        
        # コンソール出力
        print(f"\n{'='*60}")
        print("期間別分析 総合サマリー")
        print("="*60)
        print(f"分析期間数: {len(period_results)}")
        print(f"\n【勝率相関分析】")
        print(f"勝率相関係数 - 平均: {np.mean(win_correlations):.3f} ± {np.std(win_correlations):.3f}")
        print(f"勝率決定係数 - 平均: {np.mean(win_r2_values):.3f} ± {np.std(win_r2_values):.3f}")
        print(f"最高相関期間: {max(period_results.items(), key=lambda x: x[1]['win_correlation'])[0]} ({max(win_correlations):.3f})")
        print(f"最低相関期間: {min(period_results.items(), key=lambda x: x[1]['win_correlation'])[0]} ({min(win_correlations):.3f})")
        
        print(f"\n【複勝率相関分析】")
        print(f"複勝率相関係数 - 平均: {np.mean(place_correlations):.3f} ± {np.std(place_correlations):.3f}")
        print(f"複勝率決定係数 - 平均: {np.mean(place_r2_values):.3f} ± {np.std(place_r2_values):.3f}")
        print(f"最高相関期間: {max(period_results.items(), key=lambda x: x[1]['place_correlation'])[0]} ({max(place_correlations):.3f})")
        print(f"最低相関期間: {min(period_results.items(), key=lambda x: x[1]['place_correlation'])[0]} ({min(place_correlations):.3f})")
        
    except Exception as e:
        logger.error(f"レポート生成中にエラーが発生しました: {str(e)}")
        raise

def analyze_idm_win_rate(input_path: str, output_dir: str = 'export/analysis', min_races: int = 3, three_year_periods: bool = False) -> None:
    """
    IDMと勝率の相関分析を実行
    
    Parameters
    ----------
    input_path : str
        入力CSVファイルのパスまたはディレクトリパス
    output_dir : str
        出力ディレクトリのパス
    min_races : int
        最小レース数（これ未満のデータは除外）
    three_year_periods : bool
        3年間隔分析を実行するかどうか
    """
    try:
        # 出力ディレクトリの作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # データの読み込み
        if os.path.isdir(input_path):
            # SEDで始まるCSVファイルのみを検索
            csv_files = glob.glob(os.path.join(input_path, "**", "SED*.csv"), recursive=True)
            if not csv_files:
                raise ValueError(f"{input_path} にSEDファイルが見つかりませんでした。")
            
            df_list = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    df_list.append(df)
                    logger.info(f"読み込み成功: {file}")
                except Exception as e:
                    logger.warning(f"警告: {file} の読み込みに失敗しました: {str(e)}")
            
            if not df_list:
                raise ValueError("有効なSEDファイルが見つかりませんでした。")
            
            df = pd.concat(df_list, ignore_index=True)
            logger.info(f"合計 {len(csv_files)} 件のSEDファイルを読み込みました。")
        else:
            # 単一のCSVファイルを読み込む
            if not os.path.basename(input_path).startswith('SED'):
                raise ValueError("指定されたファイルはSEDファイルではありません。")
            df = pd.read_csv(input_path)
            logger.info(f"読み込み成功: {input_path}")
        
        # 重複データの削除
        df = df.drop_duplicates()
        logger.info(f"重複削除後のデータ数: {len(df)}")
        
        if three_year_periods:
            # 3年間隔分析
            period_results = analyze_by_periods(df, output_path, min_races)
            
            # 期間別総合レポートの生成
            if period_results:
                generate_period_summary_report(period_results, output_path)
            else:
                logger.warning("期間別分析結果がありません")
        else:
            # 全期間分析（デフォルト）
            # IDMと成績の統計計算
            stats_df = calculate_horse_idm_stats(df)
            
            # 可視化と分析の実行
            create_idm_win_rate_visualization(stats_df, output_path, min_races)
            
            # 上位馬の表示
            print("\n=== IDM平均上位馬（最小レース数: {}） ===".format(min_races))
            top_horses = stats_df[stats_df['レース数'] >= min_races].sort_values('IDM平均', ascending=False).head(10)
            print(top_horses[['馬名', 'IDM平均', 'IDM標準偏差', 'レース数', '勝率', '複勝率']].to_string())
        
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='IDMと勝率の相関分析を行います')
    parser.add_argument('input_path', help='入力CSVファイルのパスまたはディレクトリパス')
    parser.add_argument('--output-dir', default='export/analysis', 
                       help='出力ディレクトリのパス（デフォルト: export/analysis）')
    parser.add_argument('--min-races', type=int, default=3,
                       help='最小レース数（これ未満のデータは除外、デフォルト: 3）')
    parser.add_argument('--three-year-periods', action='store_true',
                       help='3年間隔での期間別分析を実行（デフォルト: 全期間分析）')
    
    args = parser.parse_args()
    analyze_idm_win_rate(args.input_path, args.output_dir, args.min_races, args.three_year_periods) 
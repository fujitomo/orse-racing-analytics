#!/usr/bin/env python
"""
IDMと勝率の相関分析を行うスクリプト
"""

import pandas as pd
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

def create_idm_win_rate_visualization(stats_df: pd.DataFrame, output_path: Path, min_races: int = 3) -> None:
    """
    IDMと勝率の関係を可視化（改善版）
    """
    try:
        # レース数でフィルタリング
        filtered_df = stats_df[stats_df['レース数'] >= min_races].copy()
        logger.info(f"分析対象馬数: {len(filtered_df)}")
        
        if len(filtered_df) < 2:
            logger.error(f"分析対象データが不足しています（{len(filtered_df)}件）")
            return
        
        # 相関分析
        correlation, p_value = stats.pearsonr(filtered_df['IDM平均'], filtered_df['勝率'])
        
        # 回帰分析
        X = filtered_df['IDM平均'].values.reshape(-1, 1)
        y = filtered_df['勝率'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        # サブプロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 共通の背景設定
        for ax in [ax1, ax2]:
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('IDM平均')
            ax.set_ylabel('勝率')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # 散布図（左）
        # レース数に基づいて点の大きさとカラーを設定
        sizes = np.interp(filtered_df['レース数'], 
                         (filtered_df['レース数'].min(), filtered_df['レース数'].max()), 
                         (20, 100))  # サイズを小さく調整
        scatter = ax1.scatter(filtered_df['IDM平均'], filtered_df['勝率'],
                             s=sizes, c=filtered_df['レース数'],
                             cmap='viridis', alpha=0.3)  # 透明度を上げる
        
        # カラーバーの追加
        plt.colorbar(scatter, ax=ax1, label='レース数')
        
        # 回帰直線
        x_range = np.linspace(filtered_df['IDM平均'].min(), filtered_df['IDM平均'].max(), 100)
        y_pred = reg.predict(x_range.reshape(-1, 1))
        ax1.plot(x_range, y_pred, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2:.3f})')
        
        # 移動平均線の追加
        window_size = 50
        # 数値データのみを使用して移動平均を計算
        moving_avg_df = filtered_df[['IDM平均', '勝率']].sort_values('IDM平均')
        rolling = moving_avg_df.rolling(window=window_size, center=True, min_periods=1)
        rolling_mean = rolling.mean()
        
        ax1.plot(rolling_mean['IDM平均'], rolling_mean['勝率'],
                color='green', label=f'移動平均（窓幅: {window_size}）')
        
        ax1.set_title('散布図とトレンド')
        ax1.legend()
        
        # 六角形ビンプロット（右）
        hb = ax2.hexbin(filtered_df['IDM平均'], filtered_df['勝率'],
                       gridsize=30, cmap='YlOrRd',
                       mincnt=1)
        plt.colorbar(hb, ax=ax2, label='データ密度')
        
        # 回帰直線を六角形ビンプロットにも追加
        ax2.plot(x_range, y_pred, color='red', linestyle='--',
                label=f'回帰直線 (R² = {r2:.3f})')
        
        ax2.set_title('密度プロット')
        ax2.legend()
        
        # 全体のタイトル
        fig.suptitle(f'IDM平均と勝率の関係\n相関係数: {correlation:.3f} (p値: {p_value:.3e})',
                    fontsize=14, y=1.05)
        
        # レイアウトの調整
        plt.tight_layout()
        
        # グラフの保存
        output_file = output_path / 'idm_win_rate_correlation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相関分析グラフを保存しました: {output_file}")
        
        # 詳細な統計情報の表示
        print("\n=== IDMと勝率の相関分析 ===")
        print(f"相関係数: {correlation:.3f}")
        print(f"p値: {p_value:.3e}")
        print(f"決定係数 (R²): {r2:.3f}")
        print(f"回帰係数: {reg.coef_[0]:.3f}")
        print(f"切片: {reg.intercept_:.3f}")
        
        # レース数による層別分析
        print("\n=== レース数による層別分析 ===")
        race_bins = [3, 5, 10, 20, float('inf')]
        prev_bin = min_races
        for race_bin in race_bins:
            mask = (filtered_df['レース数'] >= prev_bin) & (filtered_df['レース数'] < race_bin)
            group_data = filtered_df[mask]
            
            if len(group_data) >= 2:  # 相関分析に必要な最小データ数をチェック
                try:
                    bin_corr, bin_p = stats.pearsonr(group_data['IDM平均'], group_data['勝率'])
                    print(f"レース数 {prev_bin}-{race_bin if race_bin != float('inf') else '∞'}:")
                    print(f"  データ数: {len(group_data)}")
                    print(f"  相関係数: {bin_corr:.3f}")
                    print(f"  p値: {bin_p:.3e}")
                except Exception as e:
                    logger.warning(f"レース数 {prev_bin}-{race_bin} の分析でエラーが発生: {str(e)}")
            else:
                logger.warning(f"レース数 {prev_bin}-{race_bin} のデータが不足しています（{len(group_data)}件）")
            
            prev_bin = race_bin
        
    except Exception as e:
        logger.error(f"可視化の作成中にエラーが発生しました: {str(e)}")
        raise

def analyze_idm_win_rate(input_path: str, output_dir: str = 'export/analysis', min_races: int = 3) -> None:
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
                       help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=3,
                       help='最小レース数（これ未満のデータは除外）')
    
    args = parser.parse_args()
    analyze_idm_win_rate(args.input_path, args.output_dir, args.min_races) 
#!/usr/bin/env python
"""
馬ごとの場コード別ポイント分析スクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import japanize_matplotlib
import numpy as np
from collections import defaultdict
import logging
import os
import glob
from scipy import stats
from sklearn.linear_model import LinearRegression

# ロガーの設定
logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ポイント計算の設定
POINT_SYSTEM = {
    1: 10,  # 1着: 10ポイント
    2: 5,   # 2着: 5ポイント
    3: 3,   # 3着: 3ポイント
    4: 1,   # 4着: 1ポイント
    5: 1    # 5着: 1ポイント
}

def calculate_horse_track_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごと・場コードごとのポイントを計算
    """
    # ポイントの計算
    df['points'] = df['着順'].map(lambda x: POINT_SYSTEM.get(x, 0))
    
    # 馬ごと・場コードごとの集計
    points_df = df.groupby(['馬名', '場コード']).agg({
        'points': ['sum', 'count'],
        '着順': lambda x: (x <= 3).sum()  # 複勝回数
    }).reset_index()
    
    # カラム名の設定
    points_df.columns = ['馬名', '場コード', '合計ポイント', 'レース数', '複勝回数']
    
    # 平均ポイントと複勝率の計算
    points_df['平均ポイント'] = points_df['合計ポイント'] / points_df['レース数']
    points_df['複勝率'] = points_df['複勝回数'] / points_df['レース数']
    
    return points_df

def create_horse_track_visualization(points_df: pd.DataFrame, output_path: Path, min_races: int = 3) -> None:
    """
    馬ごとの場コード別成績を可視化
    """
    # レース数でフィルタリング
    filtered_df = points_df[points_df['レース数'] >= min_races]
    
    # 場コードごとの平均ポイントを計算
    track_avg = filtered_df.groupby('場コード')['平均ポイント'].mean().sort_values(ascending=False)
    
    # プロット作成
    plt.figure(figsize=(15, 10))
    
    # 背景色の設定
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 散布図（個々の馬のデータ）
    sns.stripplot(data=filtered_df, x='場コード', y='平均ポイント',
                 order=track_avg.index, color='red', alpha=0.3, size=4)
    
    # グラフの設定
    plt.title(f'場コード別の馬のポイント分布\n（最小レース数: {min_races}）')
    plt.xlabel('競馬場')
    plt.ylabel('平均ポイント')
    
    # X軸ラベルの回転
    plt.xticks(rotation=45)
    
    # レイアウトの調整
    plt.tight_layout()
    
    # グラフの保存
    output_file = output_path / 'horse_track_points_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"グラフを保存しました: {output_file}")

def analyze_win_rate_points_correlation(points_df: pd.DataFrame, output_path: Path, min_races: int = 3) -> None:
    """
    勝率とポイントの相関分析を行う
    """
    try:
        # レース数でフィルタリング
        filtered_df = points_df[points_df['レース数'] >= min_races].copy()
        
        # 勝率を計算（1着の回数/レース数）
        filtered_df['勝率'] = filtered_df['複勝回数'] / filtered_df['レース数']
        
        # 相関分析
        correlation, p_value = stats.pearsonr(filtered_df['勝率'], filtered_df['平均ポイント'])
        
        # 回帰分析
        X = filtered_df['勝率'].values.reshape(-1, 1)
        y = filtered_df['平均ポイント'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        # プロット作成
        plt.figure(figsize=(12, 8))
        
        # 背景色の設定
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 散布図
        sns.scatterplot(data=filtered_df, x='勝率', y='平均ポイント', 
                       alpha=0.5, color='blue')
        
        # 回帰直線
        x_range = np.linspace(filtered_df['勝率'].min(), filtered_df['勝率'].max(), 100)
        y_pred = reg.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_pred, color='red', linestyle='--', 
                label=f'回帰直線 (R² = {r2:.3f})')
        
        # グラフの設定
        plt.title(f'勝率と平均ポイントの関係\n相関係数: {correlation:.3f} (p値: {p_value:.3e})')
        plt.xlabel('勝率')
        plt.ylabel('平均ポイント')
        
        # 凡例の設定
        plt.legend()
        
        # レイアウトの調整
        plt.tight_layout()
        
        # グラフの保存
        output_file = output_path / 'win_rate_points_correlation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相関分析グラフを保存しました: {output_file}")
        
        # 詳細な統計情報の表示
        print("\n=== 勝率と平均ポイントの相関分析 ===")
        print(f"相関係数: {correlation:.3f}")
        print(f"p値: {p_value:.3e}")
        print(f"決定係数 (R²): {r2:.3f}")
        print(f"回帰係数: {reg.coef_[0]:.3f}")
        print(f"切片: {reg.intercept_:.3f}")
        
    except Exception as e:
        logger.error(f"相関分析中にエラーが発生しました: {str(e)}")
        raise

def analyze_horse_track_points(input_path: str, output_dir: str = 'export/analysis', min_races: int = 3) -> None:
    """
    馬ごとの場コード別ポイント分析を実行
    
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
        
        # ポイントの計算
        points_df = calculate_horse_track_points(df)
        
        # 可視化の作成
        create_horse_track_visualization(points_df, output_path, min_races)
        
        # 勝率とポイントの相関分析
        analyze_win_rate_points_correlation(points_df, output_path, min_races)
        
        # 結果の表示
        display_results(points_df, min_races)
        
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {str(e)}")
        raise

def display_results(points_df: pd.DataFrame, min_races: int) -> None:
    """
    分析結果の表示
    """
    # レース数でフィルタリング
    filtered_df = points_df[points_df['レース数'] >= min_races]
    
    print(f"\n=== 場コード別の成績分析（最小レース数: {min_races}） ===")
    
    # 場コードごとの集計
    track_stats = filtered_df.groupby('場コード').agg({
        '平均ポイント': ['mean', 'std', 'count'],
        '複勝率': 'mean'
    }).round(3)
    
    # カラム名の設定
    track_stats.columns = ['平均ポイント', '標準偏差', '該当馬数', '平均複勝率']
    
    # 結果の表示
    print("\n【場コード別統計】")
    print(track_stats.sort_values('平均ポイント', ascending=False))
    
    # 最高成績の馬を表示
    print("\n【最高平均ポイントの馬（上位5頭）】")
    top_horses = filtered_df.sort_values('平均ポイント', ascending=False).head()
    print(top_horses[['馬名', '場コード', '平均ポイント', 'レース数', '複勝率']].to_string())

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='馬ごとの場コード別ポイント分析を行います')
    parser.add_argument('input_path', help='入力CSVファイルのパスまたはディレクトリパス')
    parser.add_argument('--output-dir', default='export/analysis', 
                       help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=3,
                       help='最小レース数（これ未満のデータは除外）')
    
    args = parser.parse_args()
    analyze_horse_track_points(args.input_path, args.output_dir, args.min_races) 
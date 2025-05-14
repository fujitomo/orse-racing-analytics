#!/usr/bin/env python
"""
競馬場コードと勝率の相関分析を行うスクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import japanize_matplotlib
import glob
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 競馬場レベル定義
TRACK_LEVELS = {
    'レベル1': ['東京', '中山', '阪神', '京都', '札幌'],
    'レベル2': ['中京', '函館', '新潟'],
    'レベル3': ['福島', '小倉']
}

# レベルごとの色定義
LEVEL_COLORS = {
    'レベル1': '#3498db',  # 青
    'レベル2': '#e67e22',  # オレンジ
    'レベル3': '#2ecc71'   # 緑
}

def get_track_level(track_name):
    """競馬場のレベルを取得"""
    for level, tracks in TRACK_LEVELS.items():
        if track_name in tracks:
            return level
    return 'その他'

def format_percentage(value):
    """勝率を百分率形式にフォーマット"""
    return f'{value * 100:.1f}%'

def analyze_track_win_rate(input_path: str, output_dir: str = 'export/analysis', min_races: int = 3) -> None:
    """
    場コードと勝率の相関分析を実行する

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

        # データの読み込みと前処理
        df = load_and_preprocess_data(input_path)
        
        # レース回数でフィルタリング
        df = filter_by_min_races(df, min_races)
        
        # 統計情報の計算
        track_stats = calculate_track_stats(df)
        
        # 相関分析
        correlation_stats = calculate_correlation_stats(df)
        
        # 可視化の作成
        create_visualization(df, track_stats, correlation_stats, output_path)
        
        # 結果の表示
        display_results(track_stats, correlation_stats)

    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {str(e)}")
        raise

def filter_by_min_races(df: pd.DataFrame, min_races: int) -> pd.DataFrame:
    """最小レース数でフィルタリング"""
    # 馬ごとのレース回数を計算
    race_counts = df.groupby('馬名')['着順'].count()
    
    # 最小レース数以上の馬を抽出
    qualified_horses = race_counts[race_counts >= min_races].index
    
    # フィルタリングを適用
    return df[df['馬名'].isin(qualified_horses)].copy()

def load_and_preprocess_data(input_path: str) -> pd.DataFrame:
    """データの読み込みと前処理"""
    try:
        # データの読み込み
        if os.path.isdir(input_path):
            csv_files = glob.glob(os.path.join(input_path, "*.csv"))
            if not csv_files:
                raise ValueError(f"{input_path} にCSVファイルが見つかりませんでした。")
            df_list = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    df_list.append(df)
                    logger.info(f"読み込み成功: {file}")
                except Exception as e:
                    logger.warning(f"警告: {file} の読み込みに失敗しました: {str(e)}")
            if not df_list:
                raise ValueError("有効なCSVファイルが見つかりませんでした。")
            df = pd.concat(df_list, ignore_index=True)
        else:
            try:
                df = pd.read_csv(input_path)
                logger.info(f"読み込み成功: {input_path}")
            except Exception as e:
                raise ValueError(f"ファイルの読み込みに失敗しました: {str(e)}")

        # 基本的な前処理
        df['is_win'] = (df['着順'] == 1).astype(float)  # intからfloatに変更
        df['is_placed'] = (df['着順'] <= 3).astype(float)  # intからfloatに変更
        df['レベル'] = df['場コード'].map(get_track_level)

        return df

    except Exception as e:
        logger.error(f"データの読み込みと前処理中にエラーが発生しました: {str(e)}")
        raise

def calculate_track_stats(df: pd.DataFrame) -> pd.DataFrame:
    """競馬場ごとの統計情報を計算"""
    try:
        # 競馬場ごとの基本統計
        track_stats = df.groupby(['場コード', 'レベル']).agg({
            'is_win': ['count', 'mean'],
            'is_placed': 'mean'
        }).reset_index()

        # カラム名の設定
        track_stats.columns = ['場コード', 'レベル', 'レース数', '勝率', '複勝率']

        # パーセント表示用に変換
        track_stats['勝率_表示用'] = track_stats['勝率'].apply(format_percentage)
        track_stats['複勝率_表示用'] = track_stats['複勝率'].apply(format_percentage)

        return track_stats.sort_values(['レベル', '勝率'], ascending=[True, False])

    except Exception as e:
        logger.error(f"統計情報の計算中にエラーが発生しました: {str(e)}")
        raise

def calculate_correlation_stats(df: pd.DataFrame) -> dict:
    """相関分析を実行"""
    try:
        # 場コードを数値化
        df['場コード_数値'] = pd.Categorical(df['場コード']).codes

        # 相関係数の計算
        correlation = df['場コード_数値'].corr(df['is_win'])

        # 回帰分析
        X = df['場コード_数値'].values.reshape(-1, 1)
        y = df['is_win'].values
        reg = LinearRegression().fit(X, y)
        r2 = r2_score(y, reg.predict(X))

        return {
            'correlation': correlation,
            'r2': r2,
            'regression_model': reg,
            'X': X
        }

    except Exception as e:
        logger.error(f"相関分析中にエラーが発生しました: {str(e)}")
        raise

def create_visualization(df: pd.DataFrame, track_stats: pd.DataFrame, 
                       correlation_stats: dict, output_path: Path) -> None:
    """分析結果の可視化"""
    try:
        plt.figure(figsize=(15, 10))
        
        # 背景色とグリッドの設定
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 全体の背景色設定
        plt.gcf().patch.set_facecolor('#ffffff')

        # レベルごとにプロット
        for level, color in LEVEL_COLORS.items():
            level_data = df[df['レベル'] == level]
            
            # ジッターの設定を調整（X軸のみ）
            x_jitter = np.random.normal(0, 0.1, size=len(level_data))
            x_pos = pd.Categorical(level_data['場コード']).codes + x_jitter
            
            # データポイントをプロット
            plt.scatter(x_pos, level_data['is_win'], 
                       alpha=0.3,  # 透明度を上げる
                       color=color, 
                       label=f'{level}',
                       s=30)  # ポイントサイズを小さく

        # 回帰直線
        X = correlation_stats['X']
        y_pred = correlation_stats['regression_model'].predict(X)
        plt.plot(np.unique(X), np.unique(y_pred), 
                color='red', 
                linestyle='--', 
                alpha=0.5, 
                label='回帰直線')

        # 軸の設定
        plt.xlabel('競馬場')
        plt.ylabel('勝率')
        
        # Y軸の範囲と目盛りの設定
        plt.ylim(0.0, 0.15)  # 実際のデータ範囲に合わせて調整
        yticks = np.arange(0, 0.16, 0.02)  # 0.02刻みで設定
        plt.yticks(yticks)
        
        # Y軸の目盛りラベルを小数点表示に変更
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # X軸のラベル設定
        unique_tracks = df['場コード'].unique()
        plt.xticks(range(len(unique_tracks)), unique_tracks, rotation=45)

        # グラフタイトルの設定
        plt.title('競馬場コードと勝率の関係', pad=20)

        # 凡例の設定
        plt.legend(title='競馬場レベル', 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left')

        # 相関係数とR2スコアの表示
        correlation_text = f'相関係数: {correlation_stats["correlation"]:.3f}\nR2スコア: {correlation_stats["r2"]:.3f}'
        plt.text(1.05, 0.5, correlation_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # レイアウトの調整
        plt.tight_layout()

        # グラフの保存
        output_file = output_path / 'track_win_rate_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"グラフを保存しました: {output_file}")

    except Exception as e:
        logger.error(f"可視化の作成中にエラーが発生しました: {str(e)}")
        raise

def display_results(track_stats: pd.DataFrame, correlation_stats: dict) -> None:
    """結果の表示"""
    try:
        print(f"\n分析が完了しました。")
        print(f"\n競馬場コードと勝率の相関係数: {correlation_stats['correlation']:.3f}")
        print(f"決定係数 (R²): {correlation_stats['r2']:.3f}")
        print("\n各競馬場の統計（レベル別）:")
        
        # レベル別に統計情報を表示
        for level in TRACK_LEVELS.keys():
            print(f"\n{level}:")
            level_stats = track_stats[track_stats['レベル'] == level]
            print(level_stats[['場コード', 'レース数', '勝率_表示用']].to_string(index=False))

    except Exception as e:
        logger.error(f"結果の表示中にエラーが発生しました: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='競馬場コードと勝率の相関分析を行います')
    parser.add_argument('input_path', help='入力CSVファイルのパスまたはディレクトリパス')
    parser.add_argument('--output-dir', default='export/analysis', 
                       help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=3,
                       help='最小レース数（これ未満のデータは除外）')
    
    args = parser.parse_args()
    analyze_track_win_rate(args.input_path, args.output_dir, args.min_races) 
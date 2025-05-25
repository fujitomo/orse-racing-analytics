#!/usr/bin/env python
"""
馬の脚質とトラックバイアスの相関分析スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIを使わない非対話式バックエンド
# その後で他のインポートを行う
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# 機械学習関連のインポート追加
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoostがインストールされていません。XGBoost分析は実行できません。")

import japanize_matplotlib
import os
import argparse
from pathlib import Path
import statsmodels.api as sm
from datetime import datetime
import re
from sklearn.cluster import KMeans

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def load_data(input_path):
    """データの読み込み"""
    # ファイルの存在確認
    if not os.path.exists(input_path):
        raise ValueError(f"指定されたファイル '{input_path}' が存在しません。正しいパスを指定してください。")
        
    # ディレクトリ指定の場合の処理
    if os.path.isdir(input_path):
        # 最新のCSVファイルを探す
        csv_files = sorted([f for f in os.listdir(input_path) if f.endswith('_formatted.csv')], reverse=True)
        if not csv_files:
            raise ValueError(f"'{input_path}' ディレクトリ内に '_formatted.csv' で終わるCSVファイルが見つかりません。")
        
        print(f"\nディレクトリ内のCSVファイル:")
        for i, file in enumerate(csv_files[:5], 1):  # 最新5件を表示
            print(f"{i}. {file}")
        if len(csv_files) > 5:
            print(f"... 他 {len(csv_files) - 5} 件")
        
        # すべてのファイルを読み込んで結合
        df_list = []
        for file in csv_files:
            file_path = os.path.join(input_path, file)
            try:
                # 試行するエンコーディングのリスト
                encodings = ['shift_jis', 'utf-8', 'cp932', 'euc-jp']
                
                # 各エンコーディングで試行
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        df_list.append(df)
                        print(f"読み込み成功: {file} ({encoding})")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"警告: {file} の読み込みに失敗しました: {str(e)}")
                        break
            except Exception as e:
                print(f"警告: {file} の処理中にエラーが発生しました: {str(e)}")
                continue
        
        if not df_list:
            raise ValueError("有効なCSVファイルが見つかりませんでした。")
        
        # データフレームを結合
        df = pd.concat(df_list, ignore_index=True)
        print(f"\n合計 {len(df)} 行のデータを読み込みました")
        return df
    else:
        # 単一ファイルを読み込む
        encodings = ['shift_jis', 'utf-8', 'cp932', 'euc-jp']
        
        # 各エンコーディングで試行
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                print(f"ファイルを {encoding} エンコーディングで読み込みました")
                print(f"合計 {len(df)} 行のデータを読み込みました")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue
        
        raise ValueError("ファイルを読み込めませんでした。サポートされているエンコーディングで試行しましたが、すべて失敗しました。")

def get_favorable_styles_by_course():
    """競馬場・芝ダート別の有利脚質マスタを返す"""
    # 競馬場・芝ダート別の有利脚質リスト（ユーザー提供データ）
    favorable_styles = {
        ('東京', '芝'): ['差し', '追い込み'],
        ('中山', '芝'): ['逃げ', '先行'],
        ('京都', '芝'): ['逃げ', '先行'],
        ('阪神', '芝', '内'): ['先行'],
        ('阪神', '芝', '外'): ['差し'],
        ('新潟', '芝'): ['追い込み'],
        ('札幌', '芝'): ['先行', '逃げ'],
        ('函館', '芝'): ['逃げ', '先行'],
        ('福島', '芝'): ['逃げ', '先行'],
        ('小倉', '芝'): ['逃げ', '先行'],
        ('中京', '芝'): ['差し', '追い込み'],
        # ダートは便宜上先行有利としておく（未提供）
        ('東京', 'ダート'): ['逃げ', '先行'],
        ('中山', 'ダート'): ['逃げ', '先行'],
        ('京都', 'ダート'): ['逃げ', '先行'],
        ('阪神', 'ダート'): ['逃げ', '先行'],
        ('新潟', 'ダート'): ['逃げ', '先行'],
        ('札幌', 'ダート'): ['逃げ', '先行'],
        ('函館', 'ダート'): ['逃げ', '先行'],
        ('福島', 'ダート'): ['逃げ', '先行'],
        ('小倉', 'ダート'): ['逃げ', '先行'],
        ('中京', 'ダート'): ['逃げ', '先行'],
    }
    return favorable_styles

def preprocess_data(df):
    """データの前処理"""
    # レース脚質カラムのみを使用
    if 'レース脚質' not in df.columns:
        raise ValueError('レース脚質カラムが見つかりません')
    running_style_col = 'レース脚質'

    # 着順カラム名の特定
    rank_col = None
    for col in ['着順', '順位', '着']:
        if col in df.columns:
            rank_col = col
            break
    if rank_col is None:
        raise ValueError('着順カラムが見つかりません')
    
    # 場コードから競馬場名への変換
    track_map = {
        1: '札幌',
        2: '函館',
        3: '福島',
        4: '新潟',
        5: '東京',
        6: '中山',
        7: '中京',
        8: '京都',
        9: '阪神',
        10: '小倉'
    }
    
    # 競馬場カラムの作成（場コードから変換、またはそのまま利用）
    if '競馬場' not in df.columns:
        if '場コード' in df.columns:
            def code_to_track(x):
                try:
                    # 数値ならマッピング
                    return track_map.get(int(float(str(x))), '不明')
                except (ValueError, TypeError):
                    # 文字列ならそのまま（例: 既に「福島」など）
                    s = str(x).strip()
                    if s in track_map.values():
                        return s
                    return '不明'
            df['競馬場'] = df['場コード'].map(code_to_track)
            print(f"場コードから競馬場情報を生成しました")
        else:
            raise ValueError('競馬場情報（場コード）が見つかりません')
    
    # 芝・ダート判定
    if '芝ダ障害コード' in df.columns:
        df['芝ダート'] = df['芝ダ障害コード'].map(lambda x: '芝' if str(x).startswith('芝') or str(x) == '1' else 'ダート')
    elif 'トラック種別' in df.columns:
        df['芝ダート'] = df['トラック種別']
    else:
        raise ValueError('芝ダート判定に必要なカラムが見つかりません')
    
    # コース（内・外）情報があれば利用
    has_course_info = 'コース' in df.columns
    
    # 数値型または数値文字列の脚質を名称に変換
    style_map = {
        1: '逃げ',
        2: '先行',
        3: '差し',
        4: '追い込み',
        5: '好位差し',
        6: '自在'
    }
    def style_code_to_name(x):
        try:
            x_int = int(float(str(x).strip()))
            return style_map.get(x_int, f'不明({x})')
        except (ValueError, TypeError):
            return x
    
    # 脚質名称の統一
    df[running_style_col] = df[running_style_col].apply(style_code_to_name)
    
    # 有利脚質の判定（競馬場・芝ダート別）
    favorable_styles_master = get_favorable_styles_by_course()
    
    def get_favorable_styles(row):
        """レースごとの有利脚質を判定"""
        track = row['競馬場']
        surface = row['芝ダート']
        
        # 阪神の内・外の判定
        if track == '阪神' and surface == '芝' and has_course_info:
            course_type = '内' if '内' in str(row['コース']) else '外'
            key = (track, surface, course_type)
            return favorable_styles_master.get(key, [])
        
        # 通常の競馬場・芝ダート判定
        key = (track, surface)
        return favorable_styles_master.get(key, [])
    
    # 各レースで有利脚質リストを判定
    df['有利脚質リスト'] = df.apply(get_favorable_styles, axis=1)
    
    # 各レースで馬の脚質が有利脚質に含まれるか判定
    df['有利脚質該当'] = df.apply(lambda row: row[running_style_col] in row['有利脚質リスト'], axis=1)
    
    return df, running_style_col, rank_col

def analyze_running_style(df, running_style_col, rank_col, output_dir):
    """脚質分析と結果の出力"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 脚質ごとの勝率
    style_stats = df.copy()
    style_stats['勝利'] = style_stats[rank_col] == 1
    style_stats['好成績'] = style_stats[rank_col].isin([1, 2, 3])  # 1-3着を好成績とする
    
    # 勝率・連対率・3着内率を計算
    style_summary = style_stats.groupby(running_style_col).agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum')
    )
    style_summary['勝率'] = style_summary['勝利数'] / style_summary['レース数']
    style_summary['好成績率'] = style_summary['好成績数'] / style_summary['レース数']
    style_summary = style_summary.sort_values('勝率', ascending=False)
    style_summary.to_csv(f"{output_dir}/running_style_stats.csv", encoding='utf-8')
    
    # 馬別の成績集計
    horse_stats = style_stats.groupby('馬名').agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum'),
        有利脚質該当数=('有利脚質該当', 'sum')
    )
    
    # 馬ごとの勝率・好成績率・有利脚質該当率を計算
    horse_stats['勝率'] = horse_stats['勝利数'] / horse_stats['レース数']
    horse_stats['好成績率'] = horse_stats['好成績数'] / horse_stats['レース数']
    horse_stats['有利脚質該当率'] = horse_stats['有利脚質該当数'] / horse_stats['レース数']
    horse_stats = horse_stats.sort_values('勝率', ascending=False)
    
    # 出走回数による絞り込み（2回以上出走した馬のみ）
    valid_horse_stats = horse_stats[horse_stats['レース数'] >= 6].copy()
    
    if len(valid_horse_stats) < 10:
        print(f"注意: {2}回以上出走した馬が{len(valid_horse_stats)}頭しか見つかりませんでした。十分なデータがあることを確認してください。")
    
    # 馬ごとの脚質出現回数集計
    valid_horse_style = style_stats.groupby(['馬名', running_style_col]).agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum')
    ).reset_index()
    valid_horse_style['勝率'] = valid_horse_style['勝利数'] / valid_horse_style['レース数']
    valid_horse_style['好成績率'] = valid_horse_style['好成績数'] / valid_horse_style['レース数']
    
    # 馬ごとに最頻脚質（または好成績脚質）を得意脚質とする
    # その脚質での勝率・好成績率・レース数も持たせる
    best_styles = valid_horse_style.sort_values(['馬名', 'レース数', '好成績率', '勝率'], ascending=[True, False, False, False])
    best_styles = best_styles.groupby('馬名').first().reset_index()
    horse_best_style = best_styles[['馬名', running_style_col, '勝率', '好成績率', 'レース数']].copy()
    horse_best_style = horse_best_style.rename(columns={running_style_col: '得意脚質'})
    
    # 相関分析用のデータ作成
    horse_analysis = valid_horse_stats.copy().reset_index()
    
    print(f"相関分析対象馬: {len(horse_analysis)}頭")
    
    # ===== 新規追加: 詳細条件ごとの分析 =====
    print("\n詳細条件別の脚質と勝率の分析を実行...")
    
    # 1. 距離帯の定義
    if '距離' in df.columns:
        print("距離帯別の分析を実行中...")
        df['距離帯'] = pd.cut(
            df['距離'].astype(float), 
            bins=[0, 1400, 2000, 9999],
            labels=['短距離', '中距離', '長距離']
        )
        # style_statsにも距離帯カラムをコピー
        style_stats['距離帯'] = df['距離帯']
        analyze_by_condition(df, style_stats, running_style_col, '距離帯', output_dir)
    else:
        print("警告: 距離カラムがないため、距離帯別分析をスキップします")
    
    # 2. 馬場状態別の分析
    if '馬場状態' in df.columns:
        print("馬場状態別の分析を実行中...")
        # style_statsに馬場状態をコピー
        style_stats['馬場状態'] = df['馬場状態']
        analyze_by_condition(df, style_stats, running_style_col, '馬場状態', output_dir)
    else:
        # 代替カラムを探す
        alt_columns = ['芝馬場状態', '馬場', '芝馬場']
        found = False
        for col in alt_columns:
            if col in df.columns:
                print(f"馬場状態別の分析を {col} カラムで実行中...")
                style_stats[col] = df[col]  # style_statsにコピー
                analyze_by_condition(df, style_stats, running_style_col, col, output_dir)
                found = True
                break
        if not found:
            print("警告: 馬場状態カラムがないため、馬場状態別分析をスキップします")
    
    # 3. コース形状（内/外）別の分析
    if 'コース' in df.columns:
        print("コース形状別の分析を実行中...")
        # 「内」「外」を含むかどうかで判定
        df['コース形状'] = df['コース'].apply(
            lambda x: '内回り' if '内' in str(x) else ('外回り' if '外' in str(x) else '不明')
        )
        # style_statsにコース形状をコピー
        style_stats['コース形状'] = df['コース形状']
        analyze_by_condition(df, style_stats, running_style_col, 'コース形状', output_dir)
    else:
        print("警告: コース情報がないため、内/外回り別分析をスキップします")
    
    # 4. 季節別の分析（開催月から季節を判定）
    if '年月日' in df.columns or '日付' in df.columns:
        date_col = '年月日' if '年月日' in df.columns else '日付'
        print(f"季節別の分析を {date_col} カラムで実行中...")
        
        try:
            # 日付から月を抽出
            df['月'] = df[date_col].astype(str).str.extract(r'(\d{1,2})月')
            
            # 月から季節を判定
            def get_season(month):
                try:
                    month = int(month)
                    if month in [3, 4, 5]:
                        return '春'
                    elif month in [6, 7, 8]:
                        return '夏'
                    elif month in [9, 10, 11]:
                        return '秋'
                    else:
                        return '冬'
                except:
                    return '不明'
            
            df['季節'] = df['月'].apply(get_season)
            # style_statsに月と季節をコピー
            style_stats['月'] = df['月']
            style_stats['季節'] = df['季節']
            analyze_by_condition(df, style_stats, running_style_col, '季節', output_dir)
        except Exception as e:
            print(f"警告: 季節判定でエラーが発生しました: {e}")
            
    # 5. 月別分析も追加
    if '月' in df.columns:
        if '月' not in style_stats.columns:
            style_stats['月'] = df['月']
        analyze_by_condition(df, style_stats, running_style_col, '月', output_dir)
    
    # 勝率と有利脚質該当率の相関を分析
    x = horse_analysis['有利脚質該当率'].values
    y = horse_analysis['勝率'].values
    
    if len(x) > 1 and len(y) > 1:
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        # スピアマンの順位相関係数（ノンパラメトリック）
        spearman_corr, p_value = stats.spearmanr(x, y)
        
        print(f"有利脚質該当率と勝率の相関係数: r = {correlation:.4f}")
        print(f"決定係数 R^2: {r_squared:.4f}")
        print(f"スピアマン順位相関係数: rho = {spearman_corr:.4f}, p値 = {p_value:.4f}")
        
        # 線形回帰
        X = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # プロット作成
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(x, y, c=horse_analysis['レース数'], cmap='Blues', 
                            s=horse_analysis['レース数'] * 2, alpha=0.6)
        plt.plot(x, predictions, color='red', linewidth=2)
        plt.xlabel('有利脚質該当率')
        plt.ylabel('勝率')
        plt.title(f'有利脚質該当率と勝率の関係\n(r={correlation:.4f}, p={p_value:.4f})')
        plt.colorbar(scatter, label='レース数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/favorable_style_win_rate.png", dpi=200)
        plt.close()
        
        # 複勝率（好成績率）と有利脚質該当率の相関を分析（新規追加）
        y_fukusho = horse_analysis['好成績率'].values
        
        if len(x) > 1 and len(y_fukusho) > 1:
            # 相関係数と統計量
            correlation_fukusho = np.corrcoef(x, y_fukusho)[0, 1]
            r_squared_fukusho = correlation_fukusho ** 2
            spearman_corr_fukusho, p_value_fukusho = stats.spearmanr(x, y_fukusho)
            
            print(f"\n有利脚質該当率と複勝率の相関係数: r = {correlation_fukusho:.4f}")
            print(f"決定係数 R^2: {r_squared_fukusho:.4f}")
            print(f"スピアマン順位相関係数: rho = {spearman_corr_fukusho:.4f}, p値 = {p_value_fukusho:.4f}")
            
            # 線形回帰（複勝率用）
            X = x.reshape(-1, 1)
            model_fukusho = LinearRegression()
            model_fukusho.fit(X, y_fukusho)
            predictions_fukusho = model_fukusho.predict(X)
            
            # 複勝率のプロット作成
            plt.figure(figsize=(10, 7))
            scatter_fukusho = plt.scatter(x, y_fukusho, c=horse_analysis['レース数'], cmap='Greens', 
                                       s=horse_analysis['レース数'] * 2, alpha=0.6)
            plt.plot(x, predictions_fukusho, color='red', linewidth=2)
            plt.xlabel('有利脚質該当率')
            plt.ylabel('複勝率（1-3着率）')
            plt.title(f'有利脚質該当率と複勝率の関係\n(r={correlation_fukusho:.4f}, p={p_value_fukusho:.4f})')
            plt.colorbar(scatter_fukusho, label='レース数')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/favorable_style_fukusho_rate.png", dpi=200)
            plt.close()
            
            # 複合グラフ：勝率と複勝率を同時表示
            plt.figure(figsize=(12, 8))
            plt.scatter(x, y, c='blue', s=horse_analysis['レース数'] * 1.5, alpha=0.5, label='勝率')
            plt.plot(x, predictions, color='blue', linewidth=2, linestyle='-', label='勝率回帰線')
            plt.scatter(x, y_fukusho, c='green', s=horse_analysis['レース数'] * 1.5, alpha=0.5, label='複勝率')
            plt.plot(x, predictions_fukusho, color='green', linewidth=2, linestyle='-', label='複勝率回帰線')
            plt.xlabel('有利脚質該当率')
            plt.ylabel('勝率・複勝率')
            plt.title(f'有利脚質該当率と勝率・複勝率の関係\n(勝率r={correlation:.4f}, 複勝率r={correlation_fukusho:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/favorable_style_win_fukusho_combined.png", dpi=200)
            plt.close()
            
            # =============== 芝のみの分析と箱ひげ図 (新規追加) ===============
            print("\n芝コースのみのデータで有利脚質該当率と勝率・複勝率の関係を分析中...")
            
            # 元データから芝のみのレースを抽出
            turf_races = df[df['芝ダート'] == '芝']
            print(f"芝レース数: {len(turf_races)}行 ({len(turf_races)/len(df)*100:.1f}%)")
            
            # 高速化のため、芝馬の分析は簡略化したアプローチを使用
            # 現在分析中の馬（horse_analysis）から、芝レース出走割合が高い馬を選ぶ
            print("芝出走馬を抽出中...")
            
            # 芝コースでの出走回数が一定以上の馬のみ対象
            min_turf_races = 5  # 最低出走回数
            
            # より効率的な抽出方法 - pandas集計関数を使用
            turf_horses = set()
            
            try:
                # まずレースIDと馬名と芝ダートのみの小さいデータフレームを作成
                if 'レースID' in df.columns:
                    race_horse_df = df[['レースID', '馬名', '芝ダート']].copy()
                    
                    # 馬ごとの芝レース数を集計
                    turf_races_by_horse = race_horse_df[race_horse_df['芝ダート'] == '芝'].groupby('馬名').size()
                    all_races_by_horse = race_horse_df.groupby('馬名').size()
                    
                    # 十分な数の芝レースに出走している馬を抽出
                    qualified_horses = turf_races_by_horse[turf_races_by_horse >= min_turf_races].index
                    turf_horses = set(qualified_horses)
                else:
                    # レースIDがない場合の代替手段
                    # 各馬について芝レースの割合を計算
                    # ここでは単純に芝レースに出走した回数を集計
                    horse_surface_counts = df.groupby(['馬名', '芝ダート']).size().unstack(fill_value=0)
                    
                    # 芝カラムがない場合は作成
                    if '芝' not in horse_surface_counts.columns:
                        horse_surface_counts['芝'] = 0
                    
                    # 芝レースの出走回数でフィルタリング
                    qualified_horses = horse_surface_counts[horse_surface_counts['芝'] >= min_turf_races].index
                    turf_horses = set(qualified_horses)
                
                print(f"芝{min_turf_races}回以上出走した馬: {len(turf_horses)}頭")
                
            except Exception as e:
                print(f"芝馬抽出でエラー: {e}")
                # フォールバック: 全馬を対象とする
                turf_horses = set(df['馬名'].unique())
                print("エラーのため全馬を分析対象とします")
            
            # horse_analysisから芝専門馬のみ抽出
            turf_horse_analysis = horse_analysis[horse_analysis['馬名'].isin(turf_horses)].copy()
            print(f"芝{min_turf_races}回以上出走した分析対象馬: {len(turf_horse_analysis)}頭")
            
            if len(turf_horse_analysis) >= 10:  # 十分なデータがある場合のみ分析
                # 芝馬の有利脚質該当率と勝率・複勝率の相関分析
                turf_x = turf_horse_analysis['有利脚質該当率'].values
                turf_y = turf_horse_analysis['勝率'].values
                turf_y_fukusho = turf_horse_analysis['好成績率'].values
                
                # 相関係数計算
                turf_correlation = np.corrcoef(turf_x, turf_y)[0, 1]
                turf_correlation_fukusho = np.corrcoef(turf_x, turf_y_fukusho)[0, 1]
                
                # スピアマン相関
                turf_spearman, turf_p = stats.spearmanr(turf_x, turf_y)
                turf_spearman_fukusho, turf_p_fukusho = stats.spearmanr(turf_x, turf_y_fukusho)
                
                print(f"芝のみ - 有利脚質該当率と勝率の相関: r = {turf_correlation:.4f}, p = {turf_p:.4f}")
                print(f"芝のみ - 有利脚質該当率と複勝率の相関: r = {turf_correlation_fukusho:.4f}, p = {turf_p_fukusho:.4f}")
                
                # 線形回帰
                turf_X = turf_x.reshape(-1, 1)
                turf_model = LinearRegression().fit(turf_X, turf_y)
                turf_model_fukusho = LinearRegression().fit(turf_X, turf_y_fukusho)
                
                turf_predictions = turf_model.predict(turf_X)
                turf_predictions_fukusho = turf_model_fukusho.predict(turf_X)
                
                # 芝のみの散布図（勝率）
                plt.figure(figsize=(10, 7))
                turf_scatter = plt.scatter(turf_x, turf_y, c=turf_horse_analysis['レース数'], 
                                         cmap='Blues', s=turf_horse_analysis['レース数'] * 2, alpha=0.6)
                plt.plot(turf_x, turf_predictions, color='red', linewidth=2)
                plt.xlabel('有利脚質該当率')
                plt.ylabel('勝率')
                plt.title(f'芝のみ: 有利脚質該当率と勝率の関係\n(r={turf_correlation:.4f}, p={turf_p:.4f})')
                plt.colorbar(turf_scatter, label='レース数')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/turf_favorable_style_win_rate.png", dpi=200)
                plt.close()
                
                # 芝のみの散布図（複勝率）
                plt.figure(figsize=(10, 7))
                turf_scatter_fukusho = plt.scatter(turf_x, turf_y_fukusho, c=turf_horse_analysis['レース数'], 
                                                cmap='Greens', s=turf_horse_analysis['レース数'] * 2, alpha=0.6)
                plt.plot(turf_x, turf_predictions_fukusho, color='red', linewidth=2)
                plt.xlabel('有利脚質該当率')
                plt.ylabel('複勝率（1-3着率）')
                plt.title(f'芝のみ: 有利脚質該当率と複勝率の関係\n(r={turf_correlation_fukusho:.4f}, p={turf_p_fukusho:.4f})')
                plt.colorbar(turf_scatter_fukusho, label='レース数')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/turf_favorable_style_fukusho_rate.png", dpi=200)
                plt.close()
                
                # 芝のみの複合グラフ
                plt.figure(figsize=(12, 8))
                plt.scatter(turf_x, turf_y, c='blue', s=turf_horse_analysis['レース数'] * 1.5, alpha=0.5, label='勝率')
                plt.plot(turf_x, turf_predictions, color='blue', linewidth=2, linestyle='-', label='勝率回帰線')
                plt.scatter(turf_x, turf_y_fukusho, c='green', s=turf_horse_analysis['レース数'] * 1.5, alpha=0.5, label='複勝率')
                plt.plot(turf_x, turf_predictions_fukusho, color='green', linewidth=2, linestyle='-', label='複勝率回帰線')
                plt.xlabel('有利脚質該当率')
                plt.ylabel('勝率・複勝率')
                plt.title(f'芝のみ: 有利脚質該当率と勝率・複勝率の関係\n(勝率r={turf_correlation:.4f}, 複勝率r={turf_correlation_fukusho:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/turf_favorable_style_win_fukusho_combined.png", dpi=200)
                plt.close()
                
                # ======== 箱ひげ図の作成 ========
                # 有利脚質該当率を5区間に分割
                turf_horse_analysis['有利脚質該当率区分'] = pd.cut(
                    turf_horse_analysis['有利脚質該当率'], 
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
                )
                
                # 勝率の箱ひげ図
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='有利脚質該当率区分', y='勝率', data=turf_horse_analysis, palette='Blues')
                plt.title('芝のみ: 有利脚質該当率区分ごとの勝率分布')
                plt.xlabel('有利脚質該当率')
                plt.ylabel('勝率')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/turf_favorable_style_win_rate_boxplot.png", dpi=200)
                plt.close()
                
                # 複勝率の箱ひげ図
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='有利脚質該当率区分', y='好成績率', data=turf_horse_analysis, palette='Greens')
                plt.title('芝のみ: 有利脚質該当率区分ごとの複勝率分布')
                plt.xlabel('有利脚質該当率')
                plt.ylabel('複勝率（1-3着率）')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/turf_favorable_style_fukusho_rate_boxplot.png", dpi=200)
                plt.close()
                
                # 区分ごとの統計量を計算して出力
                rate_group_stats = turf_horse_analysis.groupby('有利脚質該当率区分').agg({
                    '勝率': ['mean', 'median', 'std', 'count'],
                    '好成績率': ['mean', 'median', 'std', 'count']
                })
                
                print("\n芝のみ: 有利脚質該当率区分ごとの統計量")
                print(rate_group_stats)
                
                # 統計量をCSVに保存
                rate_group_stats.to_csv(f"{output_dir}/turf_favorable_style_rate_stats.csv", encoding='utf-8')
                
                # 追加: 有利脚質該当率の高低による統計的差異の検定
                high_rates = turf_horse_analysis[turf_horse_analysis['有利脚質該当率'] >= 0.6]['勝率']
                low_rates = turf_horse_analysis[turf_horse_analysis['有利脚質該当率'] < 0.4]['勝率']
                
                if len(high_rates) >= 5 and len(low_rates) >= 5:
                    t_stat, p_val = stats.ttest_ind(high_rates, low_rates, equal_var=False)
                    print(f"\n芝のみ: 有利脚質該当率高群(>=60%)と低群(<40%)の勝率差のt検定: t={t_stat:.3f}, p={p_val:.4f}")
                    
                    # 複勝率についても同様の検定
                    high_fukusho = turf_horse_analysis[turf_horse_analysis['有利脚質該当率'] >= 0.6]['好成績率']
                    low_fukusho = turf_horse_analysis[turf_horse_analysis['有利脚質該当率'] < 0.4]['好成績率']
                    t_stat_f, p_val_f = stats.ttest_ind(high_fukusho, low_fukusho, equal_var=False)
                    print(f"芝のみ: 有利脚質該当率高群(>=60%)と低群(<40%)の複勝率差のt検定: t={t_stat_f:.3f}, p={p_val_f:.4f}")
            else:
                print(f"芝{min_turf_races}回以上出走した馬が少なすぎるため、芝のみの分析をスキップします。")
        
        # ===== 新規追加：脚質ごとの勝率分析 =====
        print("\n脚質と勝率の関係分析を実行中...")
        
        # 各脚質の勝率集計
        style_win_rates = style_summary[['勝率']].reset_index()
        
        # 脚質と勝率の散布図（水平バープロット）
        plt.figure(figsize=(10, 6))
        plt.barh(style_win_rates[running_style_col], style_win_rates['勝率'], color='skyblue')
        plt.xlabel('勝率')
        plt.ylabel('脚質')
        plt.title('脚質別の平均勝率')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/running_style_win_rate_bar.png", dpi=200)
        plt.close()
        
        # 馬ごとの脚質利用率と勝率の関係分析
        # 馬ごとの各脚質使用率を計算
        horse_style_usage = pd.pivot_table(
            valid_horse_style,
            values='レース数',
            index='馬名',
            columns=running_style_col,
            aggfunc='sum',
            fill_value=0
        )
        
        # 各馬のレース総数で割って使用率を計算
        total_races = horse_style_usage.sum(axis=1)
        for style in horse_style_usage.columns:
            horse_style_usage[f'{style}_率'] = horse_style_usage[style] / total_races
            
        # 勝率と結合
        style_analysis = pd.merge(
            horse_analysis[['馬名', '勝率', 'レース数']],
            horse_style_usage,
            on='馬名',
            how='left'
        )
        
        # 脚質使用率と勝率の相関分析（使用率カラムのみ）
        usage_cols = [col for col in style_analysis.columns if col.endswith('_率')]
        correlation_results = []
        
        plt.figure(figsize=(15, 10))
        for i, style_col in enumerate(usage_cols):
            style_name = style_col.replace('_率', '')
            
            # 相関係数とp値を計算
            x_style = style_analysis[style_col].values
            y_style = style_analysis['勝率'].values
            mask = ~(np.isnan(x_style) | np.isnan(y_style))
            
            if sum(mask) > 10:  # 十分なデータがある場合のみ
                corr, p = stats.pearsonr(x_style[mask], y_style[mask])
                spearman, sp = stats.spearmanr(x_style[mask], y_style[mask])
                
                # 回帰直線
                X_style = x_style[mask].reshape(-1, 1)
                Y_style = y_style[mask]
                if len(X_style) > 1:
                    reg = LinearRegression().fit(X_style, Y_style)
                    pred = reg.predict(X_style)
                    
                    # 散布図と回帰直線（サブプロット）
                    plt.subplot(2, 3, i+1)
                    plt.scatter(x_style[mask], y_style[mask], 
                                c=style_analysis['レース数'].values[mask], 
                                cmap='viridis', alpha=0.6, s=style_analysis['レース数'].values[mask])
                    plt.plot(X_style, pred, 'r-', linewidth=2)
                    plt.xlabel(f'{style_name}使用率')
                    plt.ylabel('勝率')
                    plt.title(f'{style_name}使用率と勝率の関係\n(r={corr:.3f}, p={p:.4f})')
                    plt.grid(True, alpha=0.3)
                    
                    # 結果を保存
                    correlation_results.append({
                        '脚質': style_name,
                        'ピアソン相関係数': corr,
                        'p値': p,
                        'スピアマン順位相関': spearman,
                        'スピアマンp値': sp,
                        '回帰係数': reg.coef_[0],
                        '切片': reg.intercept_,
                        '決定係数': reg.score(X_style, Y_style)
                    })
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/style_win_rate_correlation.png", dpi=200)
        plt.close()
        
        # 相関結果の出力
        if correlation_results:
            corr_df = pd.DataFrame(correlation_results)
            corr_df = corr_df.sort_values('ピアソン相関係数', ascending=False)
            corr_df.to_csv(f"{output_dir}/style_win_rate_correlation.csv", index=False, encoding='utf-8')
            
            print("\n脚質使用率と勝率の相関分析結果:")
            print(corr_df[['脚質', 'ピアソン相関係数', 'p値', '決定係数']].to_string(index=False))
        
        # ロジスティック回帰分析
        print("\nロジスティック回帰分析を実行中...")
        logistic_success = False
        
        # 複数のしきい値設定を試行
        thresholds_to_try = [
            ('中央値', horse_analysis['勝率'].median()),
            ('平均値', horse_analysis['勝率'].mean()),
            ('75パーセンタイル', horse_analysis['勝率'].quantile(0.75)),
            ('25パーセンタイル', horse_analysis['勝率'].quantile(0.25)),
            ('固定値', 0.1)  # 競馬の標準的な勝率として10%を基準として採用
        ]
        
        for threshold_name, threshold_value in thresholds_to_try:
            # 勝率を2値に変換
            horse_analysis['高勝率'] = (horse_analysis['勝率'] >= threshold_value).astype(int)
            
            # クラスの分布を確認
            class_counts = horse_analysis['高勝率'].value_counts()
            class_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 1 else 0
            
            # 訓練データとテストデータに分割
            X = horse_analysis['有利脚質該当率'].values.reshape(-1, 1)
            y = horse_analysis['高勝率'].values
            
            if len(np.unique(y)) > 1 and class_ratio >= 0.1:  # クラスが2つ以上あり、少なくとも10%のサンプルが少数クラスにある場合
                print(f"\nしきい値モード: {threshold_name} = {threshold_value:.4f}")
                print(f"クラス分布: {dict(class_counts)}")
                
                try:
                    # 訓練・テスト分割の比率を調整（データが少ない場合は比率を変える）
                    test_size = 0.3 if len(y) >= 50 else 0.2
                    
                    # 層化サンプリングでクラスバランスを維持
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # ロジスティック回帰モデルのトレーニング（正則化強度を調整）
                    log_reg = LogisticRegression(class_weight='balanced', 
                                               solver='liblinear', 
                                               C=1.0)
                    log_reg.fit(X_train, y_train)
                    
                    # モデル評価
                    train_score = log_reg.score(X_train, y_train)
                    test_score = log_reg.score(X_test, y_test)
                    y_pred = log_reg.predict(X_test)
                    
                    print(f"ロジスティック回帰 - トレーニングスコア: {train_score:.4f}")
                    print(f"ロジスティック回帰 - テストスコア: {test_score:.4f}")
                    
                    # ROC曲線
                    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlabel('偽陽性率')
                    plt.ylabel('真陽性率')
                    plt.title(f'有利脚質該当率による高勝率予測のROC曲線\nしきい値: {threshold_name}={threshold_value:.3f}')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{output_dir}/logistic_regression_roc_{threshold_name}.png", dpi=200)
                    plt.close()
                    
                    # 混同行列
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('予測値')
                    plt.ylabel('実際の値')
                    plt.title(f'混同行列 (しきい値: {threshold_name}={threshold_value:.3f})')
                    plt.savefig(f"{output_dir}/confusion_matrix_{threshold_name}.png", dpi=200)
                    plt.close()
                    
                    # 分類レポート
                    report = classification_report(y_test, y_pred)
                    print("分類レポート:")
                    print(report)
                    
                    # statsmodelsを使った詳細な回帰分析
                    X_sm = sm.add_constant(X)  # 切片を追加
                    logit_model = sm.Logit(y, X_sm)
                    try:
                        result = logit_model.fit(disp=0)  # 詳細な出力を抑制
                        print(result.summary().tables[1])  # 係数情報のみ出力
                        
                        # 結果をCSVに保存
                        with open(f"{output_dir}/logistic_regression_{threshold_name}.txt", 'w', encoding='utf-8') as f:
                            f.write(f"しきい値モード: {threshold_name} = {threshold_value}\n\n")
                            f.write(f"クラス分布: {dict(class_counts)}\n\n")
                            f.write(f"トレーニングスコア: {train_score:.4f}\n")
                            f.write(f"テストスコア: {test_score:.4f}\n\n")
                            f.write("分類レポート:\n")
                            f.write(report)
                            f.write("\n\n回帰分析詳細:\n")
                            f.write(str(result.summary()))
                    except Exception as e:
                        print(f"statsmodelsでの解析エラー: {e}")
                    
                    logistic_success = True
                    break  # 成功したら他のしきい値は試さない
                    
                except Exception as e:
                    print(f"ロジスティック回帰分析でエラーが発生しました: {e}")
            else:
                if len(np.unique(y)) <= 1:
                    print(f"しきい値モード {threshold_name} = {threshold_value:.4f}: クラスが1つしかありません")
                else:
                    print(f"しきい値モード {threshold_name} = {threshold_value:.4f}: クラス比率が不均衡です ({class_ratio:.2f})")
                
        if not logistic_success:
            print("\n警告: すべてのしきい値設定でロジスティック回帰分析が実行できませんでした。")
            print("データのばらつきやサンプル数が不足している可能性があります。")
        
        # 複数の指標との相関プロット
        metrics = ['勝率', '好成績率']
        fig, ax = plt.subplots(1, len(metrics), figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            ax[i].scatter(horse_analysis['有利脚質該当率'], horse_analysis[metric], 
                         s=horse_analysis['レース数'], alpha=0.6)
            ax[i].set_xlabel('有利脚質該当率')
            ax[i].set_ylabel(metric)
            ax[i].set_title(f'有利脚質該当率と{metric}の関係')
            ax[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/favorable_style_multiple_metrics.png", dpi=200)
        plt.close()
        
        # 有利脚質該当・非該当の比較
        plt.figure(figsize=(10, 6))
        horse_analysis['有利脚質該当'] = (horse_analysis['有利脚質該当率'] > 0.5).astype(str)
        for metric in ['勝率', '好成績率']:
            sns.boxplot(x='有利脚質該当', y=metric, data=horse_analysis)
            plt.title(f'有利脚質該当別の{metric}分布')
            plt.savefig(f"{output_dir}/favorable_style_{metric}_boxplot.png", dpi=200)
            plt.close()
            
            # t検定
            group_true = horse_analysis[horse_analysis['有利脚質該当'] == 'True'][metric]
            group_false = horse_analysis[horse_analysis['有利脚質該当'] == 'False'][metric]
            
            if len(group_true) > 0 and len(group_false) > 0:
                t_stat, p_val = stats.ttest_ind(group_true, group_false, equal_var=False)
                print(f"{metric}の有利脚質該当/非該当グループ間のt検定: t={t_stat:.4f}, p={p_val:.4f}")
        
        return horse_analysis, correlation, p_value, model
    else:
        print("警告: 相関分析に十分なデータがありません")
        return horse_analysis, None, None, None

def analyze_by_condition(df, style_stats, running_style_col, condition_col, output_dir):
    """条件別の脚質と勝率の関係を分析"""
    os.makedirs(f"{output_dir}/conditions", exist_ok=True)
    
    # 条件ごとの脚質勝率を計算
    condition_stats = style_stats.groupby([condition_col, running_style_col]).agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum')
    ).reset_index()
    
    condition_stats['勝率'] = condition_stats['勝利数'] / condition_stats['レース数']
    condition_stats['好成績率'] = condition_stats['好成績数'] / condition_stats['レース数']
    
    # 結果をCSV出力
    condition_stats.to_csv(f"{output_dir}/conditions/{condition_col}_style_stats.csv", encoding='utf-8')
    
    # 十分なデータがある条件のみ可視化
    conditions = condition_stats[condition_col].unique()
    if len(conditions) > 1 and len(conditions) <= 12:  # 2～12条件なら可視化
        # 可視化：条件×脚質のヒートマップ
        pivot_data = condition_stats.pivot_table(
            values='勝率',
            index=condition_col, 
            columns=running_style_col,
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt='.3f', linewidths=.5)
        plt.title(f'{condition_col}別・脚質別の勝率ヒートマップ')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/conditions/{condition_col}_style_heatmap.png", dpi=200)
        plt.close()
        
        # 可視化：条件別の有利脚質該当と勝率の関係（散布図）
        fig, axes = plt.subplots(len(conditions), 1, figsize=(10, 4*len(conditions)))
        if len(conditions) == 1:
            axes = [axes]  # 1条件の場合はリストに変換
            
        for i, condition in enumerate(sorted(conditions)):
            condition_data = style_stats[style_stats[condition_col] == condition]
            
            # 馬ごとの集計
            horse_condition = condition_data.groupby('馬名').agg(
                レース数=('勝利', 'count'),
                勝利数=('勝利', 'sum'),
                好成績数=('好成績', 'sum'),
                有利脚質該当数=('有利脚質該当', 'sum')
            ).reset_index()
            
            horse_condition['勝率'] = horse_condition['勝利数'] / horse_condition['レース数']
            horse_condition['有利脚質該当率'] = horse_condition['有利脚質該当数'] / horse_condition['レース数']
            
            # レース数3以上に絞る
            valid_horses = horse_condition[horse_condition['レース数'] >= 3]
            
            if len(valid_horses) > 10:  # 十分なデータがある場合のみ
                # 散布図
                x = valid_horses['有利脚質該当率'].values
                y = valid_horses['勝率'].values
                
                # 相関係数を計算
                if len(x) > 2:
                    correlation, p_val = stats.pearsonr(x, y)
                    corr_text = f"r={correlation:.3f}, p={p_val:.4f}"
                else:
                    corr_text = "データ不足"
                
                # 散布図と回帰直線
                axes[i].scatter(x, y, c=valid_horses['レース数'], cmap='viridis', alpha=0.7, s=valid_horses['レース数']*2)
                
                if len(x) > 2:
                    # 回帰直線
                    X = x.reshape(-1, 1)
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    axes[i].plot(x, y_pred, 'r-', linewidth=2)
                
                axes[i].set_title(f'{condition_col}: {condition} ({corr_text})')
                axes[i].set_xlabel('有利脚質該当率')
                axes[i].set_ylabel('勝率')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'{condition}：十分なデータがありません ({len(valid_horses)}頭)', 
                           ha='center', va='center', fontsize=14)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/conditions/{condition_col}_correlation.png", dpi=200)
        plt.close()
        
        # 条件別の最適脚質ランキング
        best_style_by_condition = condition_stats.sort_values(['勝率'], ascending=False)
        best_style_by_condition = best_style_by_condition.groupby(condition_col).head(3)  # 各条件で勝率上位3脚質
        
        # 条件ごとの最適脚質テーブル作成
        best_style_table = best_style_by_condition.pivot_table(
            values='勝率',
            index=condition_col,
            columns=running_style_col,
            aggfunc='max'
        ).fillna(0)
        
        # 条件別最適脚質レポート
        optimal_style_report = {}
        for condition in conditions:
            condition_rows = condition_stats[condition_stats[condition_col] == condition]
            if len(condition_rows) > 0:
                top_styles = condition_rows.sort_values('勝率', ascending=False).head(2)
                optimal_style_report[condition] = top_styles[[running_style_col, '勝率', 'レース数']].values.tolist()
        
        # レポート出力
        with open(f"{output_dir}/conditions/{condition_col}_optimal_styles.txt", 'w', encoding='utf-8') as f:
            f.write(f"# {condition_col}別の最適脚質レポート\n\n")
            for condition, styles in optimal_style_report.items():
                f.write(f"## {condition}\n")
                for style_info in styles:
                    f.write(f"- {style_info[0]}: 勝率 {style_info[1]:.3f} ({style_info[2]}レース)\n")
                f.write("\n")
    else:
        print(f"条件数が{len(conditions)}個のため、{condition_col}の可視化をスキップします")
        
    return condition_stats

def analyze_last_3f_and_style(df, running_style_col, rank_col, output_dir):
    """上がり3F_sec（上がり3ハロン）と脚質の関係分析"""
    print("\n上がり3F_sec（上がり3ハロン）と脚質の関係を分析中...")
    
    # すでに「上がり3F_sec」カラムがあればそれを使う
    if '上がり3F_sec' not in df.columns:
        # 変換処理（従来のlast_3f_col自動検出）
        last_3f_cols = ['上がり3F', '上がり', '上がり3ハロン', '上り3F', '上り3ハロン', 'アガリ3F', 'アガリ', '上3F', '上り', 'F3', '3F', '上り３F']
        last_3f_col = None
        for col in last_3f_cols:
            matching_cols = [c for c in df.columns if col in c]
            if matching_cols:
                last_3f_col = matching_cols[0]
                print(f"上がり3ハロンカラム: {last_3f_col}")
                break
        if last_3f_col is None:
            print(f"上がり3Fに関連するカラムが見つかりませんでした。利用可能なカラム（一部）: {df.columns[:10]}")
            return None
        # データ型変換
        def convert_3f_time(x):
            if pd.isnull(x):
                return np.nan
            x_str = str(x).strip()
            if x_str == '' or x_str == ' ':
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            try:
                return float(x_str.replace(',', '.'))
            except ValueError:
                try:
                    return float(x_str.replace('-', '.'))
                except ValueError:
                    import re
                    pattern = r'(\d+[\.\,\-]?\d*)'
                    match = re.search(pattern, x_str)
                    if match:
                        try:
                            num_str = match.group(1).replace(',', '.').replace('-', '.')
                            return float(num_str)
                        except ValueError:
                            return np.nan
                    return np.nan
        df['上がり3F_sec'] = df[last_3f_col].apply(convert_3f_time)
    
    # 以降は「上がり3F_sec」ベースで分析
    valid_count = df['上がり3F_sec'].notna().sum()
    total_count = len(df)
    valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
    print(f"上がり3F_sec変換結果: {valid_count}/{total_count}行 ({valid_percent:.1f}%)が有効な数値")
    if valid_count < 10:
        print("有効なデータが少なすぎます。上がり3F_sec分析をスキップします。")
        return None
    # 範囲チェック
    q1 = df['上がり3F_sec'].quantile(0.25)
    q3 = df['上がり3F_sec'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(30, q1 - 1.5 * iqr)
    upper_bound = min(60, q3 + 1.5 * iqr)
    print(f"上がり3F_secの有効範囲: {lower_bound:.1f}秒～{upper_bound:.1f}秒")
    valid_df = df[(df['上がり3F_sec'] >= lower_bound) & (df['上がり3F_sec'] <= upper_bound)]
    invalid_count = len(df) - len(valid_df)
    if invalid_count > 0:
        print(f"範囲外のデータ {invalid_count}行を除外しました")
    print(f"分析対象データ: {len(valid_df)}行")
    if len(valid_df) < 10:
        print("有効なデータが少なすぎます。上がり3F_sec分析をスキップします。")
        return None
    # 脚質ごとの上がり3F_sec平均
    style_3f_stats = valid_df.groupby(running_style_col).agg(
        データ数=('上がり3F_sec', 'count'),
        平均上がり3F_sec=('上がり3F_sec', 'mean'),
        最速上がり3F_sec=('上がり3F_sec', 'min'),
        最遅上がり3F_sec=('上がり3F_sec', 'max'),
        標準偏差=('上がり3F_sec', 'std')
    ).reset_index()
    style_3f_stats = style_3f_stats[style_3f_stats['データ数'] >= 3]
    if style_3f_stats.empty:
        print("十分なデータがある脚質がありません。上がり3F_sec分析をスキップします。")
        return None
    style_3f_stats = style_3f_stats.sort_values('平均上がり3F_sec')
    style_3f_stats.to_csv(f"{output_dir}/style_3f_analysis.csv", encoding='utf-8', index=False)
    print(style_3f_stats)
    # 可視化: 脚質ごとの上がり3F_secボックスプロット
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=running_style_col, y='上がり3F_sec', data=valid_df)
    plt.title('脚質ごとの上がり3F_secタイム分布')
    plt.ylabel('上がり3F_sec(秒)')
    plt.xlabel('脚質')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/style_3f_boxplot.png", dpi=200)
    plt.close()
    # 脚質×上がり3F_secランク×勝率のヒートマップ
    try:
        valid_df['上がり3F_secランク'] = pd.qcut(
            valid_df['上がり3F_sec'], 
            q=3, 
            labels=['速い', '普通', '遅い']
        )
        style_3f_rank = valid_df.groupby([running_style_col, '上がり3F_secランク']).agg(
            レース数=(rank_col, 'count'),
            勝利数=(valid_df[rank_col] == 1, 'sum')
        )
        style_3f_rank['勝率'] = style_3f_rank['勝利数'] / style_3f_rank['レース数']
        style_3f_rank = style_3f_rank.reset_index()
        min_races = 3
        filtered_data = style_3f_rank[style_3f_rank['レース数'] >= min_races]
        if len(filtered_data) > 3:
            pivot = filtered_data.pivot_table(
                values='勝率',
                index=running_style_col,
                columns='上がり3F_secランク'
            ).fillna(0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.3f')
            plt.title('脚質×上がり3F_secランク別の勝率')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/style_3f_heatmap.png", dpi=200)
            plt.close()
            pivot.to_csv(f"{output_dir}/style_3f_rank_win_rate.csv", encoding='utf-8')
        else:
            print(f"十分なデータがありません ({len(filtered_data)}行)。ヒートマップをスキップします。")
    except Exception as e:
        print(f"ヒートマップ作成中にエラー: {e}")
    # 散布図: 勝率と上がり3F_secの関係（脚質別）
    try:
        horse_3f_data = valid_df.groupby('馬名').agg(
            平均上がり3F_sec=('上がり3F_sec', 'mean'),
            レース数=(rank_col, 'count'),
            勝利数=(valid_df[rank_col] == 1, 'sum')
        )
        horse_style = valid_df.groupby(['馬名', running_style_col]).size().reset_index(name='count')
        horse_style_max = horse_style.sort_values('count', ascending=False).groupby('馬名').first()
        horse_style_max = horse_style_max.rename(columns={running_style_col: '得意脚質'})
        horse_3f_data = horse_3f_data.merge(
            horse_style_max[['得意脚質']], 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        horse_3f_data['勝率'] = horse_3f_data['勝利数'] / horse_3f_data['レース数']
        horse_3f_data = horse_3f_data[horse_3f_data['レース数'] >= 3]
        if len(horse_3f_data) >= 10:
            plt.figure(figsize=(12, 8))
            styles = horse_3f_data['得意脚質'].dropna().unique()
            legend_added = False
            
            for style in styles:
                style_data = horse_3f_data[horse_3f_data['得意脚質'] == style]
                if len(style_data) >= 5:
                    plt.scatter(style_data['平均上がり3F_sec'], style_data['勝率'], 
                            label=style, alpha=0.7, s=style_data['レース数'] * 3)
                    legend_added = True
                    
            if legend_added:
                plt.xlabel('平均上がり3F_sec(秒)')
                plt.ylabel('勝率')
                plt.title('脚質別: 平均上がり3F_secと勝率の関係')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/style_3f_scatter.png", dpi=200)
                plt.close()
                
                for style in styles:
                    style_data = horse_3f_data[horse_3f_data['得意脚質'] == style]
                    if len(style_data) >= 5:
                        corr, p = stats.pearsonr(style_data['平均上がり3F_sec'], style_data['勝率'])
                        print(f"{style}脚質 - 上がり3F_secと勝率の相関: r={corr:.3f}, p={p:.3f}")
            else:
                print("十分なデータがある脚質がないため、散布図をスキップします")
        else:
            print(f"散布図分析に十分な馬のデータがありません ({len(horse_3f_data)}頭)")
    except Exception as e:
        print(f"散布図作成中にエラー: {e}")
    return style_3f_stats

def analyze_corner_position(df, running_style_col, rank_col, output_dir):
    """コーナー通過順位と最終着順の関係分析"""
    print("\nコーナー通過順位と最終着順の関係を分析中...")
    
    # コーナー通過順位カラムを特定
    corner_patterns = {
        '4角': ['4角', '4コーナー', '四角', '最終コーナー'],
        '3角': ['3角', '3コーナー', '三角'],
        '2角': ['2角', '2コーナー', '二角'],
        '1角': ['1角', '1コーナー', '一角', '最初のコーナー']
    }
    
    corner_columns = {}
    for corner, patterns in corner_patterns.items():
        for pattern in patterns:
            matches = [col for col in df.columns if pattern in col]
            if matches:
                corner_columns[corner] = matches[0]
                print(f"{corner}のカラム: {matches[0]}")
                break
    
    if not corner_columns:
        print("コーナー通過順位のカラムが見つかりません。この分析をスキップします。")
        return None
    
    try:
        # データ前処理
        # コーナー通過順位を数値に変換
        for corner, col in corner_columns.items():
            # より堅牢なデータ変換関数
            def extract_position(x):
                if pd.isnull(x):
                    return np.nan
                
                x_str = str(x).strip()
                if x_str == '' or x_str == ' ':
                    return np.nan
                
                try:
                    # "3/16" のような表記から先頭の数字を抽出
                    if '/' in x_str:
                        return int(x_str.split('/')[0])
                    
                    # 数値への直接変換を試みる
                    return int(float(x_str))
                except (ValueError, TypeError):
                    # 数字パターンを抽出（例: "3番手" → 3）
                    import re
                    match = re.search(r'(\d+)', x_str)
                    if match:
                        return int(match.group(1))
                    return np.nan
            
            # 各コーナーの順位を抽出して新しいカラムに格納
            df[f'{corner}_順位'] = df[col].apply(extract_position)
            
            # デバッグ情報を出力
            non_numeric = df[df[f'{corner}_順位'].isna() & df[col].notna()]
            if len(non_numeric) > 0:
                sample_values = non_numeric[col].head(5).tolist()
                print(f"  注意: {len(non_numeric)}行で数値に変換できない値がありました。例: {sample_values}")
        
        # 分析1: コーナー別の平均順位と平均着順の関係
        corner_stats = pd.DataFrame()
        for corner in corner_columns.keys():
            corner_col = f'{corner}_順位'
            if corner_col in df.columns:
                # NaNを除外してから集計
                valid_data = df.dropna(subset=[corner_col, rank_col])
                
                if len(valid_data) < 10:
                    print(f"  警告: {corner}の有効なデータが少なすぎます ({len(valid_data)}行)。このコーナーの分析をスキップします。")
                    continue
                
                # 勝利フラグを事前に計算
                is_winner = valid_data[rank_col] == 1
                
                # 各順位ごとの平均着順と勝率
                # エラー修正: グループ化処理の修正
                try:
                    pos_stats = valid_data.groupby(corner_col).agg(
                        データ数=(rank_col, 'count'),
                        平均着順=(rank_col, 'mean')
                    )
                    # 勝利数は別途計算
                    wins_by_pos = valid_data[is_winner].groupby(corner_col).size()
                    pos_stats['勝利数'] = wins_by_pos
                    pos_stats['勝利数'] = pos_stats['勝利数'].fillna(0)
                    pos_stats['勝率'] = pos_stats['勝利数'] / pos_stats['データ数']
                    pos_stats = pos_stats.reset_index()
                    pos_stats['コーナー'] = corner
                    corner_stats = pd.concat([corner_stats, pos_stats])
                except Exception as e:
                    print(f"  警告: {corner}のデータ集計中にエラーが発生しました: {e}")
                    continue
        
        if corner_stats.empty:
            print("  警告: 有効なコーナーデータがありません。コーナー分析をスキップします。")
            return None
        
        # 結果を保存
        corner_stats.to_csv(f"{output_dir}/corner_position_stats.csv", encoding='utf-8', index=False)
        print(f"  コーナー分析データを保存しました ({len(corner_stats)}行)")
        
        # 可視化1: コーナー別通過順位と勝率の関係
        plt.figure(figsize=(15, 8))
        colors = plt.cm.get_cmap('tab10', len(corner_columns))
        for idx, corner in enumerate(corner_columns.keys()):
            corner_data = corner_stats[corner_stats['コーナー'] == corner]
            # データ数5件未満の順位は除外
            filtered = corner_data[corner_data['データ数'] >= 5]
            if not filtered.empty and len(filtered) > 1:
                # 順位でソートしてからプロット
                filtered = filtered.sort_values(f'{corner}_順位')
                plt.plot(
                    filtered[f'{corner}_順位'],
                    filtered['勝率'],
                    marker='o',
                    label=f'{corner}',
                    linewidth=2,
                    color=colors(idx)
                )
        plt.xlabel('通過順位', fontsize=14)
        plt.ylabel('勝率', fontsize=14)
        plt.title('コーナー別通過順位と勝率の関係（データ数5件未満は除外）', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/corner_position_win_rate.png", dpi=200)
        plt.close()
        
        # 可視化2: ヒートマップ（4角通過順位×脚質×勝率）
        if '4角_順位' in df.columns:
            # 有効なデータのみを抽出
            valid_4c = df.dropna(subset=['4角_順位', running_style_col, rank_col])
            
            if len(valid_4c) < 10:
                print("  警告: 4角の有効データが少なすぎます。ヒートマップの作成をスキップします。")
            else:
                # 順位を5位ごとにグループ化
                # 最大値を確認して適切な区分を設定
                max_pos = valid_4c['4角_順位'].max()
                if pd.notna(max_pos):
                    if max_pos <= 18:  # 一般的なレース頭数に対応
                        bins = [0, 3, 6, 9, 12, float('inf')]
                        labels = ['1-3位', '4-6位', '7-9位', '10-12位', '13位以降']
                    else:
                        # 大人数の場合は区分を調整
                        bins = [0, 5, 10, 15, float('inf')]
                        labels = ['1-5位', '6-10位', '11-15位', '16位以降']
                    
                    valid_4c['4角グループ'] = pd.cut(valid_4c['4角_順位'], bins=bins, labels=labels)
                    
                    # 勝利フラグを事前に計算
                    valid_4c_is_winner = valid_4c[rank_col] == 1
                    
                    # 集計（エラーを避けるために別々に計算）
                    try:
                        # まずレース数を集計
                        race_counts = valid_4c.groupby([running_style_col, '4角グループ']).size().reset_index(name='レース数')
                        
                        # 次に勝利数を集計
                        winners = valid_4c[valid_4c_is_winner]
                        if len(winners) > 0:
                            win_counts = winners.groupby([running_style_col, '4角グループ']).size().reset_index(name='勝利数')
                            
                            # データを結合
                            style_corner = pd.merge(
                                race_counts, 
                                win_counts, 
                                on=[running_style_col, '4角グループ'], 
                                how='left'
                            )
                        else:
                            style_corner = race_counts.copy()
                            style_corner['勝利数'] = 0
                        
                        # 欠損値を0に置換
                        style_corner['勝利数'] = style_corner['勝利数'].fillna(0)
                        style_corner['勝率'] = style_corner['勝利数'] / style_corner['レース数']
                        
                        # 最低レース数での絞り込み（安定した勝率計算のため）
                        min_races = 5
                        style_corner_filtered = style_corner[style_corner['レース数'] >= min_races]
                        
                        if style_corner_filtered.empty:
                            print(f"  警告: {min_races}レース以上のグループがありません。フィルタを緩和します。")
                            style_corner_filtered = style_corner  # フィルタを解除
                        
                        # ピボットテーブルとヒートマップ
                        if len(style_corner_filtered) > 1:
                            pivot = style_corner_filtered.pivot_table(
                                values='勝率',
                                index=running_style_col,
                                columns='4角グループ'
                            ).fillna(0)
                            
                            if not pivot.empty and pivot.size > 1:
                                plt.figure(figsize=(12, 8))
                                sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.3f')
                                plt.title('脚質×4角通過位置別の勝率')
                                plt.tight_layout()
                                plt.savefig(f"{output_dir}/style_4corner_heatmap.png", dpi=200)
                                plt.close()
                            else:
                                print("  警告: 有効なデータが不足しているため、ヒートマップを作成できません。")
                        else:
                            print("  警告: 十分なデータポイントがないため、ヒートマップを作成できません。")
                    except Exception as e:
                        print(f"  警告: ヒートマップ作成中にエラーが発生しました: {e}")
                else:
                    print("  警告: 4角順位の最大値が不正です。ヒートマップをスキップします。")
        
        # 分析3: 脚質ごとのコーナー通過順位変化
        if len(corner_columns) > 1:
            # 必要なカラムが少なくとも1つあるレコードのみを対象に
            corner_cols = [f'{corner}_順位' for corner in corner_columns.keys() if f'{corner}_順位' in df.columns]
            if len(corner_cols) < 2:
                print("  警告: 複数のコーナーデータがないため、通過順位変化の分析をスキップします。")
            else:
                try:
                    # 少なくとも1つのコーナーデータがあるレコードを抽出
                    has_corner_data = df[corner_cols].notna().any(axis=1)
                    valid_df = df[has_corner_data]
                    
                    corner_change = pd.DataFrame()
                    for style in valid_df[running_style_col].unique():
                        style_df = valid_df[valid_df[running_style_col] == style]
                        if len(style_df) < 5:  # 最低5レースのデータがある脚質のみ分析
                            continue
                            
                        change_row = {'脚質': style, 'データ数': len(style_df)}
                        
                        # 各コーナーの平均順位
                        for corner in corner_columns.keys():
                            corner_col = f'{corner}_順位'
                            if corner_col in df.columns:
                                change_row[f'{corner}_平均順位'] = style_df[corner_col].mean()
                        
                        corner_change = pd.concat([corner_change, pd.DataFrame([change_row])], ignore_index=True)
                    
                    if len(corner_change) > 0:
                        # 結果を保存
                        corner_change.to_csv(f"{output_dir}/style_corner_position_change.csv", encoding='utf-8', index=False)
                        
                        # 可視化: 脚質ごとのコーナー通過順位変化
                        plt.figure(figsize=(12, 8))
                        corners = [corner for corner in ['1角', '2角', '3角', '4角'] 
                                  if f'{corner}_平均順位' in corner_change.columns]
                        
                        if len(corners) >= 2:  # 少なくとも2つのコーナーデータが必要
                            has_valid_data = False
                            for _, row in corner_change.iterrows():
                                # すべてのコーナーデータが有効な行のみプロット
                                if all(not pd.isna(row[f'{corner}_平均順位']) for corner in corners):
                                    plt.plot(
                                        corners, 
                                        [row[f'{corner}_平均順位'] for corner in corners], 
                                        marker='o', label=row['脚質'], linewidth=2
                                    )
                                    has_valid_data = True
                            
                            if has_valid_data:
                                plt.xlabel('コーナー')
                                plt.ylabel('平均通過順位')
                                plt.gca().invert_yaxis()  # 順位は小さい方が上位なので反転
                                plt.title('脚質ごとのコーナー通過順位変化')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                plt.savefig(f"{output_dir}/style_corner_position_change.png", dpi=200)
                                plt.close()
                            else:
                                print("  警告: 有効なデータ系列がありません。通過順位変化図の作成をスキップします。")
                        else:
                            print("  警告: 連続するコーナーデータが不足しています。通過順位変化図の作成をスキップします。")
                    else:
                        print("  警告: 有効なコーナーデータが不足しています。通過順位変化の分析をスキップします。")
                except Exception as e:
                    print(f"  警告: 通過順位変化分析中にエラーが発生しました: {e}")
                
        return corner_stats
    except Exception as e:
        print(f"コーナー通過順位分析中にエラーが発生しました: {str(e)}")
        import traceback
        print(f"エラーの詳細:\n{traceback.format_exc()}")
        return None

def analyze_feature_importance(df, running_style_col, rank_col, output_dir):
    """ランダムフォレストによる重要変数の同定"""
    print("\nランダムフォレストによるトラックバイアス関連特徴量の重要度分析を実行中...")
    
    # トラックバイアス関連の特徴量のリスト
    trackbias_cols = [
        '競馬場', '芝ダート', 'コース', '距離', '馬場状態', 
        '季節', '月', '有利脚質該当', '有利脚質該当率'
    ]
    # コース関連列やコーナー通過順位も含める
    trackbias_patterns = [
        '角_順位', '脚質', '通過', 'コーナー', '馬場',
        '上がり', '内外', '回り', 'コース'
    ]
    
    # 数値型特徴量のみ抽出
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # トラックバイアス関連の特徴量のみを選択
    feature_cols = []
    for col in numeric_cols:
        # 明示的に指定された列か、パターンに一致する列のみ追加
        if col in trackbias_cols or any(pattern in col for pattern in trackbias_patterns):
            feature_cols.append(col)
    
    # 明示的に除外する特徴量
    exclude_cols = ['単勝', '複勝', '賞金', 'オッズ', '人気']
    feature_cols = [col for col in feature_cols if not any(ex in col for ex in exclude_cols)]
    
    if len(feature_cols) < 3:
        print("  警告: 分析に十分な特徴量がありません。特徴量重要度分析をスキップします")
        return None, None
    
    # 欠損値のみの列を事前に除外する
    valid_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:  # 少なくとも1つの有効な値がある
            valid_features.append(col)
        else:
            print(f"  警告: {col}は全て欠損値のため除外します")
    
    feature_cols = valid_features
    print(f"  選択されたトラックバイアス関連特徴量: {feature_cols} ({len(feature_cols)}個)")
    
    # データ準備
    X = df[feature_cols].copy()
    y_win = (df[rank_col] == 1).astype(int)  # 勝利フラグ
    
    # 欠損値処理
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # ランダムフォレストによる重要度算出
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_win, test_size=0.3, random_state=42)
    
    # モデル学習
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 性能評価
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    print(f"  ランダムフォレスト - トレーニングスコア: {train_score:.4f}")
    print(f"  ランダムフォレスト - テストスコア: {test_score:.4f}")
    print(f"  特徴量数: {len(feature_cols)}, 重要度配列長: {len(rf.feature_importances_)}")
    
    # 特徴量重要度の抽出（長さの確認と調整）
    if len(feature_cols) == len(rf.feature_importances_):
        importance = pd.DataFrame({
            '特徴量': feature_cols,
            '重要度': rf.feature_importances_
        }).sort_values('重要度', ascending=False)
        
        # 結果保存
        importance.to_csv(f"{output_dir}/feature_importance.csv", index=False, encoding='utf-8')
        
        # 上位10特徴のみ可視化
        top_n = min(10, len(importance))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='重要度', y='特徴量', data=importance.head(top_n))
        plt.title('トラックバイアスに関する特徴量重要度（ランダムフォレスト）')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=200)
        plt.close()
        
        return importance, rf
    else:
        print(f"  エラー: 特徴量リスト({len(feature_cols)})と重要度配列({len(rf.feature_importances_)})の長さが一致しません")
        return None, rf

def analyze_xgboost_prediction(df, running_style_col, rank_col, output_dir, importance=None):
    """XGBoostによる勝率予測モデル構築"""
    print("\nXGBoostによる勝率予測モデル構築を実行中...")
    
    # XGBoostが利用可能か確認
    if xgb is None:
        print("  エラー: XGBoostがインストールされていません。この分析をスキップします。")
        return None, None
    
    # 特徴選択：数値型カラムのみ抽出
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 不要カラムの除外
    exclude_cols = [rank_col, '馬名', '騎手', 'レースID', 'インデックス', '年月日', '日付', '開催']
    # exclude_cols拡張（独自ルール）
    feature_cols = [col for col in numeric_cols if not any(ex in col for ex in exclude_cols)]
    
    # 重要な特徴量のみを使用する（ランダムフォレストの結果があれば）
    if importance is not None and len(importance) > 5:
        top_features = importance.head(10)['特徴量'].tolist()
        feature_cols = [col for col in feature_cols if col in top_features]
        print(f"  ランダムフォレストの重要度に基づき上位特徴量を選択: {feature_cols}")
    
    if len(feature_cols) < 3:
        print("  警告: XGBoost分析に十分な特徴量がありません。分析をスキップします")
        return None, None
    
    # データ準備
    X = df[feature_cols].copy().fillna(df[feature_cols].mean())
    y_win = (df[rank_col] == 1).astype(int)  # 勝利フラグ
    
    # 欠損値処理
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # データ分割
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_win, test_size=0.3, random_state=42)
    
    # XGBoostモデル構築
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'seed': 42
    }
    
    # 交差検証
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"  交差検証 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # モデル訓練
    xgb_model.fit(X_train, y_train)
    
    # 性能評価
    from sklearn.metrics import roc_auc_score
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  テストデータ AUC: {auc:.4f}")
    
    # 結果を保存
    with open(f"{output_dir}/xgboost_model_report.txt", 'w', encoding='utf-8') as f:
        f.write("XGBoost勝率予測モデルレポート\n")
        f.write(f"交差検証 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"テストデータ AUC: {auc:.4f}\n\n")
        f.write("分類レポート:\n")
        f.write(classification_report(y_test, y_pred))
    
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率')
    plt.ylabel('真陽性率')
    plt.title('XGBoost勝率予測のROC曲線')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/xgboost_roc_curve.png", dpi=200)
    plt.close()
    
    # 特徴量重要度の可視化
    xgb_importance = pd.DataFrame({
        '特徴量': feature_cols,
        '重要度': xgb_model.feature_importances_
    }).sort_values('重要度', ascending=False)
    
    xgb_importance.to_csv(f"{output_dir}/xgboost_importance.csv", index=False, encoding='utf-8')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='重要度', y='特徴量', data=xgb_importance)
    plt.title('XGBoost 特徴量重要度')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xgboost_importance.png", dpi=200)
    plt.close()
    
    return xgb_model, auc

def analyze_horse_clustering(df, running_style_col, rank_col, output_dir):
    """競走馬のクラスタリングによる脚質グループ分類"""
    print("\n競走馬のクラスタリングによる脚質グループ分類を実行中...")
    
    # 馬ごとの統計量を計算（特徴エンジニアリング）
    if '馬名' not in df.columns:
        print("  エラー: '馬名'カラムが見つかりません。クラスタリングをスキップします。")
        return None, None
    
    print("  馬ごとの特徴量を生成中...")
    
    # 必要なカラムがなければ作成
    df_work = df.copy()
    if '勝利' not in df_work.columns:
        df_work['勝利'] = df_work[rank_col] == 1
        print("  「勝利」フラグを作成しました")
    
    if '好成績' not in df_work.columns:
        df_work['好成績'] = df_work[rank_col].isin([1, 2, 3])  # 1-3着を好成績とする
        print("  「好成績」フラグを作成しました")
    
    # 各馬の基本統計量を生成
    horse_stats = df_work.groupby('馬名').agg(
        レース数=('勝利', 'count'),
        勝率=(rank_col, lambda x: (x == 1).mean()),
        好成績率=('好成績', 'mean'),
        平均着順=(rank_col, 'mean')
    )
    
    # コーナー通過順位の特徴量を追加（あれば）
    corner_columns = ['1角_順位', '2角_順位', '3角_順位', '4角_順位']
    for corner in corner_columns:
        if corner in df.columns:
            horse_stats[f'平均{corner}'] = df.groupby('馬名')[corner].mean()
    
    # 上がり3Fの特徴量を追加（あれば）
    if '上がり3F_sec' in df.columns:
        horse_stats['平均上がり3F'] = df.groupby('馬名')['上がり3F_sec'].mean()
    
    # 脚質の頻度を追加
    style_counts = pd.crosstab(df['馬名'], df[running_style_col])
    style_counts = style_counts.div(style_counts.sum(axis=1), axis=0)  # 割合に変換
    
    # スタイル接頭辞をつける
    style_counts = style_counts.add_prefix('脚質_')
    
    # 結合
    horse_features = horse_stats.join(style_counts)
    
    # レース数が少ない馬は除外
    min_races = 5
    horse_features = horse_features[horse_features['レース数'] >= min_races]
    
    # 欠損値の処理
    horse_features = horse_features.fillna(0)
    
    if len(horse_features) < 10:
        print(f"  警告: クラスタリングに十分なデータがありません（{len(horse_features)}頭のみ）。スキップします。")
        return None, None
    
    print(f"  クラスタリング対象馬: {len(horse_features)}頭")
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(horse_features)
    
    # 最適なクラスタ数を決定（シルエットスコア）
    max_clusters = min(10, len(horse_features) // 5)
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # シルエットスコア計算（クラスタ評価指標）
        if len(set(cluster_labels)) > 1:  # 2つ以上のクラスタがある場合
            score = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append((n_clusters, score))
    
    # 最適クラスタ数を選択
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"  最適クラスタ数: {best_n_clusters}")
    
    # 最終的なクラスタリング
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # 結果を元のデータフレームに追加
    horse_features['クラスタ'] = cluster_labels
    
    # 各クラスタの特性を分析
    cluster_profiles = horse_features.groupby('クラスタ').mean()
    cluster_profiles['頭数'] = horse_features.groupby('クラスタ').size()
    cluster_profiles = cluster_profiles.sort_values('頭数', ascending=False)
    
    # 結果保存
    horse_features.to_csv(f"{output_dir}/horse_clusters.csv", encoding='utf-8')
    cluster_profiles.to_csv(f"{output_dir}/cluster_profiles.csv", encoding='utf-8')
    
    print("  クラスタプロファイル:")
    print(cluster_profiles[['頭数', '勝率', '好成績率', '平均着順']])
    
    # クラスタの命名（脚質の特徴から）
    style_cols = [col for col in horse_features.columns if col.startswith('脚質_')]
    style_profiles = cluster_profiles[style_cols]
    
    # 各クラスタの主要脚質を特定
    cluster_names = {}
    for cluster in style_profiles.index:
        # そのクラスタの最も高い脚質割合を特定
        main_style = style_profiles.loc[cluster].idxmax().replace('脚質_', '')
        second_style = style_profiles.loc[cluster].sort_values(ascending=False).index[1].replace('脚質_', '')
        
        # 主要脚質とクラスタ番号から名前を生成
        cluster_names[cluster] = f"C{cluster}:{main_style}主体"
    
    # クラスタの可視化（PCAで次元削減）
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # 可視化用データフレーム
    viz_df = pd.DataFrame({
        '馬名': horse_features.index,
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'クラスタ': cluster_labels,
        '勝率': horse_features['勝率']
    })
    
    # クラスタ可視化
    plt.figure(figsize=(12, 10))
    
    for cluster in sorted(viz_df['クラスタ'].unique()):
        cluster_data = viz_df[viz_df['クラスタ'] == cluster]
        plt.scatter(
            cluster_data['PC1'], 
            cluster_data['PC2'], 
            s=cluster_data['勝率'] * 500 + 30,  # 勝率に応じてサイズ変更
            alpha=0.7, 
            label=f"{cluster_names[cluster]}({len(cluster_data)}頭)"
        )
    
    plt.title('競走馬のクラスタリング結果（PCA次元削減）')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/horse_clusters_pca.png", dpi=200)
    plt.close()
    
    # クラスタごとの勝率分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='クラスタ', y='勝率', data=viz_df)
    plt.title('クラスタ別の勝率分布')
    plt.xticks(range(best_n_clusters), [cluster_names[i] for i in range(best_n_clusters)])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_win_rate_boxplot.png", dpi=200)
    plt.close()
    
    return horse_features, cluster_profiles
        
def analyze_last_3f_index_and_style(df, running_style_col, rank_col, output_dir):
    """上がり指数と脚質の関係分析"""
    print("\n上がり指数と脚質の関係を分析中...")
    
    # すでに「上がり指数」カラムがあればそれを使う
    if '上がり指数' not in df.columns:
        print("上がり指数カラムが見つかりませんでした。分析をスキップします。")
        return None

    # 上がり指数の型変換（数値化）
    df['上がり指数_num'] = pd.to_numeric(df['上がり指数'], errors='coerce')
    
    # 以降は「上がり指数_num」ベースで分析
    valid_count = df['上がり指数_num'].notna().sum()
    total_count = len(df)
    valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
    print(f"上がり指数変換結果: {valid_count}/{total_count}行 ({valid_percent:.1f}%)が有効な数値")
    
    if valid_count < 10:
        print("有効なデータが少なすぎます。上がり指数分析をスキップします。")
        return None
        
    # 範囲チェック
    q1 = df['上がり指数_num'].quantile(0.25)
    q3 = df['上がり指数_num'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f"上がり指数の有効範囲: {lower_bound:.1f}～{upper_bound:.1f}")
    
    valid_df = df[(df['上がり指数_num'] >= lower_bound) & (df['上がり指数_num'] <= upper_bound)]
    invalid_count = len(df) - len(valid_df)
    
    if invalid_count > 0:
        print(f"範囲外のデータ {invalid_count}行を除外しました")
    
    print(f"分析対象データ: {len(valid_df)}行")
    
    if len(valid_df) < 10:
        print("有効なデータが少なすぎます。上がり指数分析をスキップします。")
        return None
        
    # 脚質ごとの上がり指数平均
    style_3f_stats = valid_df.groupby(running_style_col).agg(
        データ数=('上がり指数_num', 'count'),
        平均上がり指数=('上がり指数_num', 'mean'),
        最高上がり指数=('上がり指数_num', 'max'),
        最低上がり指数=('上がり指数_num', 'min'),
        標準偏差=('上がり指数_num', 'std')
    ).reset_index()
    
    style_3f_stats = style_3f_stats[style_3f_stats['データ数'] >= 3]
    
    if style_3f_stats.empty:
        print("十分なデータがある脚質がありません。上がり指数分析をスキップします。")
        return None
        
    style_3f_stats = style_3f_stats.sort_values('平均上がり指数', ascending=False)
    style_3f_stats.to_csv(f"{output_dir}/style_agari_index_analysis.csv", encoding='utf-8', index=False)
    print(style_3f_stats)
    
    # 可視化: 脚質ごとの上がり指数ボックスプロット
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=running_style_col, y='上がり指数_num', data=valid_df)
    plt.title('脚質ごとの上がり指数分布')
    plt.ylabel('上がり指数')
    plt.xlabel('脚質')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/style_agari_index_boxplot.png", dpi=200)
    plt.close()
    
    # 脚質×上がり指数ランク×勝率のヒートマップ
    try:
        valid_df['上がり指数ランク'] = pd.qcut(
            valid_df['上がり指数_num'], 
            q=3, 
            labels=['低い', '普通', '高い']
        )
        style_3f_rank = valid_df.groupby([running_style_col, '上がり指数ランク']).agg(
            レース数=(rank_col, 'count'),
            勝利数=(valid_df[rank_col] == 1, 'sum')
        )
        style_3f_rank['勝率'] = style_3f_rank['勝利数'] / style_3f_rank['レース数']
        style_3f_rank = style_3f_rank.reset_index()
        
        min_races = 3
        filtered_data = style_3f_rank[style_3f_rank['レース数'] >= min_races]
        
        if len(filtered_data) > 3:
            pivot = filtered_data.pivot_table(
                values='勝率',
                index=running_style_col,
                columns='上がり指数ランク'
            ).fillna(0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.3f')
            plt.title('脚質×上がり指数ランク別の勝率')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/style_agari_index_heatmap.png", dpi=200)
            plt.close()
            pivot.to_csv(f"{output_dir}/style_agari_index_rank_win_rate.csv", encoding='utf-8')
        else:
            print(f"十分なデータがありません ({len(filtered_data)}行)。ヒートマップをスキップします。")
    except Exception as e:
        print(f"ヒートマップ作成中にエラー: {e}")
    
    # 散布図: 勝率と上がり指数の関係（脚質別）
    try:
        horse_3f_data = valid_df.groupby('馬名').agg(
            平均上がり指数=('上がり指数_num', 'mean'),
            レース数=(rank_col, 'count'),
            勝利数=(valid_df[rank_col] == 1, 'sum')
        )
        
        horse_style = valid_df.groupby(['馬名', running_style_col]).size().reset_index(name='count')
        horse_style_max = horse_style.sort_values('count', ascending=False).groupby('馬名').first()
        horse_style_max = horse_style_max.rename(columns={running_style_col: '得意脚質'})
        
        horse_3f_data = horse_3f_data.merge(
            horse_style_max[['得意脚質']], 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        horse_3f_data['勝率'] = horse_3f_data['勝利数'] / horse_3f_data['レース数']
        horse_3f_data = horse_3f_data[horse_3f_data['レース数'] >= 3]
        
        if len(horse_3f_data) >= 10:
            plt.figure(figsize=(12, 8))
            styles = horse_3f_data['得意脚質'].dropna().unique()
            legend_added = False
            
            for style in styles:
                style_data = horse_3f_data[horse_3f_data['得意脚質'] == style]
                if len(style_data) >= 5:
                    plt.scatter(style_data['平均上がり指数'], style_data['勝率'], 
                            label=style, alpha=0.7, s=style_data['レース数'] * 3)
                    legend_added = True
                    
            if legend_added:
                plt.xlabel('平均上がり指数')
                plt.ylabel('勝率')
                plt.title('脚質別: 平均上がり指数と勝率の関係')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/style_agari_index_scatter.png", dpi=200)
                plt.close()
                
                for style in styles:
                    style_data = horse_3f_data[horse_3f_data['得意脚質'] == style]
                    if len(style_data) >= 5:
                        corr, p = stats.pearsonr(style_data['平均上がり指数'], style_data['勝率'])
                        print(f"{style}脚質 - 上がり指数と勝率の相関: r={corr:.3f}, p={p:.3f}")
            else:
                print("十分なデータがある脚質がないため、散布図をスキップします")
        else:
            print(f"散布図分析に十分な馬のデータがありません ({len(horse_3f_data)}頭)")
    except Exception as e:
        print(f"散布図作成中にエラー: {e}")
        
    # 勝率と上がり指数の全体相関
    try:
        all_corr, all_p = stats.pearsonr(horse_3f_data['平均上がり指数'], horse_3f_data['勝率'])
        print(f"全体 - 上がり指数と勝率の相関: r={all_corr:.3f}, p={all_p:.3f}")
        
        # 全体相関の散布図と回帰直線
        plt.figure(figsize=(10, 8))
        
        # データ点の散布図
        plt.scatter(horse_3f_data['平均上がり指数'], horse_3f_data['勝率'], 
                   alpha=0.7, s=horse_3f_data['レース数'] * 3, c='blue')
        
        # 回帰直線
        if len(horse_3f_data) >= 10:
            X = horse_3f_data['平均上がり指数'].values.reshape(-1, 1)
            y = horse_3f_data['勝率'].values
            reg = LinearRegression().fit(X, y)
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            plt.plot(X_range, reg.predict(X_range), color='red', linewidth=2)
            
            plt.xlabel('平均上がり指数')
            plt.ylabel('勝率')
            plt.title(f'上がり指数と勝率の関係 (r={all_corr:.3f}, p={all_p:.3f})')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/agari_index_win_rate_regression.png", dpi=200)
            plt.close()
    except Exception as e:
        print(f"相関分析中にエラー: {e}")
        
    return style_3f_stats

def analyze_agari_index_win_rate(df, rank_col, output_dir):
    """上がり指数と勝率の関係に特化した分析"""
    print("\n上がり指数と勝率の詳細分析を実行中...")
    
    os.makedirs(f"{output_dir}/agari_index_analysis", exist_ok=True)
    
    # すでに「上がり指数」カラムがあればそれを使う
    if '上がり指数' not in df.columns:
        print("上がり指数カラムが見つかりませんでした。分析をスキップします。")
        return None

    # 上がり指数の型変換（数値化）
    df['上がり指数_num'] = pd.to_numeric(df['上がり指数'], errors='coerce')
    
    # 以降は「上がり指数_num」ベースで分析
    valid_count = df['上がり指数_num'].notna().sum()
    total_count = len(df)
    valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
    print(f"上がり指数変換結果: {valid_count}/{total_count}行 ({valid_percent:.1f}%)が有効な数値")
    
    if valid_count < 10:
        print("有効なデータが少なすぎます。上がり指数分析をスキップします。")
        return None
    
    # 勝利フラグを追加
    df['勝利'] = df[rank_col] == 1
    
    # 1. 上がり指数の基本統計量
    stats = {
        '平均': df['上がり指数_num'].mean(),
        '中央値': df['上がり指数_num'].median(),
        '最小値': df['上がり指数_num'].min(),
        '最大値': df['上がり指数_num'].max(),
        '標準偏差': df['上がり指数_num'].std(),
    }
    print("上がり指数の基本統計量:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 2. 上がり指数を5つの分位に分割し、各分位の勝率を計算
    try:
        n_bins = 5
        df['上がり指数_bins'] = pd.qcut(df['上がり指数_num'], q=n_bins, labels=[f'第{i+1}分位' for i in range(n_bins)])
        
        # 各分位の勝率計算
        bin_stats = df.groupby('上がり指数_bins').agg(
            データ数=('勝利', 'count'),
            勝利数=('勝利', 'sum')
        )
        bin_stats['勝率'] = bin_stats['勝利数'] / bin_stats['データ数']
        
        print("\n上がり指数の分位別勝率:")
        print(bin_stats)
        
        # CSVに保存
        bin_stats.to_csv(f"{output_dir}/agari_index_analysis/index_bins_win_rate.csv", encoding='utf-8')
        
        # 可視化: 上がり指数の分位別勝率
        plt.figure(figsize=(10, 6))
        plt.bar(bin_stats.index, bin_stats['勝率'], color='skyblue')
        plt.xlabel('上がり指数の分位')
        plt.ylabel('勝率')
        plt.title('上がり指数の分位別勝率')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/agari_index_analysis/index_bins_win_rate.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"分位別勝率分析でエラー: {e}")
    
    # 3. 上がり指数と勝率の関係を連続的に分析
    try:
        # 上がり指数をより細かく10区間に分割
        n_detailed_bins = 10
        df['上がり指数_detailed'] = pd.qcut(df['上がり指数_num'], q=n_detailed_bins, duplicates='drop')
        
        # 各区間の統計情報
        detailed_stats = df.groupby('上がり指数_detailed').agg(
            平均上がり指数=('上がり指数_num', 'mean'),
            データ数=('勝利', 'count'),
            勝利数=('勝利', 'sum')
        )
        detailed_stats['勝率'] = detailed_stats['勝利数'] / detailed_stats['データ数']
        
        # CSVに保存
        detailed_stats = detailed_stats.sort_values('平均上がり指数')
        detailed_stats.to_csv(f"{output_dir}/agari_index_analysis/detailed_index_win_rate.csv", encoding='utf-8')
        
        # 可視化: 上がり指数の連続的な勝率変化
        plt.figure(figsize=(12, 6))
        
        # 折れ線グラフ
        plt.plot(detailed_stats['平均上がり指数'], detailed_stats['勝率'], 'o-', color='blue', markersize=8)
        
        # サンプルサイズに応じたマーカーサイズで散布図もプロット
        sizes = detailed_stats['データ数'] / detailed_stats['データ数'].max() * 200 + 50
        plt.scatter(detailed_stats['平均上がり指数'], detailed_stats['勝率'], s=sizes, alpha=0.5, color='blue')
        
        plt.xlabel('上がり指数')
        plt.ylabel('勝率')
        plt.title('上がり指数と勝率の関係')
        plt.grid(True, alpha=0.3)
        
        # 回帰直線を追加
        X = detailed_stats['平均上がり指数'].values.reshape(-1, 1)
        y = detailed_stats['勝率'].values
        if len(X) > 1:
            reg = LinearRegression().fit(X, y)
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            plt.plot(X_range, reg.predict(X_range), 'r--', linewidth=2)
            
            # 相関係数を計算
            corr, p_value = stats.pearsonr(X.flatten(), y)
            plt.title(f'上がり指数と勝率の関係 (r={corr:.3f}, p={p_value:.3f})')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/agari_index_analysis/continuous_index_win_rate.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"連続的な勝率分析でエラー: {e}")
    
    # 4. 上がり指数を馬場状態とクロス分析
    if '馬場状態' in df.columns:
        try:
            # 馬場状態ごとの上がり指数と勝率の関係
            track_conditions = df['馬場状態'].dropna().unique()
            
            plt.figure(figsize=(14, 8))
            
            for i, condition in enumerate(track_conditions):
                condition_df = df[df['馬場状態'] == condition]
                
                # 十分なデータがある場合のみ分析
                if len(condition_df) >= 20:
                    try:
                        # 上がり指数をビンに分割
                        condition_df['上がり指数_bin'] = pd.qcut(condition_df['上がり指数_num'], 5, duplicates='drop')
                        
                        # ビンごとの勝率を計算
                        condition_stats = condition_df.groupby('上がり指数_bin').agg(
                            平均上がり指数=('上がり指数_num', 'mean'),
                            データ数=('勝利', 'count'),
                            勝利数=('勝利', 'sum')
                        )
                        condition_stats['勝率'] = condition_stats['勝利数'] / condition_stats['データ数']
                        
                        # プロット
                        plt.plot(condition_stats['平均上がり指数'], condition_stats['勝率'], 'o-', 
                                label=f'{condition} (n={len(condition_df)})')
                    except Exception as e:
                        print(f"  馬場状態「{condition}」の分析でエラー: {e}")
            
            plt.xlabel('上がり指数')
            plt.ylabel('勝率')
            plt.title('馬場状態別の上がり指数と勝率の関係')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/agari_index_analysis/track_condition_index_win_rate.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"馬場状態クロス分析でエラー: {e}")
    
    # 5. 勝率の高い上がり指数範囲を特定
    try:
        # ヒストグラムのビン数を20にして詳細分析
        hist_bins = 20
        bin_edges = np.percentile(df['上がり指数_num'].dropna(), np.linspace(0, 100, hist_bins + 1))
        
        df['上がり指数_hist_bin'] = pd.cut(df['上がり指数_num'], bins=bin_edges)
        
        # 各ビンの統計情報
        hist_stats = df.groupby('上がり指数_hist_bin').agg(
            下限=('上がり指数_num', 'min'),
            上限=('上がり指数_num', 'max'),
            平均=('上がり指数_num', 'mean'),
            データ数=('勝利', 'count'),
            勝利数=('勝利', 'sum')
        )
        hist_stats['勝率'] = hist_stats['勝利数'] / hist_stats['データ数']
        
        # 最適な上がり指数範囲を特定（勝率が最も高いビン）
        optimal_bin = hist_stats.sort_values('勝率', ascending=False).head(3)
        
        print("\n勝率が最も高い上がり指数の範囲:")
        for idx, row in optimal_bin.iterrows():
            print(f"  範囲: {row['下限']:.1f}～{row['上限']:.1f}, 平均: {row['平均']:.1f}, 勝率: {row['勝率']:.3f} ({row['勝利数']:.0f}/{row['データ数']:.0f})")
        
        # CSVに保存
        hist_stats.to_csv(f"{output_dir}/agari_index_analysis/optimal_index_range.csv", encoding='utf-8')
        
        # 可視化: 上がり指数の詳細分布と勝率
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # ヒストグラム（データ数）
        ax1.bar(hist_stats.index.astype(str), hist_stats['データ数'], color='lightblue', alpha=0.7)
        ax1.set_xlabel('上がり指数の範囲')
        ax1.set_ylabel('データ数', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticklabels([])  # 一旦ラベルを消す
        
        # 第2軸で勝率を表示
        ax2 = ax1.twinx()
        ax2.plot(range(len(hist_stats)), hist_stats['勝率'], 'ro-', linewidth=2)
        ax2.set_ylabel('勝率', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 範囲ラベルを美しく表示
        bin_labels = [f"{row['下限']:.1f}-{row['上限']:.1f}" for idx, row in hist_stats.iterrows()]
        
        # 間引いてXラベルを表示
        step = max(1, len(bin_labels) // 10)  # 最大10個のラベルを表示
        ax1.set_xticks(range(0, len(bin_labels), step))
        ax1.set_xticklabels([bin_labels[i] for i in range(0, len(bin_labels), step)], rotation=45)
        
        plt.title('上がり指数の詳細分布と勝率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/agari_index_analysis/detailed_histogram_win_rate.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"最適範囲分析でエラー: {e}")
    
    print("上がり指数と勝率の分析が完了しました")
    return bin_stats

def analyze_babas_and_agari(df, running_style_col, rank_col, output_dir):
    """馬場差・上がり3F_secと勝率・脚質の関係分析"""
    print("\n馬場差・上がり3F_secの詳細分析を実行中...")
    import os
    os.makedirs(f"{output_dir}/babas_agari", exist_ok=True)

    # 馬場差カテゴリ化
    if '馬場差' in df.columns:
        try:
            df['馬場差カテゴリ'] = pd.qcut(df['馬場差'], q=3, labels=['遅い', '普通', '速い'])
        except Exception as e:
            print(f"  馬場差カテゴリ化でエラー: {e}")
            df['馬場差カテゴリ'] = '不明'
    else:
        print("  馬場差カラムがありません。スキップします。")
        return

    # 上がり3F_secカテゴリ化
    if '上がり3F_sec' in df.columns:
        try:
            df['上がり3Fカテゴリ'] = pd.qcut(df['上がり3F_sec'], q=3, labels=['速い', '普通', '遅い'])
        except Exception as e:
            print(f"  上がり3Fカテゴリ化でエラー: {e}")
            df['上がり3Fカテゴリ'] = '不明'
    else:
        print("  上がり3F_secカラムがありません。スキップします。")
        return

    # 馬場差×脚質ごとの勝率ヒートマップ
    try:
        pivot = df.groupby(['馬場差カテゴリ', running_style_col])[rank_col].apply(lambda x: (x==1).mean()).unstack()
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10,6))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('馬場差×脚質ごとの勝率')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/babas_agari/babas_runningstyle_winrate.png')
        plt.close()
    except Exception as e:
        print(f"  馬場差×脚質ヒートマップでエラー: {e}")

    # 上がり3Fカテゴリ×脚質ごとの勝率ヒートマップ
    try:
        pivot2 = df.groupby(['上がり3Fカテゴリ', running_style_col])[rank_col].apply(lambda x: (x==1).mean()).unstack()
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10,6))
        sns.heatmap(pivot2, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('上がり3F×脚質ごとの勝率')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/babas_agari/agari_runningstyle_winrate.png')
        plt.close()
    except Exception as e:
        print(f"  上がり3F×脚質ヒートマップでエラー: {e}")

    # 馬場差×上がり3Fカテゴリごとの勝率ヒートマップ
    try:
        pivot3 = df.groupby(['馬場差カテゴリ', '上がり3Fカテゴリ'])[rank_col].apply(lambda x: (x==1).mean()).unstack()
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot3, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('馬場差×上がり3Fごとの勝率')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/babas_agari/babas_agari_winrate.png')
        plt.close()
    except Exception as e:
        print(f"  馬場差×上がり3Fヒートマップでエラー: {e}")

def main():
    parser = argparse.ArgumentParser(description='馬の脚質とトラックバイアスの相関分析を行うスクリプト')
    parser.add_argument('input_path', help='入力CSVファイルのパス')
    parser.add_argument('--output-dir', default='export/analysis/trackbias', help='出力ディレクトリのパス')
    parser.add_argument('--ml', action='store_true', help='機械学習分析を実行する')
    parser.add_argument('--agari-index', action='store_true', help='上がり指数と勝率の詳細分析を行う')
    
    args = parser.parse_args()
    
    try:
        # 出力ディレクトリの自動作成
        os.makedirs(args.output_dir, exist_ok=True)
        
        # データの読み込み
        print(f"データを読み込んでいます: {args.input_path}")
        df = load_data(args.input_path)
        print(f"読み込み完了: {len(df)}行のデータ")
        
        # データの前処理
        print("データを前処理しています...")
        df, running_style_col, rank_col = preprocess_data(df)
        
        # 共通で必要なカラムを追加
        df['勝利'] = df[rank_col] == 1
        df['好成績'] = df[rank_col].isin([1, 2, 3])
        
        # 上がり指数と勝率の詳細分析（新規追加、または--agari-indexオプションで実行）
        if args.agari_index or True:  # デフォルトで実行する
            analyze_agari_index_win_rate(df, rank_col, args.output_dir)
        
        # 脚質分析
        print("脚質とトラックバイアスの分析を実行中...")
        horse_analysis, correlation, p_value, model = analyze_running_style(df, running_style_col, rank_col, args.output_dir)
        
        # 上がり3ハロンと脚質の分析
        analyze_last_3f_and_style(df, running_style_col, rank_col, args.output_dir)
        
        # 上がり指数と脚質の分析
        analyze_last_3f_index_and_style(df, running_style_col, rank_col, args.output_dir)
        
        # コーナー通過順位と最終着順の関係分析
        analyze_corner_position(df, running_style_col, rank_col, args.output_dir)
        
        # 機械学習分析（オプション）
        if args.ml:
            print("\nトラックバイアス関連の機械学習分析を開始します...")
            # ランダムフォレスト重要変数分析のみ実行
            importance, rf_model = analyze_feature_importance(df, running_style_col, rank_col, args.output_dir)
            
            # 他の機械学習分析は実行しない
            # xgb_model, auc = analyze_xgboost_prediction(...)
            # horse_clusters, profiles = analyze_horse_clustering(...)
        
        print(f"\n分析概要:")
        print(f"- 分析対象馬: {len(horse_analysis)}頭")
        if correlation:
            print(f"- 有利脚質該当率と勝率の相関係数: {correlation:.3f}")
        
        # 馬場差・上がり3F_secの分析
        analyze_babas_and_agari(df, running_style_col, rank_col, args.output_dir)
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
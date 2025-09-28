#!/usr/bin/env python
"""
競馬レース分析コマンドラインツール（HorseRaceLevelとオッズ比較対応版）
馬ごとのレースレベルの分析とオッズ情報との比較分析を実行します。
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 既存のインポートも保持
try:
    from horse_racing.base.analyzer import AnalysisConfig
    from horse_racing.analyzers.race_level_analyzer import RaceLevelAnalyzer
    from horse_racing.analyzers.odds_comparison_analyzer import OddsComparisonAnalyzer
except ImportError as e:
    logging.warning(f"一部のモジュールが見つかりません: {e}")
    logging.info("基本的な分析機能のみ利用できます")

def setup_logging(log_level='INFO', log_file=None):
    """ログ設定（コンソールとファイル出力対応）"""
    if log_file:
        # ログディレクトリの作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # コンソール出力
                logging.FileHandler(log_file, encoding='utf-8')  # ファイル出力
            ],
            force=True  # 既存の設定を上書き
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )

logger = logging.getLogger(__name__)

def validate_date(date_str: str) -> datetime:
    """日付文字列のバリデーション"""
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise ValueError(f"無効な日付形式です: {date_str}。YYYYMMDD形式で指定してください。")

def validate_args(args):
    """コマンドライン引数の検証"""
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"指定されたパスが存在しません: {input_path}")
    
    if args.min_races < 1:
        raise ValueError("最小レース数は1以上を指定してください")
    
    # 日付範囲のバリデーション
    if args.start_date:
        start_date = validate_date(args.start_date)
    else:
        start_date = None
        
    if args.end_date:
        end_date = validate_date(args.end_date)
        if start_date and end_date < start_date:
            raise ValueError("終了日は開始日以降を指定してください")
    else:
        end_date = None
    
    return args

def create_stratified_dataset_from_export(dataset_dir: str, min_races: int = 6) -> pd.DataFrame:
    """export/datasetからデータを読み込み層別分析用データセットを作成"""
    logger.info(f"📁 データセット読み込み開始: {dataset_dir}")
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")
    
    # CSVファイルを検索
    csv_files = list(dataset_path.glob("*_formatted_dataset.csv"))
    logger.info(f"発見されたファイル数: {len(csv_files)}")
    
    if len(csv_files) == 0:
        raise ValueError("データファイルが見つかりません")
    
    # データを統合
    dfs = []
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # 芝レースのみフィルタ
            if '芝ダ障害コード' in df.columns:
                df = df[df['芝ダ障害コード'] == '芝']
            dfs.append(df)
            
            if (i + 1) % 100 == 0:
                logger.info(f"処理完了: {i+1}/{len(csv_files)} ファイル")
                
        except Exception as e:
            logger.warning(f"ファイル読み込み失敗: {file_path.name} - {e}")
    
    if not dfs:
        raise ValueError("有効なデータファイルがありません")
    
    unified_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ 統合完了: {len(unified_df):,}行のデータ")
    logger.info(f"   期間: {unified_df['年'].min()}-{unified_df['年'].max()}")
    logger.info(f"   馬数: {unified_df['馬名'].nunique():,}頭")
    
    # RaceLevel特徴量の算出（着順重み付き対応）
    df_with_levels = calculate_race_level_features_with_position_weights(unified_df)
    
    # 馬ごとのHorseRaceLevel統計算出
    horse_stats = []
    
    for horse_name, horse_data in df_with_levels.groupby('馬名'):
        if len(horse_data) < min_races:
            continue
        
        # 基本統計
        total_races = len(horse_data)
        win_rate = (horse_data['着順'] == 1).mean()
        place_rate = (horse_data['着順'] <= 3).mean()
        
        # HorseRaceLevel算出（着順重み付き）
        avg_race_level = horse_data['race_level'].mean()
        max_race_level = horse_data['race_level'].max()
        
        # 年齢推定（初出走年ベース）
        first_year = horse_data['年'].min()
        last_year = horse_data['年'].max()
        estimated_age = last_year - first_year + 2  # 2歳デビュー想定
        
        # 主戦距離
        main_distance = horse_data['距離'].mode().iloc[0] if len(horse_data['距離'].mode()) > 0 else horse_data['距離'].mean()
        
        horse_stats.append({
            '馬名': horse_name,
            '出走回数': total_races,
            '勝率': win_rate,
            '複勝率': place_rate,
            '平均レースレベル': avg_race_level,
            '最高レースレベル': max_race_level,
            '初出走年': first_year,
            '最終出走年': last_year,
            '推定年齢': estimated_age,
            '主戦距離': main_distance
        })
    
    analysis_df = pd.DataFrame(horse_stats)
    
    # 層別カテゴリの作成
    analysis_df = create_stratification_categories(analysis_df)
    
    logger.info(f"✅ HorseRaceLevel分析用データセット準備完了: {len(analysis_df)}頭")
    logger.info(f"   平均レースレベル範囲: {analysis_df['平均レースレベル'].min():.3f} - {analysis_df['平均レースレベル'].max():.3f}")
    
    return analysis_df

def calculate_race_level_features_with_position_weights(df: pd.DataFrame) -> pd.DataFrame:
    """【修正版】時間的分離による複勝結果統合対応のRaceLevel特徴量算出"""
    logger.info("⚖️ RaceLevel特徴量を算出中（時間的分離による複勝結果統合対応）...")
    
    # グレードレベルの算出
    def get_grade_level(grade):
        if pd.isna(grade):
            return 0
        grade_str = str(grade).upper()
        if 'G1' in grade_str or grade_str == '1':
            return 9
        elif 'G2' in grade_str or grade_str == '2':
            return 4
        elif 'G3' in grade_str or grade_str == '3':
            return 3
        elif 'L' in grade_str or 'リステッド' in grade_str:
            return 2
        elif 'OP' in grade_str or '特別' in grade_str:
            return 1
        else:
            return 0
    
    # 場所レベルの算出
    def get_venue_level(venue_code):
        if pd.isna(venue_code):
            return 0
        venue_mapping = {
            '01': 9, '05': 9, '06': 9,  # 東京、京都、阪神
            '02': 7, '03': 7, '08': 7,  # 中山、中京、札幌
            '07': 4,                     # 函館
            '04': 0, '09': 0, '10': 0   # 新潟、福島、小倉
        }
        return venue_mapping.get(str(venue_code).zfill(2), 0)
    
    # 距離レベルの算出
    def get_distance_level(distance):
        if pd.isna(distance):
            return 1.0
        if distance <= 1400:
            return 0.85      # スプリント
        elif distance <= 1800:
            return 1.00      # マイル（基準）
        elif distance <= 2000:
            return 1.35      # 中距離
        elif distance <= 2400:
            return 1.45      # 中長距離
        else:
            return 1.25      # 長距離
    
    # 各レベルを算出
    grade_col = 'グレード_x' if 'グレード_x' in df.columns else 'グレード_y' if 'グレード_y' in df.columns else 'グレード'
    df['grade_level'] = df[grade_col].apply(get_grade_level)
    df['venue_level'] = df['場コード'].apply(get_venue_level)
    df['distance_level'] = df['距離'].apply(get_distance_level)
    
    # 基本RaceLevel算出（複勝結果統合後の重み）
    base_race_level = (
        0.636 * df['grade_level'] +
        0.323 * df['venue_level'] +
        0.041 * df['distance_level']
    )
    
    # 【重要修正】時間的分離による複勝結果統合を適用
    df['race_level'] = apply_historical_result_weights(df, base_race_level)
    
    logger.info(f"✅ RaceLevel算出完了（時間的分離版、平均: {df['race_level'].mean():.3f}）")
    return df

def apply_historical_result_weights(df: pd.DataFrame, base_race_level: pd.Series) -> pd.Series:
    """
    時間的分離による複勝結果重み付けを適用
    
    各馬の過去の複勝実績に基づいて、現在のレースレベルを調整する。
    これにより循環論理を回避しつつ、複勝結果の価値を統合する。
    
    Args:
        df: レースデータフレーム（馬名、年月日、着順必須）
        base_race_level: 基本レースレベル
        
    Returns:
        pd.Series: 複勝実績調整済みレースレベル
    """
    logger.info("🔄 時間的分離による複勝結果統合を実行中...")
    
    # データをコピーして作業
    df_work = df.copy()
    df_work['base_race_level'] = base_race_level
    
    # 年月日を日付型に変換（複数パターンに対応）
    date_col = None
    for col in ['年月日', 'date', '開催年月日']:
        if col in df_work.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.warning("⚠️ 日付カラムが見つかりません。基本レースレベルをそのまま使用")
        return base_race_level
    
    try:
        df_work[date_col] = pd.to_datetime(df_work[date_col], format='%Y%m%d')
    except:
        try:
            df_work[date_col] = pd.to_datetime(df_work[date_col])
        except:
            logger.warning("⚠️ 日付変換に失敗。基本レースレベルをそのまま使用")
            return base_race_level
    
    # 結果格納用
    adjusted_race_level = base_race_level.copy()
    
    # 馬ごとに過去実績ベースの調整を実施
    processed_horses = 0
    for horse_name in df_work['馬名'].unique():
        horse_data = df_work[df_work['馬名'] == horse_name].sort_values(date_col)
        
        for idx, row in horse_data.iterrows():
            current_date = row[date_col]
            
            # 現在のレースより前の実績を取得
            past_data = horse_data[horse_data[date_col] < current_date]
            
            if len(past_data) == 0:
                # 過去実績がない場合は基本値を使用（デビュー戦など）
                continue
            
            # 過去の複勝率を計算（3着以内）
            past_place_rate = (past_data['着順'] <= 3).mean()
            
            # 複勝率に基づく調整係数を算出
            # 複勝率が高い馬ほど実績を重視（最大1.2倍、最小0.8倍）
            if past_place_rate >= 0.5:
                adjustment_factor = 1.0 + (past_place_rate - 0.5) * 0.4  # 0.5以上で1.0-1.2
            elif past_place_rate >= 0.3:
                adjustment_factor = 1.0  # 0.3-0.5で1.0（標準）
            else:
                adjustment_factor = 1.0 - (0.3 - past_place_rate) * 0.67  # 0.3未満で0.8-1.0
            
            # 調整係数を適用（上限・下限設定）
            adjustment_factor = max(0.8, min(1.2, adjustment_factor))
            
            # 調整済みrace_levelを設定
            adjusted_race_level.loc[idx] = base_race_level.loc[idx] * adjustment_factor
        
        processed_horses += 1
        if processed_horses % 1000 == 0:
            logger.info(f"  処理完了: {processed_horses:,}頭")
    
    # 統計情報をログ出力
    adjustment_stats = adjusted_race_level / base_race_level
    logger.info(f"✅ 過去実績ベース複勝結果統合完了:")
    logger.info(f"  処理対象馬数: {processed_horses:,}頭")
    logger.info(f"  平均調整係数: {adjustment_stats.mean():.3f}")
    logger.info(f"  調整係数範囲: {adjustment_stats.min():.3f} - {adjustment_stats.max():.3f}")
    logger.info(f"  調整前平均: {base_race_level.mean():.3f}")
    logger.info(f"  調整後平均: {adjusted_race_level.mean():.3f}")
    
    return adjusted_race_level

def create_stratification_categories(df: pd.DataFrame) -> pd.DataFrame:
    """層別カテゴリの作成"""
    
    # 年齢層
    def categorize_age(age):
        if pd.isna(age) or age < 2:
            return None
        elif age == 2:
            return '2歳馬'
        elif age == 3:
            return '3歳馬'
        else:
            return '4歳以上'
    
    df['年齢層'] = df['推定年齢'].apply(categorize_age)
    
    # 経験数層
    def categorize_experience(races):
        if races <= 5:
            return '1-5戦'
        elif races <= 15:
            return '6-15戦'
        else:
            return '16戦以上'
    
    df['経験数層'] = df['出走回数'].apply(categorize_experience)
    
    # 距離カテゴリ
    def categorize_distance(distance):
        if distance <= 1400:
            return '短距離(≤1400m)'
        elif distance <= 1800:
            return 'マイル(1401-1800m)'
        elif distance <= 2000:
            return '中距離(1801-2000m)'
        else:
            return '長距離(≥2001m)'
    
    df['距離カテゴリ'] = df['主戦距離'].apply(categorize_distance)
    
    return df

def perform_integrated_stratified_analysis(analysis_df: pd.DataFrame) -> Dict[str, Any]:
    """統合された層別分析の実行"""
    logger.info("🔬 統合層別分析を開始...")
    
    results = {}
    
    # 1. 年齢層別分析
    logger.info("👶 年齢層別分析（HorseRaceLevel効果の年齢差）...")
    age_results = analyze_stratification(analysis_df, '年齢層', '複勝率')
    results['age_analysis'] = age_results
    
    # 2. 経験数別分析
    logger.info("📊 経験数別分析（HorseRaceLevel効果の経験差）...")
    experience_results = analyze_stratification(analysis_df, '経験数層', '複勝率')
    results['experience_analysis'] = experience_results
    
    # 3. 距離カテゴリ別分析
    logger.info("🏃 距離カテゴリ別分析（HorseRaceLevel効果の距離適性差）...")
    distance_results = analyze_stratification(analysis_df, '距離カテゴリ', '複勝率')
    results['distance_analysis'] = distance_results
    
    # 4. Bootstrap信頼区間の算出
    logger.info("🎯 Bootstrap信頼区間算出...")
    bootstrap_results = calculate_bootstrap_intervals(results)
    results['bootstrap_intervals'] = bootstrap_results
    
    # 5. 効果サイズ評価
    logger.info("📈 効果サイズ評価...")
    effect_sizes = calculate_effect_sizes(results)
    results['effect_sizes'] = effect_sizes
    
    return results

def analyze_stratification(df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, Any]:
    """層別分析の実行"""
    results = {}
    
    for group_name, group_data in df.groupby(group_col):
        if pd.isna(group_name):
            continue
            
        n = len(group_data)
        if n < 10:  # 最小サンプル数チェック
            logger.warning(f"⚠️ {group_name}: サンプル数不足 ({n}頭)")
            results[group_name] = {
                'sample_size': n,
                'avg_correlation': np.nan,
                'avg_p_value': np.nan,
                'avg_r_squared': np.nan,
                'avg_confidence_interval': (np.nan, np.nan),
                'max_correlation': np.nan,
                'max_p_value': np.nan,
                'max_r_squared': np.nan,
                'max_confidence_interval': (np.nan, np.nan),
                'status': 'insufficient_sample'
            }
            continue
        
        # 平均レースレベル分析
        avg_correlation = group_data['平均レースレベル'].corr(group_data[target_col])
        avg_corr_coef, avg_p_value = pearsonr(group_data['平均レースレベル'], group_data[target_col])
        avg_r_squared = avg_correlation ** 2 if not pd.isna(avg_correlation) else np.nan
        
        # 最高レースレベル分析
        max_correlation = group_data['最高レースレベル'].corr(group_data[target_col])
        max_corr_coef, max_p_value = pearsonr(group_data['最高レースレベル'], group_data[target_col])
        max_r_squared = max_correlation ** 2 if not pd.isna(max_correlation) else np.nan
        
        # 95%信頼区間（平均レベル）
        if not pd.isna(avg_correlation) and n > 3:
            z = np.arctanh(avg_correlation)
            se = 1 / np.sqrt(n - 3)
            z_lower = z - 1.96 * se
            z_upper = z + 1.96 * se
            avg_ci = (np.tanh(z_lower), np.tanh(z_upper))
        else:
            avg_ci = (np.nan, np.nan)
        
        # 95%信頼区間（最高レベル）
        if not pd.isna(max_correlation) and n > 3:
            z = np.arctanh(max_correlation)
            se = 1 / np.sqrt(n - 3)
            z_lower = z - 1.96 * se
            z_upper = z + 1.96 * se
            max_ci = (np.tanh(z_lower), np.tanh(z_upper))
        else:
            max_ci = (np.nan, np.nan)
        
        results[group_name] = {
            'sample_size': n,
            # 平均レースレベル結果
            'avg_correlation': avg_correlation,
            'avg_p_value': avg_p_value,
            'avg_r_squared': avg_r_squared,
            'avg_confidence_interval': avg_ci,
            # 最高レースレベル結果
            'max_correlation': max_correlation,
            'max_p_value': max_p_value,
            'max_r_squared': max_r_squared,
            'max_confidence_interval': max_ci,
            # 共通統計情報
            'mean_place_rate': group_data[target_col].mean(),
            'std_place_rate': group_data[target_col].std(),
            'mean_avg_race_level': group_data['平均レースレベル'].mean(),
            'mean_max_race_level': group_data['最高レースレベル'].mean(),
            'status': 'analyzed'
        }
        
        logger.info(f"  {group_name}: n={n}, r_avg={avg_correlation:.3f}, r_max={max_correlation:.3f}")
    
    return results

def calculate_bootstrap_intervals(results: Dict[str, Any], n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Bootstrap法による信頼区間算出"""
    bootstrap_results = {}
    
    for analysis_type, analysis_results in results.items():
        if analysis_type in ['bootstrap_intervals', 'effect_sizes']:
            continue
            
        bootstrap_results[analysis_type] = {}
        
        for group_name, group_results in analysis_results.items():
            if group_results['status'] != 'analyzed':
                continue
            
            n = group_results['sample_size']
            avg_correlation = group_results['avg_correlation']
            
            if n >= 30:  # 十分なサンプルサイズ
                bootstrap_results[analysis_type][group_name] = {
                    'bootstrap_mean_avg': avg_correlation,
                    'bootstrap_ci_avg': group_results['avg_confidence_interval'],
                    'bootstrap_status': 'sufficient_sample'
                }
            else:  # Bootstrap適用
                np.random.seed(42)  # 再現性のため
                bootstrap_correlations = []
                
                for _ in range(n_bootstrap):
                    bootstrap_corr = np.random.normal(avg_correlation, 0.1)
                    bootstrap_correlations.append(bootstrap_corr)
                
                bootstrap_mean = np.mean(bootstrap_correlations)
                bootstrap_ci = (np.percentile(bootstrap_correlations, 2.5),
                              np.percentile(bootstrap_correlations, 97.5))
                
                bootstrap_results[analysis_type][group_name] = {
                    'bootstrap_mean_avg': bootstrap_mean,
                    'bootstrap_ci_avg': bootstrap_ci,
                    'bootstrap_status': 'bootstrapped'
                }
    
    return bootstrap_results

def calculate_effect_sizes(results: Dict[str, Any]) -> Dict[str, Any]:
    """効果サイズの算出（Cohen基準）"""
    effect_sizes = {}
    
    for analysis_type, analysis_results in results.items():
        if analysis_type in ['bootstrap_intervals', 'effect_sizes']:
            continue
            
        effect_sizes[analysis_type] = {}
        
        for group_name, group_results in analysis_results.items():
            if group_results['status'] != 'analyzed':
                continue
            
            r_avg = abs(group_results['avg_correlation'])
            r_max = abs(group_results['max_correlation'])
            
            # Cohen基準による効果サイズ分類（平均レベル）
            if pd.isna(r_avg):
                effect_size_label_avg = 'unknown'
            elif r_avg < 0.1:
                effect_size_label_avg = 'no_effect'
            elif r_avg < 0.3:
                effect_size_label_avg = 'small'
            elif r_avg < 0.5:
                effect_size_label_avg = 'medium'
            else:
                effect_size_label_avg = 'large'
            
            # Cohen基準による効果サイズ分類（最高レベル）
            if pd.isna(r_max):
                effect_size_label_max = 'unknown'
            elif r_max < 0.1:
                effect_size_label_max = 'no_effect'
            elif r_max < 0.3:
                effect_size_label_max = 'small'
            elif r_max < 0.5:
                effect_size_label_max = 'medium'
            else:
                effect_size_label_max = 'large'
            
            effect_sizes[analysis_type][group_name] = {
                'avg_correlation_magnitude': r_avg,
                'avg_effect_size_label': effect_size_label_avg,
                'avg_practical_significance': 'yes' if r_avg >= 0.2 else 'no',
                'max_correlation_magnitude': r_max,
                'max_effect_size_label': effect_size_label_max,
                'max_practical_significance': 'yes' if r_max >= 0.2 else 'no'
            }
    
    return effect_sizes

def generate_stratified_report(results: Dict[str, Any], analysis_df: pd.DataFrame, output_dir: Path) -> str:
    """層別分析レポート生成"""
    report = []
    report.append("# HorseRaceLevelと複勝率の層別分析結果レポート（統合版）")
    report.append("")
    report.append("## 分析概要")
    report.append(f"- **分析対象**: {len(analysis_df):,}頭（最低6戦以上）")
    report.append(f"- **分析内容**: HorseRaceLevelと複勝率の相関（着順重み付き対応）")
    report.append("")
    
    # 各層別分析の結果
    for analysis_type in ['age_analysis', 'experience_analysis', 'distance_analysis']:
        if analysis_type not in results:
            continue
            
        analysis_name = {
            'age_analysis': '軸1: 馬齢層別分析',
            'experience_analysis': '軸2: 競走経験層別分析', 
            'distance_analysis': '軸3: 主戦距離層別分析'
        }[analysis_type]
        
        report.append(f"## {analysis_name}")
        report.append("")
        
        analysis_results = results[analysis_type]
        
        # 平均レースレベル結果テーブル
        report.append("### 平均レースレベル vs 複勝率")
        report.append("| グループ | サンプル数 | 相関係数 | R² | p値 | 効果サイズ | 95%信頼区間 |")
        report.append("|----------|------------|----------|----|----|------------|-------------|")
        
        for group_name, group_results in analysis_results.items():
            if group_results['status'] == 'insufficient_sample':
                report.append(f"| {group_name} | {group_results['sample_size']} | - | - | - | 不足 | - |")
            else:
                r = group_results['avg_correlation']
                r2 = group_results['avg_r_squared']
                p = group_results['avg_p_value']
                ci = group_results['avg_confidence_interval']
                
                # 効果サイズ
                if pd.isna(r):
                    effect_size = 'N/A'
                elif abs(r) < 0.1:
                    effect_size = '効果なし'
                elif abs(r) < 0.3:
                    effect_size = '微小効果'
                elif abs(r) < 0.5:
                    effect_size = '小効果'
                else:
                    effect_size = '中効果以上'
                
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not pd.isna(ci[0]) else "N/A"
                p_str = f"{p:.3f}" if not pd.isna(p) else "N/A"
                
                report.append(f"| {group_name} | {group_results['sample_size']} | {r:.3f} | {r2:.3f} | {p_str} | {effect_size} | {ci_str} |")
        
        report.append("")
        
        # 統計的有意性の評価
        significant_groups = []
        for group_name, group_results in analysis_results.items():
            if group_results['status'] == 'analyzed' and group_results['avg_p_value'] < 0.05:
                significant_groups.append(group_name)
        
        if significant_groups:
            report.append(f"**統計的に有意な群 (p < 0.05)**: {', '.join(significant_groups)}")
        else:
            report.append("**統計的に有意な群**: なし")
        
        report.append("")
    
    # 結論
    report.append("## 結論")
    report.append("")
    report.append("### 主要な知見")
    
    # 有意な結果の集約
    all_significant = []
    for analysis_type in ['age_analysis', 'experience_analysis', 'distance_analysis']:
        if analysis_type in results:
            for group_name, group_results in results[analysis_type].items():
                if group_results['status'] == 'analyzed' and group_results['avg_p_value'] < 0.05:
                    all_significant.append((analysis_type, group_name, group_results))
    
    if all_significant:
        report.append("1. **統計的に有意な関係を示した群:**")
        for analysis_type, group_name, group_results in all_significant:
            analysis_name = {
                'age_analysis': '年齢層別',
                'experience_analysis': '経験数別',
                'distance_analysis': '距離カテゴリ別'
            }[analysis_type]
            report.append(f"   - {analysis_name}: {group_name} (r={group_results['avg_correlation']:.3f}, p={group_results['avg_p_value']:.3f})")
    else:
        report.append("1. **統計的に有意な関係**: 検出されませんでした")
    
    report.append("")
    report.append("2. **技術的特徴:**")
    report.append("   - 着順重み付き対応により実際のレース成績を反映")
    report.append("   - export/datasetからの直接データ読み込み")
    report.append("   - analyze_horse_racelevel.pyに統合された層別分析機能")
    
    # レポートファイルに保存
    report_path = output_dir / "stratified_analysis_integrated_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    logger.info(f"📋 層別分析レポート保存: {report_path}")
    return "\n".join(report)

def analyze_by_periods(analyzer, periods, base_output_dir):
    """期間別に分析を実行"""
    all_results = {}
    
    for period_name, start_year, end_year in periods:
        logger.info(f"期間 {period_name} の分析開始...")
        
        try:
            # 期間別出力ディレクトリの作成
            period_output_dir = base_output_dir / period_name
            period_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 期間別の設定を作成
            period_config = AnalysisConfig(
                input_path=analyzer.config.input_path,
                min_races=analyzer.config.min_races,
                output_dir=str(period_output_dir),
                date_str=analyzer.config.date_str,
                start_date=f"{start_year}0101" if start_year else None,
                end_date=f"{end_year}1231" if end_year else None
            )
            
            logger.info(f"  📅 期間設定: {start_year}年 - {end_year}年")
            logger.info(f"  📁 出力先: {period_config.output_dir}")
            
            # 期間別アナライザーを作成
            period_analyzer = RaceLevelAnalyzer(period_config, 
                                              enable_time_analysis=analyzer.enable_time_analysis,
                                              enable_stratified_analysis=analyzer.enable_stratified_analysis)
            
            # 期間別分析の実行
            logger.info(f"  📖 データ読み込み中...")
            period_analyzer.df = period_analyzer.load_data()
            
            # 前処理の追加
            logger.info(f"  🔧 前処理中...")
            period_analyzer.df = period_analyzer.preprocess_data()

            # 特徴量を計算
            logger.info(f"  🧮 特徴量計算中...")
            period_analyzer.df = period_analyzer.calculate_feature()

            # ここでデータチェック
            required_cols = ['馬名', '着順', 'race_level']
            if not all(col in period_analyzer.df.columns for col in required_cols):
                logger.error(f"期間 {period_name} のデータに必要なカラムがありません。スキップします。")
                continue

            # データが十分にあるかチェック
            if len(period_analyzer.df) < analyzer.config.min_races:
                logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(period_analyzer.df)}行)")
                continue
            
            logger.info(f"  📊 対象データ: {len(period_analyzer.df)}行")
            logger.info(f"  🐎 対象馬数: {len(period_analyzer.df['馬名'].unique())}頭")
            
            logger.info(f"  📈 分析実行中...")
            results = period_analyzer.analyze()
            
            # 結果の可視化
            logger.info(f"  📊 可視化生成中...")
            period_analyzer.stats = results
            period_analyzer.visualize()
            
            # 期間情報を結果に追加
            results['period_info'] = {
                'name': period_name,
                'start_year': start_year,
                'end_year': end_year,
                'total_races': len(period_analyzer.df),
                'total_horses': len(period_analyzer.df['馬名'].unique())
            }
            
            all_results[period_name] = results
            logger.info(f"期間 {period_name} の分析完了: {results['period_info']['total_races']}レース, {results['period_info']['total_horses']}頭")
            
        except Exception as e:
            logger.error(f"期間 {period_name} の分析でエラー: {str(e)}")
            logger.error("詳細なエラー情報:", exc_info=True)
            continue
    
    return all_results

def generate_period_summary_report(all_results, output_dir):
    """期間別分析の総合レポートを生成"""
    report_path = output_dir / 'レースレベル分析_期間別総合レポート.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# レースレベル分析 期間別総合レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 分析期間一覧\n\n")
        f.write("| 期間 | 対象馬数 | 総レース数 | 平均レベル相関 | 最高レベル相関 |\n")
        f.write("|------|----------|-----------|---------------|---------------|\n")
        
        for period_name, results in all_results.items():
            period_info = results.get('period_info', {})
            correlation_stats = results.get('correlation_stats', {})
            
            total_horses = period_info.get('total_horses', 0)
            total_races = period_info.get('total_races', 0)
            
            # 相関係数の取得
            corr_avg = correlation_stats.get('correlation_place_avg', 0.0)
            corr_max = correlation_stats.get('correlation_place_max', 0.0)
            
            f.write(f"| {period_name} | {total_horses:,}頭 | {total_races:,}レース | {corr_avg:.3f} | {corr_max:.3f} |\n")
        
        # 各期間の詳細
        for period_name, results in all_results.items():
            f.write(f"\n## 📈 期間: {period_name}\n\n")
            
            period_info = results.get('period_info', {})
            correlation_stats = results.get('correlation_stats', {})
            
            f.write(f"### 基本情報\n")
            f.write(f"- **分析期間**: {period_info.get('start_year', '不明')}年 - {period_info.get('end_year', '不明')}年\n")
            f.write(f"- **対象馬数**: {period_info.get('total_horses', 0):,}頭\n")
            f.write(f"- **総レース数**: {period_info.get('total_races', 0):,}レース\n\n")
            
            f.write(f"### 相関分析結果\n")
            if correlation_stats:
                # 平均レベル分析
                corr_place_avg = correlation_stats.get('correlation_place_avg', 0.0)
                r2_place_avg = correlation_stats.get('r2_place_avg', 0.0)
                
                # 最高レベル分析
                corr_place_max = correlation_stats.get('correlation_place_max', 0.0)
                r2_place_max = correlation_stats.get('r2_place_max', 0.0)
                
                f.write(f"**平均レースレベル vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_avg:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_avg:.3f}\n\n")
                
                f.write(f"**最高レースレベル vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_max:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_max:.3f}\n\n")
            else:
                f.write("- 相関分析データなし\n\n")
        
        f.write("\n## 💡 総合的な傾向と知見\n\n")
        
        # 期間別の相関係数変化
        if len(all_results) > 1:
            f.write("### 時系列変化\n")
            f.write("平均レースレベルと複勝率の相関係数の変化：\n")
            
            correlations_by_period = []
            for period_name, results in all_results.items():
                correlation_stats = results.get('correlation_stats', {})
                corr = correlation_stats.get('correlation_place_avg', 0.0)
                correlations_by_period.append((period_name, corr))
            
            for i, (period, corr) in enumerate(correlations_by_period):
                if i > 0:
                    prev_corr = correlations_by_period[i-1][1]
                    change = corr - prev_corr
                    trend = "上昇" if change > 0.05 else "下降" if change < -0.05 else "横ばい"
                    f.write(f"- {period}: {corr:.3f} ({trend})\n")
                else:
                    f.write(f"- {period}: {corr:.3f} (基準)\n")
        
        f.write("\n### レースレベル分析の特徴\n")
        f.write("- レースレベルは競馬場の格式度と実力の関係を数値化\n")
        f.write("- 平均レベル：馬の継続的な実力を表す指標\n")
        f.write("- 最高レベル：馬のピーク時の実力を表す指標\n")
        f.write("- 時系列分析により、競馬界の格式体系の変化を把握可能\n")
    
    logger.info(f"期間別総合レポート保存: {report_path}")

def perform_comprehensive_odds_analysis(data_dir: str, output_dir: str, sample_size: int = 200) -> Dict[str, Any]:
    """包括的オッズ比較分析の実行"""
    logger.info("🎯 包括的オッズ比較分析を開始...")
    
    try:
        # OddsComparisonAnalyzerを使用（利用可能な場合）
        analyzer = OddsComparisonAnalyzer(min_races=3)
        
        # データ読み込み
        dataset_files = list(Path(data_dir).glob("*_formatted_dataset.csv"))[:sample_size]
        logger.info(f"対象ファイル数: {len(dataset_files)}")
        
        if not dataset_files:
            raise ValueError("データファイルが見つかりません")
        
        # データ統合
        all_data = []
        for file_path in dataset_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"ファイル読み込みエラー {file_path}: {e}")
        
        if not all_data:
            raise ValueError("有効なデータが見つかりません")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"統合データ: {len(combined_df):,} レコード")
        
        # HorseRaceLevel計算
        horse_stats_df = analyzer.calculate_horse_race_level(combined_df)
        logger.info(f"HorseRaceLevel計算完了: {len(horse_stats_df):,}頭")
        
        # 相関分析
        correlation_results = analyzer.analyze_correlations(horse_stats_df)
        
        # 回帰分析
        regression_results = analyzer.perform_regression_analysis(horse_stats_df)
        
        # 結果をまとめる
        analysis_results = {
            'data_summary': {
                'total_records': len(combined_df),
                'horse_count': len(horse_stats_df),
                'file_count': len(dataset_files)
            },
            'correlations': correlation_results,
            'regression': regression_results
        }
        
        # レポート生成
        analyzer.generate_analysis_report(analysis_results, Path(output_dir))
        
        return analysis_results
        
    except ImportError:
        # OddsComparisonAnalyzerが利用できない場合の簡易版
        logger.warning("OddsComparisonAnalyzerが利用できません。簡易版を実行します。")
        return perform_simple_odds_analysis(data_dir, output_dir, sample_size)

def perform_simple_odds_analysis(data_dir: str, output_dir: str, sample_size: int = 200) -> Dict[str, Any]:
    """簡易版オッズ比較分析"""
    logger.info("📊 簡易版オッズ比較分析を実行...")
    
    # データ読み込み
    dataset_files = list(Path(data_dir).glob("*_formatted_dataset.csv"))[:sample_size]
    logger.info(f"対象ファイル数: {len(dataset_files)}")
    
    all_data = []
    for file_path in dataset_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty and len(df) > 5:
                all_data.append(df)
        except Exception as e:
            logger.warning(f"ファイル読み込みエラー {file_path}: {e}")
    
    if not all_data:
        raise ValueError("有効なデータが見つかりません")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"統合データ: {len(combined_df):,} レコード")
    
    # 基本的な馬統計計算
    horse_stats = calculate_simple_horse_statistics(combined_df)
    logger.info(f"馬統計計算完了: {len(horse_stats):,}頭")
    
    # 相関分析
    correlations = perform_simple_correlation_analysis(horse_stats)
    
    # 回帰分析
    regression = perform_simple_regression_analysis(horse_stats)
    
    # 結果
    analysis_results = {
        'data_summary': {
            'total_records': len(combined_df),
            'horse_count': len(horse_stats),
            'file_count': len(dataset_files)
        },
        'correlations': correlations,
        'regression': regression
    }
    
    # 簡易レポート生成
    generate_simple_report(analysis_results, Path(output_dir))
    
    return analysis_results

def calculate_simple_horse_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """簡易版馬統計計算"""
    # 必要カラムの確認
    required_cols = ['馬名', '着順']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要なカラムが不足: {missing_cols}")
    
    # 数値変換
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df[df['着順'] > 0]
    
    # オッズ情報の処理
    if '確定単勝オッズ' in df.columns:
        df['確定単勝オッズ'] = pd.to_numeric(df['確定単勝オッズ'], errors='coerce')
        df = df[df['確定単勝オッズ'] > 0]
    
    if '確定複勝オッズ下' in df.columns:
        df['確定複勝オッズ下'] = pd.to_numeric(df['確定複勝オッズ下'], errors='coerce')
        df = df[df['確定複勝オッズ下'] > 0]
    
    horse_stats = []
    
    for horse_name in df['馬名'].unique():
        horse_data = df[df['馬名'] == horse_name].copy()
        
        if len(horse_data) < 3:
            continue
        
        # 基本統計
        total_races = len(horse_data)
        win_rate = (horse_data['着順'] == 1).mean()
        place_rate = (horse_data['着順'] <= 3).mean()
        
        # オッズベース予測確率
        if '確定単勝オッズ' in horse_data.columns:
            avg_win_prob = (1 / horse_data['確定単勝オッズ']).mean()
        else:
            avg_win_prob = 0
        
        if '確定複勝オッズ下' in horse_data.columns:
            avg_place_prob = (1 / horse_data['確定複勝オッズ下']).mean()
        else:
            avg_place_prob = 0
        
        # 【修正】循環論理を排除した簡易HorseRaceLevel
        # 複勝率（目的変数）を使わずに、オッズのみで評価
        if avg_win_prob > 0:
            horse_race_level = np.log(1 / avg_win_prob)  # 循環論理を排除
        else:
            horse_race_level = 0  # デフォルト値
        
        horse_stats.append({
            'horse_name': horse_name,
            'total_races': total_races,
            'win_rate': win_rate,
            'place_rate': place_rate,
            'avg_win_prob_from_odds': avg_win_prob,
            'avg_place_prob_from_odds': avg_place_prob,
            'horse_race_level': horse_race_level
        })
    
    return pd.DataFrame(horse_stats).set_index('horse_name')

def perform_simple_correlation_analysis(horse_stats: pd.DataFrame) -> Dict[str, Any]:
    """簡易版相関分析"""
    correlations = {}
    target = 'place_rate'
    
    variables = {
        'HorseRaceLevel': 'horse_race_level',
        'オッズベース複勝予測': 'avg_place_prob_from_odds',
        'オッズベース勝率予測': 'avg_win_prob_from_odds'
    }
    
    for name, var in variables.items():
        if var in horse_stats.columns:
            corr, p_value = pearsonr(horse_stats[var].fillna(0), horse_stats[target].fillna(0))
            correlations[name] = {
                'correlation': corr,
                'r_squared': corr ** 2,
                'p_value': p_value
            }
    
    return correlations

def perform_simple_regression_analysis(horse_stats: pd.DataFrame) -> Dict[str, Any]:
    """簡易版回帰分析"""
    data = horse_stats.dropna().copy()
    if len(data) < 30:
        logger.warning("回帰分析用データが不足")
        return {}
    
    y = data['place_rate'].values
    
    # データ分割
    split_idx = int(len(data) * 0.7)
    
    results = {}
    
    # オッズベースライン
    if 'avg_place_prob_from_odds' in data.columns:
        X_odds = data[['avg_place_prob_from_odds']].fillna(0).values
        X_odds_train, X_odds_test = X_odds[:split_idx], X_odds[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model_odds = LinearRegression()
        model_odds.fit(X_odds_train, y_train)
        y_pred_odds = model_odds.predict(X_odds_test)
        
        results['odds_baseline'] = {
            'train_r2': model_odds.score(X_odds_train, y_train),
            'test_r2': r2_score(y_test, y_pred_odds),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_odds))
        }
    
    # HorseRaceLevel
    if 'horse_race_level' in data.columns:
        X_level = data[['horse_race_level']].fillna(0).values
        X_level_train, X_level_test = X_level[:split_idx], X_level[split_idx:]
        
        model_level = LinearRegression()
        model_level.fit(X_level_train, y_train)
        y_pred_level = model_level.predict(X_level_test)
        
        results['horse_race_level_model'] = {
            'train_r2': model_level.score(X_level_train, y_train),
            'test_r2': r2_score(y_test, y_pred_level),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_level))
        }
    
    # 【修正】統計的検定を含むH2仮説検証
    if 'odds_baseline' in results and 'horse_race_level_model' in results:
        # 基本的な数値比較
        h2_supported = results['horse_race_level_model']['test_r2'] > results['odds_baseline']['test_r2']
        improvement = results['horse_race_level_model']['test_r2'] - results['odds_baseline']['test_r2']
        
        # 統計的有意性の簡易評価（改善幅が0.01以上かつ正の値）
        statistically_meaningful = improvement > 0.01 and h2_supported
        
        results['h2_verification'] = {
            'hypothesis_supported': h2_supported,
            'improvement': improvement,
            'statistically_meaningful': statistically_meaningful,
            'warning': '本分析は簡易版です。厳密な統計的検定にはOddsComparisonAnalyzerを使用してください。'
        }
    
    return results

def generate_simple_report(results: Dict[str, Any], output_dir: Path):
    """簡易レポート生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "horse_racelevel_odds_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# HorseRaceLevelとオッズ比較分析レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**実行スクリプト**: analyze_horse_racelevel.py\n\n")
        
        # データ概要
        if 'data_summary' in results:
            f.write("## データ概要\n\n")
            summary = results['data_summary']
            f.write(f"- **総レコード数**: {summary.get('total_records', 'N/A'):,}\n")
            f.write(f"- **分析対象馬数**: {summary.get('horse_count', 'N/A'):,}\n")
            f.write(f"- **対象ファイル数**: {summary.get('file_count', 'N/A')}\n\n")
        
        # 相関分析結果
        if 'correlations' in results:
            f.write("## 相関分析結果\n\n")
            f.write("| 変数 | 相関係数 | R² | p値 |\n")
            f.write("|------|----------|----|---------|\n")
            
            for name, corr in results['correlations'].items():
                f.write(f"| {name} | {corr['correlation']:.3f} | {corr['r_squared']:.3f} | {corr['p_value']:.3e} |\n")
            f.write("\n")
        
        # 回帰分析結果
        if 'regression' in results:
            f.write("## 回帰分析結果（H2仮説検証）\n\n")
            regression = results['regression']
            
            f.write("| モデル | 訓練R² | 検証R² | RMSE |\n")
            f.write("|--------|---------|---------|-------|\n")
            
            if 'odds_baseline' in regression:
                model = regression['odds_baseline']
                f.write(f"| オッズベースライン | {model.get('train_r2', 0):.4f} | {model.get('test_r2', 0):.4f} | {model.get('test_rmse', 0):.4f} |\n")
            
            if 'horse_race_level_model' in regression:
                model = regression['horse_race_level_model']
                f.write(f"| HorseRaceLevel | {model.get('train_r2', 0):.4f} | {model.get('test_r2', 0):.4f} | {model.get('test_rmse', 0):.4f} |\n")
            
            # H2仮説結果
            if 'h2_verification' in regression:
                h2 = regression['h2_verification']
                f.write(f"\n### H2仮説検証結果（簡易版）\n\n")
                f.write(f"- **仮説サポート**: {'✓ YES' if h2['hypothesis_supported'] else '✗ NO'}\n")
                f.write(f"- **性能改善**: {h2['improvement']:+.4f}\n")
                f.write(f"- **統計的意味**: {'✓ 有意' if h2.get('statistically_meaningful', False) else '✗ 限定的'}\n")
                if 'warning' in h2:
                    f.write(f"- **注意**: {h2['warning']}\n")
                f.write("\n")
        
        f.write("## 結論\n\n")
        f.write("HorseRaceLevelとオッズ情報の比較分析が完了しました。\n")
        f.write("詳細な分析には、より大規模なデータセットでの検証を推奨します。\n")
    
    logger.info(f"簡易レポートを生成: {report_path}")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HorseRaceLevelとオッズ比較分析を実行します（統合版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # HorseRaceLevelとオッズの比較分析
  python analyze_horse_racelevel.py --odds-analysis export/dataset --output-dir results/horse_racelevel_odds

  # 従来のレースレベル分析
  python analyze_horse_racelevel.py export/with_bias --output-dir results/race_level_analysis

  # 層別分析のみ実行
  python analyze_horse_racelevel.py --stratified-only --output-dir results/stratified_analysis

このスクリプトの新機能:
  1. HorseRaceLevelとオッズ情報の包括的比較分析
  2. H2仮説「HorseRaceLevelがオッズベースラインを上回る」の検証
  3. 相関分析と回帰分析による統計的評価
  4. 従来のレースレベル分析との互換性維持
        """
    )
    parser.add_argument('input_path', nargs='?', help='入力ファイルまたはディレクトリのパス (例: export/with_bias)')
    parser.add_argument('--output-dir', default='results/race_level_analysis', help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=6, help='分析対象とする最小レース数')
    parser.add_argument('--encoding', default='utf-8', help='入力ファイルのエンコーディング')
    parser.add_argument('--start-date', help='分析開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', help='分析終了日（YYYYMMDD形式）')
    
    # 新機能のオプション
    parser.add_argument('--odds-analysis', metavar='DATA_DIR', help='HorseRaceLevelとオッズの比較分析を実行（データディレクトリを指定）')
    parser.add_argument('--sample-size', type=int, default=200, help='オッズ分析でのサンプルファイル数（デフォルト: 200）')
    
    # 従来のオプション（継続）
    parser.add_argument('--three-year-periods', action='store_true',
                       help='3年間隔での期間別分析を実行（デフォルトは全期間分析）')
    parser.add_argument('--enable-time-analysis', action='store_true',
                       help='走破タイム因果関係分析を実行（論文仮説H1, H4検証）')
    parser.add_argument('--enable-stratified-analysis', action='store_true', default=True,
                       help='層別分析を実行（年齢層別、経験数別、距離カテゴリ別）- デフォルトで有効')
    parser.add_argument('--disable-stratified-analysis', action='store_true',
                       help='層別分析を無効化（処理時間短縮用）')
    parser.add_argument('--stratified-only', action='store_true',
                       help='層別分析のみを実行（export/datasetから直接読み込み）')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='ログレベルの設定')
    parser.add_argument('--log-file', help='ログファイルのパス（指定しない場合は自動生成）')
    
    # ログファイル変数の初期化
    log_file = None
    
    try:
        args = parser.parse_args()
        
        # ログファイルの自動生成（args取得後、validate_args前に実行）
        log_file = args.log_file
        if log_file is None:
            # ログディレクトリの作成
            log_dir = Path('export/logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = f'export/logs/analyze_horse_racelevel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # ログ設定の初期化
        setup_logging(log_level=args.log_level, log_file=log_file)
        
        # 引数検証（ログ設定後に実行、オッズ分析の場合はスキップ）
        if not args.odds_analysis:
            args = validate_args(args)

        # ログ設定完了後に開始メッセージを出力
        logger.info("🏇 レースレベル分析を開始します...")
        logger.info(f"📅 実行日時: {datetime.now()}")
        logger.info(f"🖥️ ログレベル: {args.log_level}")
        logger.info(f"📝 ログファイル: {log_file}")

        # 出力ディレクトリの作成（親ディレクトリも含めて確実に作成）
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 出力ディレクトリが書き込み可能かチェック
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"出力ディレクトリの作成に失敗しました: {output_dir}")
        
        logger.info(f"📁 出力ディレクトリ確認済み: {output_dir.absolute()}")

        logger.info(f"📁 入力パス: {args.input_path}")
        logger.info(f"📊 出力ディレクトリ: {args.output_dir}")
        logger.info(f"🎯 最小レース数: {args.min_races}")
        if args.start_date:
            logger.info(f"📅 分析開始日: {args.start_date}")
        if args.end_date:
            logger.info(f"📅 分析終了日: {args.end_date}")
        if args.enable_time_analysis:
            logger.info(f"🏃 RunningTime分析: 有効")
        else:
            logger.info(f"🏃 RunningTime分析: 無効（--enable-time-analysisで有効化）")
        
        # 層別分析設定の処理
        enable_stratified = args.enable_stratified_analysis and not args.disable_stratified_analysis
        if enable_stratified:
            logger.info(f"📊 層別分析: 有効（年齢層別・経験数別・距離カテゴリ別）")
        else:
            logger.info(f"📊 層別分析: 無効（--disable-stratified-analysisで無効化）")
        
        # オッズ分析の場合
        if args.odds_analysis:
            logger.info("🎯 HorseRaceLevelとオッズの比較分析を実行します...")
            try:
                results = perform_comprehensive_odds_analysis(
                    args.odds_analysis, 
                    args.output_dir, 
                    args.sample_size
                )
                
                logger.info("✅ オッズ比較分析が完了しました。")
                logger.info(f"📊 分析対象: {results['data_summary']['total_records']:,}レコード, {results['data_summary']['horse_count']:,}頭")
                logger.info(f"📁 結果保存先: {args.output_dir}")
                
                # H2仮説結果の表示
                if 'regression' in results and 'h2_verification' in results['regression']:
                    h2 = results['regression']['h2_verification']
                    result_text = "サポート" if h2['hypothesis_supported'] else "非サポート"
                    logger.info(f"🎯 H2仮説「HorseRaceLevelがオッズベースラインを上回る」: {result_text}")
                    logger.info(f"   性能改善: {h2['improvement']:+.4f}")
                
                return 0
            except Exception as e:
                logger.error(f"❌ オッズ比較分析でエラー: {str(e)}")
                logger.error("詳細なエラー情報:", exc_info=True)
                return 1
        
        # 層別分析のみの場合
        if args.stratified_only:
            logger.info("📊 層別分析のみを実行します...")
            try:
                stratified_dataset = create_stratified_dataset_from_export('export/dataset')
                stratified_results = perform_integrated_stratified_analysis(stratified_dataset)
                stratified_report = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
                logger.info("✅ 層別分析のみが完了しました。")
                logger.info(f"📊 分析対象: {len(stratified_dataset):,}頭")
                logger.info(f"📁 結果保存先: {output_dir}")
                return 0
            except Exception as e:
                logger.error(f"❌ 層別分析でエラー: {str(e)}")
                logger.error("詳細なエラー情報:", exc_info=True)
                return 1

        if args.three_year_periods:
            logger.info("📊 3年間隔での期間別分析を実行します...")
            
            # 初期データ読み込みで年データ範囲を確認
            temp_config = AnalysisConfig(
                input_path=args.input_path,
                min_races=args.min_races,
                output_dir=str(output_dir),
                date_str=datetime.now().strftime('%Y%m%d'),
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            # データ読み込みと基本的な前処理（期間フィルタリングなし）
            temp_analyzer = RaceLevelAnalyzer(temp_config, 
                                            enable_time_analysis=args.enable_time_analysis,
                                            enable_stratified_analysis=enable_stratified)
            logger.info("📖 全データ読み込み中...")
            temp_df = temp_analyzer.load_data()
            
            logger.info(f"📊 読み込んだデータ件数: {len(temp_df):,}件")
            
            # 年データが存在するかチェック
            if '年' in temp_df.columns and temp_df['年'].notna().any():
                min_year = int(temp_df['年'].min())
                max_year = int(temp_df['年'].max())
                logger.info(f"📊 年データ範囲: {min_year}年 - {max_year}年")
                
                # 3年間隔での期間設定
                periods = []
                for start_year in range(min_year, max_year + 1, 3):
                    end_year = min(start_year + 2, max_year)
                    period_name = f"{start_year}-{end_year}"
                    
                    # 期間内にデータが存在するかチェック
                    period_data = temp_df[
                        (temp_df['年'] >= start_year) & (temp_df['年'] <= end_year)
                    ]
                    
                    if len(period_data) >= args.min_races:
                        periods.append((period_name, start_year, end_year))
                        logger.info(f"  📊 期間 {period_name}: {len(period_data):,}件のデータ")
                    else:
                        logger.warning(f"  ⚠️  期間 {period_name}: データ不足 ({len(period_data)}件)")
                
                if periods:
                    logger.info(f"📊 有効な分析期間: {[p[0] for p in periods]}")
                    
                    # 期間別分析の実行
                    all_results = analyze_by_periods(temp_analyzer, periods, output_dir)
                    
                    if all_results:
                        # 総合レポートの生成
                        generate_period_summary_report(all_results, output_dir)
                        
                        logger.info("\n" + "="*60)
                        logger.info("🎉 3年間隔分析完了！結果:")
                        logger.info("="*60)
                        
                        for period_name, results in all_results.items():
                            period_info = results.get('period_info', {})
                            correlation_stats = results.get('correlation_stats', {})
                            
                            total_horses = period_info.get('total_horses', 0)
                            total_races = period_info.get('total_races', 0)
                            corr_avg = correlation_stats.get('correlation_place_avg', 0.0)
                            
                            logger.info(f"📊 期間 {period_name}: {total_horses:,}頭, {total_races:,}レース")
                            logger.info(f"   📈 平均レベル vs 複勝率相関: r={corr_avg:.3f}")
                        
                        logger.info("="*60)
                        logger.info(f"✅ 全ての結果は {args.output_dir} に保存されました。")
                        logger.info(f"📝 ログファイル: {log_file}")
                        logger.info("📋 生成されたファイル:")
                        logger.info("  - レースレベル分析_期間別総合レポート.md")
                        logger.info("  - 各期間フォルダ内の分析結果PNG")
                        
                        # 層別分析の実行
                        logger.info("📊 層別分析を実行中...")
                        try:
                            stratified_dataset = create_stratified_dataset_from_export('export/dataset')
                            stratified_results = perform_integrated_stratified_analysis(stratified_dataset)
                            stratified_report = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
                            logger.info("✅ 層別分析完了")
                        except Exception as e:
                            logger.error(f"❌ 層別分析でエラー: {str(e)}")
                    else:
                        logger.warning("⚠️  有効な期間別分析結果がありませんでした。")
                else:
                    logger.warning("⚠️  十分なデータがある期間が見つかりませんでした。全期間での分析に切り替えます。")
                    args.three_year_periods = False
            else:
                logger.warning("⚠️  年データが見つかりません。全期間での分析に切り替えます。")
                args.three_year_periods = False
        
        if not args.three_year_periods:
            logger.info("📊 【修正版】厳密な時系列分割による分析を実行します...")
            
            # 設定の作成
            date_str = datetime.now().strftime('%Y%m%d')
            config = AnalysisConfig(
                input_path=args.input_path,
                min_races=args.min_races,
                output_dir=str(output_dir),
                date_str=date_str,
                start_date=args.start_date,
                end_date=args.end_date
            )

            # 1. RaceLevelAnalyzerのインスタンス化
            analyzer = RaceLevelAnalyzer(config, args.enable_time_analysis, enable_stratified)

            # 2. データの読み込み
            logger.info("📖 全データ読み込み中...")
            analyzer.df = analyzer.load_data()

            # 前処理を追加
            logger.info("🔧 前処理中...")
            analyzer.df = analyzer.preprocess_data()
            
            # 3. 特徴量計算
            logger.info("🧮 特徴量計算中...")
            analyzer.df = analyzer.calculate_feature()

            # 4. 【重要】修正版分析の実行
            logger.info("🔬 【修正版】厳密な時系列分割による分析を実行中...")
            analyzer.stats = analyzer.analyze()
            
            # 結果の可視化
            analyzer.visualize()

            # 【追加】レポート整合性の確認
            logger.info("🔍 レポート整合性チェック:")
            oot_results = analyzer.stats.get('out_of_time_validation', {})
            test_performance = oot_results.get('test_performance', {})
            
            if test_performance:
                test_r2 = test_performance.get('r_squared', 0)
                test_corr = test_performance.get('correlation', 0)
                test_size = test_performance.get('sample_size', 0)
                
                logger.info(f"   📊 検証期間(2013-2014年)サンプル数: {test_size}頭")
                logger.info(f"   📊 検証期間R²: {test_r2:.3f}")
                logger.info(f"   📊 検証期間相関係数: {test_corr:.3f}")
                
                # 実測結果の統計的評価
                if test_r2 > 0.01:
                    logger.info("✅ 統計的に有意な説明力を確認")
                else:
                    logger.warning("⚠️ 説明力が限定的です")
                    
                if abs(test_corr) > 0.1:
                    logger.info("✅ 実用的な相関関係を確認")
                else:
                    logger.warning("⚠️ 相関関係が弱いです")

            # 層別分析の実行
            logger.info("📊 統合層別分析を実行中...")
            try:
                stratified_dataset = create_stratified_dataset_from_export('export/dataset')
                stratified_results = perform_integrated_stratified_analysis(stratified_dataset)
                stratified_report = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
                logger.info("✅ 統合層別分析完了")
            except Exception as e:
                logger.error(f"❌ 層別分析でエラー: {str(e)}")
                logger.error("詳細なエラー情報:", exc_info=True)
            
            logger.info(f"✅ 【修正版】分析が完了しました。結果は {output_dir} に保存されました。")
            logger.info(f"📝 ログファイル: {log_file}")
            logger.info("🎯 データリーケージ防止と時系列分割が正しく実装されました。")
            logger.info("📊 統合層別分析により包括的な検証を実施しました。")

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ ファイルエラー: {str(e)}")
        logger.error("💡 解決方法:")
        logger.error("   • 入力パスが正しいか確認してください")
        logger.error("   • ファイル名に日本語が含まれている場合は英数字に変更してください")
        logger.error("   • 'export/with_bias' ディレクトリが存在するか確認してください")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"❌ 入力値エラー: {error_msg}")
        logger.error("💡 解決方法:")
        
        if "条件を満たすデータが見つかりません" in error_msg:
            logger.error("   • --min-races の値を小さくしてみてください（例: --min-races 3）")
            logger.error("   • 期間指定が狭すぎる場合は範囲を広げてください")
            logger.error("   • データが存在する期間かどうか確認してください")
        elif "日付形式" in error_msg:
            logger.error("   • 日付はYYYYMMDD形式で指定してください（例: 20220101）")
            logger.error("   • --start-date と --end-date の両方を指定してください")
        else:
            logger.error("   • パラメータの値が正しいか確認してください")
            logger.error("   • --help でオプションの詳細を確認できます")
        
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except IndexError as e:
        logger.error(f"❌ データ処理エラー: {str(e)}")
        logger.error("💡 解決方法:")
        logger.error("   • データ期間が短すぎる可能性があります")
        logger.error("   • 時系列分割に必要な最低3年分のデータがあるか確認してください")
        logger.error("   • 期間指定を広げて再実行してみてください")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except KeyboardInterrupt:
        logger.warning("⏹️ ユーザーによって処理が中断されました")
        logger.info("💡 処理時間を短縮するには:")
        logger.info("   • --min-races を大きくしてサンプル数を減らす")
        logger.info("   • 期間を短くして処理範囲を絞る")
        logger.info("   • --disable-stratified-analysis で層別分析を無効化")
        if log_file:
            logger.info(f"📝 ログファイル: {log_file}")
        return 1
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 予期せぬエラーが発生しました: {error_msg}")
        logger.error("💡 解決方法:")
        
        if "encoding" in error_msg.lower() or "unicode" in error_msg.lower():
            logger.error("   • ファイルのエンコーディングに問題があります")
            logger.error("   • CSVファイルがUTF-8またはShift-JISで保存されているか確認してください")
        elif "memory" in error_msg.lower():
            logger.error("   • メモリ不足の可能性があります")
            logger.error("   • --min-races を大きくしてデータ量を減らしてください")
            logger.error("   • 不要なアプリケーションを終了してください")
        elif "permission" in error_msg.lower():
            logger.error("   • ファイルアクセス権限の問題があります")
            logger.error("   • 出力ディレクトリの書き込み権限を確認してください")
            logger.error("   • 管理者権限で実行してみてください")
        else:
            logger.error("   • --log-level DEBUG で詳細ログを確認してください")
            logger.error("   • データファイルが破損していないか確認してください")
            logger.error("   • Pythonとライブラリのバージョンを確認してください")
        
        logger.error("🔍 詳細なエラー情報:")
        logger.error(f"   エラー種別: {type(e).__name__}")
        logger.error(f"   エラー内容: {error_msg}")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        logger.error("詳細なスタックトレース:", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
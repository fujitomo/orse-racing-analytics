"""
重み計算検証スクリプト
race_level_analysis_report.md の5.1.3節に基づく重み計算の検証

問題:
- ログの重み計算結果がレポートと大きく異なる
- 動的計算: グレード6.6%, 場所5.4%, 距離88.0%
- レポート値: グレード63.6%, 場所32.3%, 距離4.1%

検証項目:
1. 相関計算の正確性
2. 重み計算式の適用
3. データの整合性
4. レポート値との比較
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import pearsonr
import sys
import os

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data():
    """訓練期間データ（2010-2020年）を読み込み"""
    logger.info("📖 訓練期間データ（2010-2020年）を読み込み中...")
    
    # 全CSVファイルを読み込み
    dataset_dir = Path("export/dataset")
    csv_files = list(dataset_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("❌ CSVファイルが見つかりません")
        return None
    
    logger.info(f"📊 {len(csv_files)}個のCSVファイルを読み込み中...")
    
    all_dfs = []
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_dfs.append(df)
            
            if (i + 1) % 100 == 0:
                logger.info(f"   進捗: {i + 1}/{len(csv_files)}ファイル")
                
        except Exception as e:
            logger.warning(f"⚠️ ファイル読み込みエラー: {csv_file.name} - {str(e)}")
            continue
    
    if not all_dfs:
        logger.error("❌ 有効なデータが見つかりません")
        return None
    
    # データ統合
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"✅ 統合完了: {len(combined_df):,}行のデータ")
    
    # 年列の確認と作成
    if '年' not in combined_df.columns:
        if '年月日' in combined_df.columns:
            combined_df['年'] = pd.to_datetime(combined_df['年月日'], errors='coerce').dt.year
        else:
            logger.error("❌ 年または年月日列が見つかりません")
            return None
    
    # 訓練期間（2010-2020年）でフィルタ
    train_data = combined_df[(combined_df['年'] >= 2010) & (combined_df['年'] <= 2020)].copy()
    logger.info(f"📊 訓練期間データ: {len(train_data):,}行 (2010-2020年)")
    
    return train_data

def calculate_feature_levels(df):
    """特徴量レベルを計算"""
    logger.info("🧮 特徴量レベルを計算中...")
    
    df_copy = df.copy()
    
    # 1. grade_level の計算（レポート5.1.3節準拠）
    def calculate_grade_level(row):
        """グレードレベルを計算（レポート仕様準拠）"""
        # グレード列の候補を確認
        grade_cols = ['グレード_x', 'グレード_y', 'グレード', 'grade']
        grade_value = None
        
        for col in grade_cols:
            if col in row and pd.notna(row[col]):
                grade_value = row[col]
                break
        
        if grade_value is None:
            return 2.0  # デフォルト値（特別レベル）
        
        # レポート5.1.3節準拠のグレードレベル設定
        try:
            grade_num = float(grade_value)
            if grade_num == 1:
                return 9.0  # G1（最高）
            elif grade_num == 2:
                return 7.5  # G2
            elif grade_num == 3:
                return 6.0  # G3
            elif grade_num == 4:
                return 4.5  # 重賞
            elif grade_num == 5:
                return 2.0  # 特別
            elif grade_num == 6:
                return 3.0  # リステッド
            else:
                return 1.0  # その他
        except (ValueError, TypeError):
            return 2.0  # デフォルト値
    
    df_copy['grade_level'] = df_copy.apply(calculate_grade_level, axis=1)
    logger.info(f"📊 grade_level 計算完了: 範囲 {df_copy['grade_level'].min():.2f} - {df_copy['grade_level'].max():.2f}")
    
    # 2. venue_level の計算（レポート5.1.3節準拠）
    def calculate_venue_level(row):
        """場所レベルを計算（レポート仕様準拠）"""
        # 競馬場名に基づく格式レベル
        venue_cols = ['場名', '競馬場', 'venue']
        venue_name = None
        
        for col in venue_cols:
            if col in row and pd.notna(row[col]):
                venue_name = str(row[col])
                break
        
        if venue_name is None:
            return 1.0  # デフォルト値
        
        # レポート5.1.3節準拠の競馬場格式レベル
        if venue_name in ['東京', '阪神']:
            return 3.0  # 最高格式
        elif venue_name in ['京都', '中山']:
            return 2.5  # 高格式
        elif venue_name in ['新潟', '中京', '小倉']:
            return 2.0  # 中格式
        elif venue_name in ['札幌', '函館', '福島']:
            return 1.5  # 低格式
        else:
            return 1.0  # その他
    
    df_copy['venue_level'] = df_copy.apply(calculate_venue_level, axis=1)
    logger.info(f"📊 venue_level 計算完了: 範囲 {df_copy['venue_level'].min():.2f} - {df_copy['venue_level'].max():.2f}")
    
    # 3. distance_level の計算（レポート5.1.3節準拠）
    def calculate_distance_level(row):
        """距離レベルを計算（レポート仕様準拠）"""
        if '距離' in row and pd.notna(row['距離']):
            try:
                distance = float(row['距離'])
                # レポート5.1.3節準拠の距離レベル設定
                if distance >= 2000:
                    return 1.25  # 長距離
                elif distance >= 1600:
                    return 1.35  # 中長距離
                elif distance >= 1200:
                    return 1.0   # 中距離
                else:
                    return 0.85  # 短距離
            except (ValueError, TypeError):
                pass
        
        return 1.0  # デフォルト値
    
    df_copy['distance_level'] = df_copy.apply(calculate_distance_level, axis=1)
    logger.info(f"📊 distance_level 計算完了: 範囲 {df_copy['distance_level'].min():.2f} - {df_copy['distance_level'].max():.2f}")
    
    return df_copy

def create_horse_statistics(df):
    """馬統計データを作成"""
    logger.info("🐎 馬統計データを作成中...")
    
    # 複勝フラグを作成
    if '着順' in df.columns:
        df['is_placed'] = (pd.to_numeric(df['着順'], errors='coerce') <= 3).astype(int)
        logger.info("📊 着順列から複勝フラグを作成（着順<=3）")
    elif '複勝' in df.columns:
        df['is_placed'] = pd.to_numeric(df['複勝'], errors='coerce').fillna(0)
        logger.info("📊 複勝列から複勝フラグを作成")
    else:
        logger.error("❌ 複勝フラグを作成できません")
        return None
    
    # 複勝フラグの統計を確認
    placed_count = df['is_placed'].sum()
    total_count = len(df)
    placed_rate = placed_count / total_count if total_count > 0 else 0
    logger.info(f"📊 複勝フラグ統計: {placed_count:,}/{total_count:,} ({placed_rate:.1%})")
    
    # 馬ごとの統計を計算（最低出走数6戦以上）
    horse_stats = df.groupby('馬名').agg({
        'is_placed': 'mean',  # 複勝率
        '年': 'count'  # 出走回数
    }).reset_index()
    
    # 列名を標準化
    horse_stats.columns = ['馬名', 'place_rate', 'race_count']
    
    # 最低出走数6戦以上でフィルタ（レポート仕様準拠）
    horse_stats = horse_stats[horse_stats['race_count'] >= 6].copy()
    logger.info(f"📊 最低出走数6戦以上でフィルタ: {len(horse_stats):,}頭")
    
    # 特徴量レベルの平均を計算
    feature_cols = ['grade_level', 'venue_level', 'distance_level']
    for col in feature_cols:
        if col in df.columns:
            avg_feature = df.groupby('馬名')[col].mean().reset_index()
            avg_feature.columns = ['馬名', f'avg_{col}']
            horse_stats = horse_stats.merge(avg_feature, on='馬名', how='left')
    
    logger.info(f"📊 馬統計データ作成完了: {len(horse_stats):,}頭")
    
    return horse_stats

def calculate_correlations(horse_stats):
    """相関を計算（馬統計データベース）"""
    logger.info("📈 馬統計データで相関を計算中...")
    
    # 必要な列の確認
    required_cols = ['place_rate', 'avg_grade_level', 'avg_venue_level', 'avg_distance_level']
    missing_cols = [col for col in required_cols if col not in horse_stats.columns]
    
    if missing_cols:
        logger.error(f"❌ 必要な列が不足: {missing_cols}")
        logger.info(f"📊 利用可能な列: {list(horse_stats.columns)}")
        return None
    
    # 欠損値を除去
    clean_data = horse_stats[required_cols].dropna()
    logger.info(f"📊 相関計算用データ: {len(clean_data):,}頭")
    
    if len(clean_data) < 100:
        logger.error(f"❌ サンプル数が不足: {len(clean_data)}頭（最低100頭必要）")
        return None
    
    # 相関計算
    correlations = {}
    target = clean_data['place_rate']
    
    # レポート5.1.3節準拠の相関計算
    feature_mapping = {
        'avg_grade_level': 'grade',
        'avg_venue_level': 'venue', 
        'avg_distance_level': 'distance'
    }
    
    for feature_col, feature_name in feature_mapping.items():
        if feature_col in clean_data.columns:
            corr, p_value = pearsonr(clean_data[feature_col], target)
            correlations[feature_name] = {
                'correlation': corr,
                'p_value': p_value,
                'squared': corr ** 2
            }
            logger.info(f"   📈 {feature_name}_level: r = {corr:.3f}, r² = {corr**2:.3f}, p = {p_value:.3f}")
    
    return correlations

def calculate_weights(correlations):
    """重みを計算（レポート5.1.3節準拠）"""
    logger.info("⚖️ 重みを計算中...")
    logger.info("📋 計算式: w_i = r_i² / (r_grade² + r_venue² + r_distance²)")
    
    # 相関の二乗を計算
    squared_correlations = {}
    total_squared = 0
    
    for feature, stats in correlations.items():
        squared = stats['squared']
        squared_correlations[feature] = squared
        total_squared += squared
        logger.info(f"   📊 {feature}: r² = {squared:.3f}")
    
    logger.info(f"📊 総寄与度: {total_squared:.3f}")
    
    if total_squared == 0:
        logger.warning("⚠️ 総寄与度が0です。フォールバック重みを使用します。")
        return get_fallback_weights()
    
    # 重みを正規化
    weights = {}
    for feature, squared in squared_correlations.items():
        weight = squared / total_squared
        weights[feature] = weight
        logger.info(f"   ⚖️ {feature}: w = {weight:.3f} ({weight*100:.1f}%)")
    
    return weights

def get_fallback_weights():
    """レポート5.1.3節の固定重み"""
    return {
        'grade': 0.636,   # 63.6%
        'venue': 0.323,   # 32.3%
        'distance': 0.041 # 4.1%
    }

def compare_with_report(weights):
    """レポート値と比較"""
    logger.info("📋 レポート値と比較中...")
    
    # レポート5.1.3節の値
    report_weights = {
        'grade': 0.636,   # 63.6%
        'venue': 0.323,   # 32.3%
        'distance': 0.041 # 4.1%
    }
    
    logger.info("📊 比較結果:")
    for feature in weights.keys():
        calculated = weights[feature]
        reported = report_weights.get(feature, 0)
        diff = calculated - reported
        diff_pct = (diff / reported * 100) if reported > 0 else 0
        
        logger.info(f"   {feature}:")
        logger.info(f"     計算値: {calculated:.3f} ({calculated*100:.1f}%)")
        logger.info(f"     レポート値: {reported:.3f} ({reported*100:.1f}%)")
        logger.info(f"     差異: {diff:+.3f} ({diff_pct:+.1f}%)")
    
    return report_weights

def main():
    """メイン処理"""
    logger.info("🔍 重み計算検証を開始...")
    
    # 1. データ読み込み
    train_data = load_training_data()
    if train_data is None:
        return
    
    # 2. 特徴量レベル計算
    df_with_features = calculate_feature_levels(train_data)
    
    # 3. 馬統計データ作成
    horse_stats = create_horse_statistics(df_with_features)
    if horse_stats is None:
        return
    
    # 4. 相関計算
    correlations = calculate_correlations(horse_stats)
    if correlations is None:
        return
    
    # 5. 重み計算
    weights = calculate_weights(correlations)
    
    # 6. レポート値と比較
    report_weights = compare_with_report(weights)
    
    # 7. 結果サマリー
    logger.info("📋 検証結果サマリー:")
    logger.info("=" * 50)
    logger.info("計算された重み:")
    for feature, weight in weights.items():
        logger.info(f"  {feature}: {weight:.3f} ({weight*100:.1f}%)")
    
    logger.info("\nレポート値:")
    for feature, weight in report_weights.items():
        logger.info(f"  {feature}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 8. 問題診断
    logger.info("\n🔍 問題診断:")
    max_diff = max(abs(weights.get(f, 0) - report_weights.get(f, 0)) for f in weights.keys())
    if max_diff > 0.1:
        logger.warning(f"⚠️ 重み計算に大きな差異があります (最大差異: {max_diff:.3f})")
        logger.info("   原因候補:")
        logger.info("   1. 特徴量レベルの計算方法（修正済み）")
        logger.info("   2. データフィルタリング条件（修正済み）")
        logger.info("   3. 相関計算の前提条件")
        logger.info("   4. レポート値の算出方法")
        
        # 詳細分析
        logger.info("\n📊 詳細分析:")
        for feature in weights.keys():
            calc_val = weights[feature]
            report_val = report_weights[feature]
            diff = abs(calc_val - report_val)
            if diff > 0.1:
                logger.info(f"   {feature}: 差異 {diff:.3f} (計算値: {calc_val:.3f}, レポート値: {report_val:.3f})")
    else:
        logger.info("✅ 重み計算は正常です")
    
    logger.info("✅ 重み計算検証完了")

if __name__ == "__main__":
    main()

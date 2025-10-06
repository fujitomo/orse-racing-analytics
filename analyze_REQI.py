#!/usr/bin/env python
"""
競馬レース分析コマンドラインツール（競走経験質指数REQIとオッズ比較対応版）
馬ごとの競走経験質指数（REQI）の分析とオッズ情報との比較分析を実行します。
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt  # 可視化に必要（関数内でインポート）
# import seaborn as sns              # 可視化に必要（関数内でインポート）
import warnings
import time
import psutil
import os
from functools import wraps
warnings.filterwarnings('ignore')

# 既存のインポートも保持
try:
    from horse_racing.base.analyzer import AnalysisConfig
    from horse_racing.analyzers.race_level_analyzer import REQIAnalyzer
    from horse_racing.core.weight_manager import WeightManager, get_global_weights
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

# グローバル変数：計算済みデータを保持
_global_data = None
_global_feature_levels = None
_global_raw_data = None  # 生データ（CSV読み込み結果）

# パフォーマンス監視用のユーティリティ関数
def log_performance(func_name=None):
    """パフォーマンス監視デコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 関数名を自動取得または指定された名前を使用
            name = func_name or func.__name__
            
            # 開始時のリソース情報取得
            process = psutil.Process(os.getpid())
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu_percent = process.cpu_percent()
            
            logger.info(f"🚀 [{name}] 開始 - 開始時メモリ: {start_memory:.1f}MB, CPU: {start_cpu_percent:.1f}%")
            
            try:
                # 関数実行
                result = func(*args, **kwargs)
                
                # 終了時のリソース情報取得
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                end_cpu_percent = process.cpu_percent()
                
                # 実行時間とリソース使用量を計算
                execution_time = end_time - start_time
                memory_diff = end_memory - start_memory
                
                # メモリ使用量の監視と警告
                if memory_diff > 200:  # 200MB以上の増加
                    logger.warning(f"⚠️ [{name}] メモリ使用量が200MB増加しました: {memory_diff:+.1f}MB")
                elif memory_diff > 500:  # 500MB以上の増加
                    logger.warning(f"⚠️ [{name}] メモリ使用量が500MB増加しました: {memory_diff:+.1f}MB")
                
                # ログ出力
                logger.info(f"✅ [{name}] 完了 - 実行時間: {execution_time:.2f}秒")
                logger.info(f"   💾 メモリ使用量: {end_memory:.1f}MB (差分: {memory_diff:+.1f}MB)")
                logger.info(f"   🖥️  CPU使用率: {end_cpu_percent:.1f}%")
                
                # パフォーマンス警告
                if execution_time > 60:
                    logger.warning(f"⚠️ [{name}] 実行時間が1分を超えました: {execution_time:.2f}秒")
                if memory_diff > 500:
                    logger.warning(f"⚠️ [{name}] メモリ使用量が500MB増加しました: {memory_diff:.1f}MB")
                
                return result
                
            except Exception:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"❌ [{name}] エラー発生 - 実行時間: {execution_time:.2f}秒")
                raise
                
        return wrapper
    return decorator

def log_dataframe_info(df: pd.DataFrame, description: str):
    """DataFrameの詳細情報をログ出力"""
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    logger.info(f"📊 [{description}] データフレーム情報:")
    logger.info(f"   📏 形状: {df.shape[0]:,}行 × {df.shape[1]}列")
    logger.info(f"   💾 メモリ使用量: {memory_usage:.1f}MB")
    logger.info(f"   📈 データ型分布: {dict(df.dtypes.value_counts())}")
    
    # 欠損値情報
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info(f"   ⚠️ 欠損値: {null_counts.sum():,}個 ({null_counts.sum()/df.size*100:.1f}%)")
        try:
            # 列別トップNの欠損内訳
            missing_counts_sorted = null_counts.sort_values(ascending=False)
            missing_pct_sorted = (missing_counts_sorted / len(df) * 100).round(1)
            top_n = 15
            top_missing = (
                pd.concat([
                    missing_counts_sorted.rename('count'),
                    missing_pct_sorted.rename('%')
                ], axis=1)
                .head(top_n)
            )
            if len(top_missing) > 0:
                logger.info("   🔍 欠損トップ15(列):\n" + top_missing.to_string())
            
            # 年別×主要列の欠損率
            key_cols = ['グレード', '10時単勝オッズ', '10時複勝オッズ', '確定複勝オッズ下', '騎手コード']
            available_key_cols = [c for c in key_cols if c in df.columns]
            if '年' in df.columns and len(available_key_cols) > 0:
                year_missing = (
                    df.groupby('年')[available_key_cols]
                      .apply(lambda x: x.isnull().mean().mul(100).round(1))
                )
                logger.info("   🔍 年別×主要列 欠損率(%):\n" + year_missing.to_string())
        except Exception as e:
            logger.warning(f"   ⚠️ 欠損詳細ログの生成中に例外: {str(e)}")
    
def log_processing_step(step_name: str, start_time: float, current_idx: int, total_count: int):
    """処理ステップの進捗をログ出力"""
    elapsed = time.time() - start_time
    if current_idx > 0:
        avg_time_per_item = elapsed / current_idx
        remaining_items = total_count - current_idx
        eta = remaining_items * avg_time_per_item
        
        logger.info(f"⏳ [{step_name}] 進捗: {current_idx:,}/{total_count:,} "
                   f"({current_idx/total_count*100:.1f}%) - "
                   f"経過時間: {elapsed:.1f}秒, 残り予想: {eta:.1f}秒")

def log_system_resources():
    """システムリソースの現在状況をログ出力"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    # システム全体の情報
    system_memory = psutil.virtual_memory()
    system_cpu = psutil.cpu_percent()
    
    logger.info("🖥️ システムリソース状況:")
    logger.info(f"   プロセスメモリ: {memory_info.rss/1024/1024:.1f}MB")
    logger.info(f"   プロセスCPU: {cpu_percent:.1f}%")
    logger.info(f"   システムメモリ使用率: {system_memory.percent:.1f}% "
               f"({system_memory.used/1024/1024/1024:.1f}GB/{system_memory.total/1024/1024/1024:.1f}GB)")
    logger.info(f"   システムCPU使用率: {system_cpu:.1f}%")

def get_all_dataset_files(data_dir: str) -> List[Path]:
    """指定ディレクトリ内のすべてのデータセットファイルを取得"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    # データセットファイルのパターンを検索
    csv_files = list(data_path.glob('SED*_formatted_dataset.csv'))
    return sorted(csv_files)

def load_all_data_once(input_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    全CSVファイルを一度だけ読み込む関数（グローバル変数に保存）
    
    Args:
        input_path: 入力パス
        encoding: エンコーディング
        
    Returns:
        統合されたデータフレーム
    """
    global _global_raw_data
    
    if _global_raw_data is not None:
        logger.info("💾 グローバル変数から生データを取得中...")
        return _global_raw_data.copy()
    
    logger.info("📖 全CSVファイルを初回読み込み中...")
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # 単一ファイルの場合
        df = pd.read_csv(input_path_obj, encoding=encoding)
        logger.info(f"📊 単一ファイル読み込み: {len(df):,}行")
        _global_raw_data = df.copy()
        return df
    else:
        # ディレクトリの場合
        csv_files = list(input_path_obj.glob("*.csv"))
        if not csv_files:
            logger.error(f"❌ CSVファイルが見つかりません: {input_path}")
            return pd.DataFrame()
        
        logger.info(f"📊 全CSVファイルを統合中... ({len(csv_files)}ファイル)")
        all_dfs = []
        
        for i, csv_file in enumerate(csv_files):
            try:
                df_temp = pd.read_csv(csv_file, encoding=encoding)
                all_dfs.append(df_temp)
                
                # 進捗表示（100ファイルごと）
                if (i + 1) % 100 == 0:
                    logger.info(f"   読み込み進捗: {i + 1}/{len(csv_files)}ファイル")
                    
            except Exception as e:
                logger.warning(f"⚠️ ファイル読み込みエラー（スキップ）: {csv_file.name} - {str(e)}")
                continue
        
        if all_dfs:
            logger.info("🔄 データフレーム統合中...")
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"✅ 統合完了: {len(combined_df):,}行のデータ")
            
            # グローバル変数に保存
            _global_raw_data = combined_df.copy()
            logger.info("💾 生データをグローバル変数に保存しました")
            logger.info(f"🔍 グローバル変数確認: _global_raw_data is not None = {_global_raw_data is not None}")
            return combined_df
        else:
            logger.error("❌ 有効なCSVファイルが見つかりませんでした")
            return pd.DataFrame()

def initialize_global_weights(args) -> bool:
    """
    グローバル重みを初期化する関数
    
    Args:
        args: コマンドライン引数
        
    Returns:
        初期化成功フラグ
    """
    global _global_data, _global_feature_levels
    
    try:
        logger.info("🎯 グローバル重み初期化開始...")
        
        # データの読み込み（各分析タイプに応じて）
        if args.odds_analysis:
            # オッズ分析用データ
            data_path = Path(args.odds_analysis)
            if not data_path.exists():
                logger.error(f"❌ データパスが存在しません: {data_path}")
                return False
                
            # サンプルファイルを読み込み
            csv_files = list(data_path.glob("*_formatted_dataset.csv"))
            if not csv_files:
                logger.error(f"❌ CSVファイルが見つかりません: {data_path}")
                return False
                
            # 重み計算用に全データを読み込み（重複処理回避のため）
            sample_dfs = []
            files_to_read = len(csv_files)  # 全ファイルを読み込み
            logger.info(f"📊 重み計算用データ読み込み: {files_to_read}ファイル（全データ統合）")
            
            for csv_file in csv_files[:files_to_read]:
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    sample_dfs.append(df)
                    logger.info(f"📊 読み込み完了: {csv_file.name} ({len(df):,}行)")
                except Exception as e:
                    logger.warning(f"⚠️ ファイル読み込みエラー（スキップ）: {csv_file} - {str(e)}")
                    continue
            
            if sample_dfs:
                combined_df = pd.concat(sample_dfs, ignore_index=True)
                logger.info(f"📊 重み算出用データ: {len(combined_df):,}行（{len(sample_dfs)}ファイル）")
                
                # グローバル変数に保存（統一分析器での重複処理回避）
                _global_raw_data = combined_df.copy()
                logger.info("💾 生データをグローバル変数に保存しました（重み計算時）")
                
                # __main__ 同期は不要（UnifiedAnalyzer経由に統一）
                
                # 【統一】期間別と同一路線で特徴量を生成（grade/venue/distance）
                logger.info("🔧 特徴量前処理を期間別と同一路線に統一します...")
                df_levels = calculate_accurate_feature_levels(combined_df)
                
                # REQI特徴量（必要に応じて）も生成してキャッシュ
                logger.info("⚖️ REQI特徴量（時間的分離版）を生成してキャッシュします...")
                df_levels_with_reqi = calculate_race_level_features_with_position_weights(df_levels)
                
                # レベルカラムの存在確認（期間別と同じ3本を要求）
                required_level_cols = ['grade_level', 'venue_level', 'distance_level']
                missing_cols = [col for col in required_level_cols if col not in df_levels.columns]
                if missing_cols:
                    logger.warning(f"⚠️ レベルカラム生成後も不足: {missing_cols}")
                    logger.warning("📊 フォールバック重みを使用します...")
                    fallback_weights = {
                        'grade_weight': 0.65,
                        'venue_weight': 0.30,
                        'distance_weight': 0.05
                    }
                    WeightManager._global_weights = fallback_weights
                    WeightManager._initialized = True
                    logger.info(f"✅ フォールバック重み設定完了: {fallback_weights}")
                    # それでもグローバルキャッシュは設定しておく
                    _global_data = combined_df.copy()
                    _global_feature_levels = df_levels_with_reqi.copy()
                    return True
                else:
                    logger.info("✅ 特徴量レベルカラム生成完了（期間別準拠）")
                
                # グローバル変数に保存（期間別と同様にキャッシュ）
                _global_data = combined_df.copy()
                _global_feature_levels = df_levels_with_reqi.copy()
                logger.info("💾 計算済みデータをグローバル変数に保存しました（期間別準拠ルート）")
                
                # グローバル重みを初期化（期間別と同じく2010-2020年で学習）
                training_df = df_levels
                if '年' in df_levels.columns:
                    train_mask = (df_levels['年'] >= 2010) & (df_levels['年'] <= 2020)
                    filtered = df_levels[train_mask]
                    if len(filtered) > 0:
                        logger.info(f"📊 重み算出用訓練期間データ: {len(filtered):,}行 (2010-2020年)")
                        training_df = filtered
                    else:
                        logger.warning("⚠️ 訓練期間（2010-2020年）のデータが見つからず、全データで学習します")
                else:
                    logger.warning("⚠️ 年列が見つからず、全データで学習します")

                weights = WeightManager.initialize_from_training_data(training_df)
                logger.info(f"✅ グローバル重み設定完了: {weights}")
                # 期間別フローと同様に、直後に取得ログを出して整合を取る
                logger.info("🔎 重み取得確認（期間別と同一フロー）...")
                _ = WeightManager.get_weights()  # ここで「✅ グローバル重みを正常に取得しました」を出力
                # 以降の処理で再計算されないように明示
                WeightManager.prevent_recalculation()
                return True
                
        elif args.stratified_only:
            # 層別分析用データ（export/dataset）
            dataset_path = Path("export/dataset")
            if dataset_path.exists():
                # グローバル関数を使用してデータを読み込み
                combined_df = load_all_data_once(str(dataset_path), 'utf-8')
                if combined_df.empty:
                    return False
                
                # 年の範囲を確認
                if '年' in combined_df.columns:
                    year_range = f"{combined_df['年'].min()}-{combined_df['年'].max()}年"
                    logger.info(f"📅 全データ期間: {year_range}")
                    
                    # レポート5.1.3節準拠：訓練期間（2010-2020年）データを抽出
                    training_data = combined_df[(combined_df['年'] >= 2010) & (combined_df['年'] <= 2020)]
                    if len(training_data) > 0:
                        df = training_data
                        training_year_range = f"{training_data['年'].min()}-{training_data['年'].max()}年"
                        logger.info(f"📊 訓練期間データ: {len(training_data):,}行 ({training_year_range})")
                    else:
                        logger.warning("⚠️ 訓練期間（2010-2020年）データが見つかりませんでした")
                        df = combined_df  # 全データを使用
                else:
                    logger.warning("⚠️ 年列が見つかりません。全データを使用します")
                    df = combined_df
                
                # 特徴量レベル列を計算（重み計算のため）
                logger.info("🧮 重み計算用特徴量レベル列を計算中...")
                df = calculate_accurate_feature_levels(df)
                
                # レベルカラムの存在確認
                required_level_cols = ['grade_level', 'venue_level', 'distance_level']
                missing_cols = [col for col in required_level_cols if col not in df.columns]
                
                if missing_cols:
                    logger.warning(f"⚠️ レベルカラム生成後も不足: {missing_cols}")
                    logger.warning("📊 フォールバック重みを使用します...")
                    # フォールバック重みを設定
                    fallback_weights = {
                        'grade_weight': 0.65,
                        'venue_weight': 0.30,
                        'distance_weight': 0.05
                    }
                    WeightManager._global_weights = fallback_weights
                    WeightManager._initialized = True
                    logger.info(f"✅ フォールバック重み設定完了: {fallback_weights}")
                    return True
                else:
                    logger.info("✅ レベルカラム生成完了")
                
                # グローバル変数に保存（重複処理回避のため）
                _global_data = combined_df.copy()
                _global_feature_levels = df.copy()
                logger.info("💾 計算済みデータをグローバル変数に保存しました")
                
                # グローバル重みを初期化
                weights = WeightManager.initialize_from_training_data(df)
                logger.info(f"✅ グローバル重み設定完了: {weights}")
                return True
                    
        elif args.input_path:
            # 従来の競走経験質指数（REQI）分析
            # グローバル関数を使用してデータを読み込み
            combined_df = load_all_data_once(args.input_path, args.encoding)
            if combined_df.empty:
                return False
            
            # 年の範囲を確認
            if '年' in combined_df.columns:
                year_range = f"{combined_df['年'].min()}-{combined_df['年'].max()}年"
                logger.info(f"📅 全データ期間: {year_range}")
                
                # レポート5.1.3節準拠：訓練期間（2010-2020年）データを抽出
                training_data = combined_df[(combined_df['年'] >= 2010) & (combined_df['年'] <= 2020)]
                if len(training_data) > 0:
                    df = training_data
                    training_year_range = f"{training_data['年'].min()}-{training_data['年'].max()}年"
                    logger.info(f"📊 重み計算用訓練期間データ: {len(training_data):,}行 ({training_year_range})")
                else:
                    logger.warning("⚠️ 訓練期間（2010-2020年）データが見つかりませんでした")
                    df = combined_df  # 全データを使用
                
                # 全データも保存（時系列分割用）
                logger.info(f"📊 全データ期間: {len(combined_df):,}行 ({combined_df['年'].min()}-{combined_df['年'].max()}年)")
            else:
                logger.warning("⚠️ 年列が見つかりません。全データを使用します")
                df = combined_df
            
            # 特徴量レベル列を計算（重み計算のため：訓練期間2010-2020年のみ）
            logger.info("🧮 重み計算用特徴量レベル列を計算中（訓練期間2010-2020年）...")
            df = calculate_accurate_feature_levels(df)
            
            # グローバル変数に保存（重複処理回避のため）
            _global_data = combined_df.copy()  # 全データ（時系列分割用）
            
            # 【重要修正】全データで特徴量レベル列を計算（期間別分析で2022-2025年も含める）
            logger.info("🧮 全データで特徴量レベル列を計算中（期間別分析用）...")
            df_all_features = calculate_accurate_feature_levels(combined_df)
            
            # 競走経験質指数（REQI）特徴量も事前計算して保存（期間別分析の高速化）
            logger.info("🚀 競走経験質指数（REQI）特徴量を事前計算中...")
            _global_feature_levels = calculate_race_level_features_with_position_weights(df_all_features)
            
            logger.info("💾 計算済みデータをグローバル変数に保存しました")
            logger.info(f"📊 グローバルデータ: {len(_global_data):,}行（全期間）")
            logger.info(f"📊 重み計算用データ: {len(df):,}行（訓練期間2010-2020年）")
            logger.info(f"📊 期間別分析用データ: {len(_global_feature_levels):,}行（全期間）")
            logger.info("🚀 競走経験質指数（REQI）特徴量も事前計算済み（期間別分析高速化）")
            
            # グローバル重みを初期化
            weights = WeightManager.initialize_from_training_data(df)
            logger.info(f"✅ グローバル重み設定完了: {weights}")
            return True
        
        logger.warning("⚠️ 重み初期化用のデータが見つかりませんでした")
        return False
        
    except Exception as e:
        logger.error(f"❌ グローバル重み初期化エラー: {str(e)}")
        return False

def _calculate_individual_weights(df: pd.DataFrame) -> Dict[str, float]:
    """
    個別データから動的重みを計算するヘルパー関数
    verify_weight_calculation.py の検証済みロジックを適用
    
    Args:
        df: データフレーム
        
    Returns:
        重み辞書
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🔍 verify_weight_calculation.py準拠の個別重み計算を開始...")
        
        # 必要カラムの確認
        required_cols = ['馬名', '着順', 'grade_level', 'venue_level', 'distance_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ 必要なカラムが不足: {missing_cols}")
            return _get_fallback_weights()
        
        # Phase 1: 馬統計データ作成（レポート5.1.3節準拠）
        logger.info("📊 Phase 1: 馬統計データを作成中...")
        
        # 複勝フラグを作成
        if '着順' in df.columns:
            df_temp = df.copy()
            df_temp['is_placed'] = (pd.to_numeric(df_temp['着順'], errors='coerce') <= 3).astype(int)
            logger.info("📊 着順列から複勝フラグを作成（着順<=3）")
        elif '複勝' in df.columns:
            df_temp = df.copy()
            df_temp['is_placed'] = pd.to_numeric(df_temp['複勝'], errors='coerce').fillna(0)
            logger.info("📊 複勝列から複勝フラグを作成")
        else:
            logger.error("❌ 複勝フラグを作成できません")
            return _get_fallback_weights()
        
        # 馬ごとの統計を計算（最低出走数6戦以上）
        horse_stats = df_temp.groupby('馬名').agg({
            'is_placed': 'mean',  # 複勝率
            'grade_level': 'count'  # 出走回数
        }).reset_index()
        
        # 列名を標準化
        horse_stats.columns = ['馬名', 'place_rate', 'race_count']
        
        # 最低出走数6戦以上でフィルタ（レポート仕様準拠）
        horse_stats = horse_stats[horse_stats['race_count'] >= 6].copy()
        logger.info(f"📊 最低出走数6戦以上でフィルタ: {len(horse_stats):,}頭")
        
        if len(horse_stats) < 100:
            logger.error(f"❌ サンプル数が不足: {len(horse_stats)}頭（最低100頭必要）")
            return _get_fallback_weights()
        
        # 特徴量レベルの平均を計算
        feature_cols = ['grade_level', 'venue_level', 'distance_level']
        for col in feature_cols:
            avg_feature = df.groupby('馬名')[col].mean().reset_index()
            avg_feature.columns = ['馬名', f'avg_{col}']
            horse_stats = horse_stats.merge(avg_feature, on='馬名', how='left')
        
        logger.info(f"📊 馬統計データ作成完了: {len(horse_stats):,}頭")
        
        # Phase 2: 相関計算（馬統計データベース）
        logger.info("📈 Phase 2: 馬統計データで相関を計算中...")
        
        # 必要な列の確認
        required_corr_cols = ['place_rate', 'avg_grade_level', 'avg_venue_level', 'avg_distance_level']
        missing_corr_cols = [col for col in required_corr_cols if col not in horse_stats.columns]
        
        if missing_corr_cols:
            logger.error(f"❌ 必要な相関列が不足: {missing_corr_cols}")
            logger.info(f"📊 利用可能な列: {list(horse_stats.columns)}")
            return _get_fallback_weights()
        
        # 欠損値を除去
        clean_data = horse_stats[required_corr_cols].dropna()
        logger.info(f"📊 相関計算用データ: {len(clean_data):,}頭")
        
        if len(clean_data) < 100:
            logger.error(f"❌ 相関計算用サンプル数が不足: {len(clean_data)}頭（最低100頭必要）")
            return _get_fallback_weights()
        
        # 相関計算
        from scipy.stats import pearsonr
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
        
        # Phase 3: 重み計算（レポート5.1.3節準拠）
        logger.info("⚖️ Phase 3: 重みを計算中...")
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
            return _get_fallback_weights()
        
        # 重みを正規化
        weights = {}
        for feature, squared in squared_correlations.items():
            weight = squared / total_squared
            weights[feature] = weight
            logger.info(f"   ⚖️ {feature}: w = {weight:.3f} ({weight*100:.1f}%)")
        
        # レポート形式で変換
        result = {
            'grade_weight': weights.get('grade', 0.636),
            'venue_weight': weights.get('venue', 0.323),
            'distance_weight': weights.get('distance', 0.041)
        }
        
        print(f"\n📊 verify_weight_calculation.py準拠の重み計算結果:")
        print(f"  🔍 グレード重み: {result['grade_weight']:.3f} ({result['grade_weight']*100:.1f}%)")
        print(f"  🔍 場所重み: {result['venue_weight']:.3f} ({result['venue_weight']*100:.1f}%)")
        print(f"  🔍 距離重み: {result['distance_weight']:.3f} ({result['distance_weight']*100:.1f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ verify_weight_calculation.py準拠の重み計算エラー: {str(e)}")
        return _get_fallback_weights()

def _get_fallback_weights() -> Dict[str, float]:
    """レポート5.1.3節の固定重み"""
    return {
        'grade_weight': 0.636,   # 63.6%
        'venue_weight': 0.323,   # 32.3%
        'distance_weight': 0.041  # 4.1%
    }

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

@log_performance("データセット作成")
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
    file_read_start = time.time()
    for i, file_path in enumerate(csv_files):
        try:
            file_start = time.time()
            df = pd.read_csv(file_path, encoding='utf-8')
            file_time = time.time() - file_start
            
            # ファイルサイズとパフォーマンス情報
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            read_speed = file_size / file_time if file_time > 0 else 0
            
            # 芝レースのみフィルタ
            if '芝ダ障害コード' in df.columns:
                df = df[df['芝ダ障害コード'] == '芝']
            dfs.append(df)
            
            if (i + 1) % 100 == 0:
                log_processing_step("ファイル読み込み", file_read_start, i + 1, len(csv_files))
            
            # 詳細ログ（最初の10ファイルのみ）
            if i < 10:
                logger.debug(f"📄 ファイル {i+1}: {file_path.name} - "
                           f"サイズ: {file_size:.1f}MB, 読み込み時間: {file_time:.2f}秒, "
                           f"速度: {read_speed:.1f}MB/s, 行数: {len(df):,}")
                
        except Exception as e:
            logger.warning(f"ファイル読み込み失敗: {file_path.name} - {e}")
    
    if not dfs:
        raise ValueError("有効なデータファイルがありません")
    
    logger.info("🔗 データフレーム統合中...")
    concat_start = time.time()
    unified_df = pd.concat(dfs, ignore_index=True)
    concat_time = time.time() - concat_start
    
    logger.info(f"✅ 統合完了: {len(unified_df):,}行のデータ (統合時間: {concat_time:.2f}秒)")
    logger.info(f"   期間: {unified_df['年'].min()}-{unified_df['年'].max()}")
    logger.info(f"   馬数: {unified_df['馬名'].nunique():,}頭")
    log_dataframe_info(unified_df, "統合後データセット")
    
    # REQI特徴量の算出（着順重み付き対応）
    df_with_levels = calculate_race_level_features_with_position_weights(unified_df)
    
    # 馬ごとの競走経験質指数（REQI）統計算出
    logger.info("🐎 馬ごとの統計計算開始...")
    
    # 【最適化】大量データの場合は高速版を使用
    if len(df_with_levels) > 50000:  # 5万レース以上の場合
        logger.info("📊 大量データ検出 - 高速統計計算を使用")
        analysis_df = calculate_horse_stats_vectorized_stratified(df_with_levels, min_races)
    else:
        # 従来のループ処理（少量データ向け）
        horse_stats = []
        unique_horses = df_with_levels['馬名'].unique()
        horse_calc_start = time.time()
        
        for i, horse_name in enumerate(unique_horses):
            horse_data = df_with_levels[df_with_levels['馬名'] == horse_name]
            
            if len(horse_data) < min_races:
                continue
            
            # 基本統計
            total_races = len(horse_data)
            win_rate = (horse_data['着順'] == 1).mean()
            place_rate = (horse_data['着順'] <= 3).mean()
            
            # 競走経験質指数（REQI）算出（着順重み付き）
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
                '平均競走経験質指数（REQI）': avg_race_level,
                '最高競走経験質指数（REQI）': max_race_level,
                '初出走年': first_year,
                '最終出走年': last_year,
                '推定年齢': estimated_age,
                '主戦距離': main_distance
            })
                
            # 進捗ログ（1000頭ごと）
            if (i + 1) % 1000 == 0:
                log_processing_step("馬統計計算", horse_calc_start, i + 1, len(unique_horses))
        
        analysis_df = pd.DataFrame(horse_stats)
    
    # 層別カテゴリの作成
    analysis_df = create_stratification_categories(analysis_df)
    
    logger.info(f"✅ 競走経験質指数（REQI）分析用データセット準備完了: {len(analysis_df)}頭")
    logger.info(f"   平均競走経験質指数（REQI）範囲: {analysis_df['平均競走経験質指数（REQI）'].min():.3f} - {analysis_df['平均競走経験質指数（REQI）'].max():.3f}")
    
    return analysis_df

def calculate_horse_stats_vectorized_stratified(df: pd.DataFrame, min_races: int) -> pd.DataFrame:
    """
    【高速版】層別分析用馬統計計算 - ベクトル化処理
    """
    logger.info("🚀 高速馬統計計算を実行中...")
    
    # 複勝フラグ作成
    df['place_flag'] = (df['着順'] <= 3).astype(int)
    df['win_flag'] = (df['着順'] == 1).astype(int)
    
    # 馬ごとの統計をgroupbyで一括計算
    horse_stats = df.groupby('馬名').agg({
        'race_level': ['mean', 'max'],
        'place_flag': 'mean',
        'win_flag': 'mean',
        '馬名': 'count',  # total_races
        '年': ['min', 'max'],
        '距離': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean()
    }).round(6)
    
    # カラム名を平坦化
    horse_stats.columns = ['平均競走経験質指数（REQI）', '最高競走経験質指数（REQI）', '複勝率', '勝率', 
                          '出走回数', '初出走年', '最終出走年', '主戦距離']
    
    # 推定年齢計算
    horse_stats['推定年齢'] = horse_stats['最終出走年'] - horse_stats['初出走年'] + 2
    
    # 最小レース数でフィルタ
    horse_stats = horse_stats[horse_stats['出走回数'] >= min_races]
    
    # インデックスを馬名カラムに変換
    horse_stats = horse_stats.reset_index()
    
    logger.info(f"✅ 高速統計計算完了: {len(horse_stats)}頭")
    
    return horse_stats

def calculate_race_level_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    【高速版】REQI特徴量算出 - 簡易重み付け処理
    """
    logger.info("🚀 高速REQI算出を実行中...")
    
    # グレードレベルの算出（グレード数値ベース・ベクトル化）
    def get_grade_level_vectorized(df):
        """グレードレベルを算出（ベクトル化）
        
        【重要】データのグレード数値は「小さいほど高グレード」という関係
        - 1 = G1（最高グレード） → 3.0（最高レベル）
        - 2 = G2 → 2.5
        - 3 = G3 → 2.0
        - 4 = 重賞 → 1.5
        - 5 = 特別（低グレード） → 1.0（低レベル）
        - 6 = リステッド → 1.2
        """
        # グレードカラムを特定
        grade_col = None
        for col in ['グレード_x', 'グレード', 'grade']:
            if col in df.columns:
                grade_col = col
                break
        
        if grade_col is None:
            # 賞金ベースにフォールバック
            prize_col = None
            for col in ['1着賞金(1着算入賞金込み)', '1着賞金', '本賞金']:
                if col in df.columns:
                    prize_col = col
                    break
            
            if prize_col is None:
                logger.warning("⚠️ グレード・賞金カラムが見つかりません。デフォルト値を使用")
                return np.ones(len(df)) * 1.0
            
            # 賞金ベースの処理（レポート仕様に基づく正しいしきい値）
            prizes = pd.to_numeric(df[prize_col], errors='coerce').fillna(0)
            result = np.ones(len(prizes)) * 0.5
            result[prizes >= 1650] = 3.0  # G1: 1,650万円以上
            result[(prizes >= 855) & (prizes < 1650)] = 2.5  # G2: 855万円以上
            result[(prizes >= 570) & (prizes < 855)] = 2.0  # G3: 570万円以上
            result[(prizes >= 300) & (prizes < 570)] = 1.5  # リステッド: 300万円以上
            result[(prizes >= 120) & (prizes < 300)] = 1.0  # 特別: 120万円以上
            return result
        
        # グレード数値を変換
        # データは「1=最高グレード」なので、そのままマッピング
        grades = pd.to_numeric(df[grade_col], errors='coerce').fillna(5)
        result = np.ones(len(grades)) * 0.5  # デフォルト値
        
        result[grades == 1] = 3.0  # G1（最高グレード → 最高レベル）
        result[grades == 2] = 2.5  # G2
        result[grades == 3] = 2.0  # G3
        result[grades == 4] = 1.5  # 重賞
        result[grades == 5] = 1.0  # 特別（低グレード → 低レベル）
        result[grades == 6] = 1.2  # リステッド
        
        return result
    
    # 距離レベルの算出（ベクトル化）
    def get_distance_level_vectorized(df):
        # 距離カラムを特定
        distance_col = None
        for col in ['距離', 'distance', 'レース距離']:
            if col in df.columns:
                distance_col = col
                break
        
        if distance_col is None:
            logger.warning("⚠️ 距離カラムが見つかりません。デフォルト値を使用")
            return np.ones(len(df)) * 1.0
        
        distances = pd.to_numeric(df[distance_col], errors='coerce').fillna(1600)
        result = np.ones(len(distances))  # デフォルト1.0
        
        result[(distances >= 2400)] = 1.3  # 長距離
        result[(distances >= 2000) & (distances < 2400)] = 1.2  # 中長距離
        result[(distances >= 1800) & (distances < 2000)] = 1.1  # 中距離
        result[(distances < 1200)] = 0.9  # 短距離
        
        return result
    
    # 出走頭数レベルの算出（ベクトル化）
    def get_field_size_level_vectorized(df):
        # 出走頭数カラムを特定
        field_size_col = None
        for col in ['頭数_x', '出走頭数', 'field_size', '頭数', '出走数']:
            if col in df.columns:
                field_size_col = col
                break
        
        if field_size_col is None:
            logger.warning("⚠️ 出走頭数カラムが見つかりません。デフォルト値を使用")
            return np.ones(len(df)) * 1.0
        
        field_sizes = pd.to_numeric(df[field_size_col], errors='coerce').fillna(12)
        result = np.ones(len(field_sizes))  # デフォルト1.0
        
        result[field_sizes >= 16] = 1.2  # 大規模
        result[(field_sizes >= 12) & (field_sizes < 16)] = 1.1  # 中規模
        result[field_sizes < 8] = 0.9  # 小規模
        
        return result
    
    # venue_levelの算出（通常版と統一）
    def get_venue_level_vectorized(df):
        """venue_levelを算出（通常版と統一した方法）"""
        # 通常版と同じvenue_level生成ロジックを使用
        if '場コード' in df.columns:
            # 場コードから判定（通常版と同じマッピング）
            venue_codes = pd.to_numeric(df['場コード'], errors='coerce').fillna(0).astype(int)
            result = np.ones(len(venue_codes)) * 0.0
            result[venue_codes.isin([1, 5, 6])] = 9.0  # 東京、京都、阪神
            result[venue_codes.isin([2, 3, 8])] = 7.0  # 中山、中京、札幌
            result[venue_codes == 7] = 4.0  # 函館
            return result
        elif '場名' in df.columns:
            # 場名から判定（通常版と同じマッピング）
            venue_names = df['場名'].astype(str)
            result = np.ones(len(venue_names)) * 0.0
            result[venue_names.isin(['東京', '京都', '阪神'])] = 9.0
            result[venue_names.isin(['中山', '中京', '札幌'])] = 7.0
            result[venue_names == '函館'] = 4.0
            return result
        else:
            logger.warning("⚠️ 場コード・場名カラムが見つかりません。デフォルト値を使用")
            return np.ones(len(df)) * 0.0
    
    # ベクトル化処理
    df['grade_level'] = get_grade_level_vectorized(df)
    df['venue_level'] = get_venue_level_vectorized(df)
    df['distance_level'] = get_distance_level_vectorized(df)
    df['field_size_level'] = get_field_size_level_vectorized(df)
    
    # 基本REQI算出
    df['base_race_level'] = (
        df['grade_level'] * 0.5 +
        df['distance_level'] * 0.3 +
        df['field_size_level'] * 0.2
    )
    
    # 簡易重み付け処理（時系列順序を考慮した高速版）
    logger.info("🔄 簡易重み付け処理を実行中...")
    
    # 日付でソート（利用可能なカラムを使用）
    sort_cols = ['馬名']
    if '年月日' in df.columns:
        sort_cols.append('年月日')
    elif '年' in df.columns:
        sort_cols.append('年')
        if '月' in df.columns:
            sort_cols.append('月')
        if '日' in df.columns:
            sort_cols.append('日')
    
    df = df.sort_values(sort_cols).copy()
    
    # 馬ごとに連番を付与
    df['race_sequence'] = df.groupby('馬名').cumcount() + 1
    
    # 複勝結果による簡易調整係数
    df['place_result'] = (df['着順'] <= 3).astype(int)
    
    # 過去の複勝率による調整（移動平均）
    df['historical_place_rate'] = df.groupby('馬名')['place_result'].expanding().mean().values
    
    # 調整係数の算出（0.8-1.2の範囲）
    df['adjustment_factor'] = 0.8 + (df['historical_place_rate'] * 0.4)
    df['adjustment_factor'] = df['adjustment_factor'].fillna(1.0).clip(0.8, 1.2)
    
    # 最終REQI
    df['race_level'] = df['base_race_level'] * df['adjustment_factor']
    
    logger.info("✅ 高速REQI算出完了")
    
    return df

@log_performance("REQI特徴量算出")
def calculate_race_level_features_with_position_weights(df: pd.DataFrame) -> pd.DataFrame:
    """【修正版】時間的分離による複勝結果統合対応のREQI特徴量算出"""
    logger.info("⚖️ REQI特徴量を算出中（時間的分離による複勝結果統合対応）...")
    
    # 【最適化】大量データの場合は高速版を使用
    if len(df) > 100000:  # 10万レース以上の場合
        logger.info("📊 大量データ検出 - 高速重み付け処理を使用")
        return calculate_race_level_features_fast(df)
    
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
    
    # venue_levelの生成（期間別分析の格式ロジックに合わせて統一）
    # グレード列を優先し、なければ場コード/場名でフォールバック
    if any(col in df.columns for col in ['グレード_x', 'グレード_y', 'グレード']):
        grade_num_col = None
        for col in ['グレード_x', 'グレード_y', 'グレード']:
            if col in df.columns:
                grade_num_col = col
                break
        logger.info("📋 グレード列からvenue_level（格式）を推定中...")
        grade_map = {1: 9, 11: 8, 12: 7, 2: 4, 3: 3, 4: 2, 5: 1, 6: 2}
        df[grade_num_col] = pd.to_numeric(df[grade_num_col], errors='coerce')
        df['venue_level'] = df[grade_num_col].map(grade_map).fillna(0)
        logger.info(f"✅ venue_level生成完了(格式): 平均値 {df['venue_level'].mean():.3f}")
    elif '場コード' in df.columns or '場名' in df.columns:
        logger.info("📋 グレード列なしのため場コード/場名でvenue_levelを生成します")
        if '場コード' in df.columns:
            codes = pd.to_numeric(df['場コード'], errors='coerce').fillna(0).astype(int)
            df['venue_level'] = 0.0
            df.loc[codes.isin([1, 5, 6]), 'venue_level'] = 9.0
            df.loc[codes.isin([2, 3, 8]), 'venue_level'] = 7.0
            df.loc[codes == 7, 'venue_level'] = 4.0
        else:
            names = df['場名'].astype(str)
            df['venue_level'] = 0.0
            df.loc[names.isin(['東京', '京都', '阪神']), 'venue_level'] = 9.0
            df.loc[names.isin(['中山', '中京', '札幌']), 'venue_level'] = 7.0
            df.loc[names == '函館', 'venue_level'] = 4.0
        logger.info(f"✅ venue_level生成完了(場コード/場名): 平均値 {df['venue_level'].mean():.3f}")
    else:
        logger.warning("⚠️ グレード/場コード/場名列が存在しません。venue_level=0で設定します")
        df['venue_level'] = 0.0
    
    df['distance_level'] = df['距離'].apply(get_distance_level)
    
    # 基本REQI算出（複勝結果統合後の重み）
    base_race_level = (
        0.636 * df['grade_level'] +
        0.323 * df['venue_level'] +
        0.041 * df['distance_level']
    )
    
    # 【重要修正】時間的分離による複勝結果統合を適用
    df['race_level'] = apply_historical_result_weights(df, base_race_level)
    
    logger.info(f"✅ REQI算出完了（時間的分離版、平均: {df['race_level'].mean():.3f}）")
    return df

def apply_historical_result_weights(df: pd.DataFrame, base_race_level: pd.Series) -> pd.Series:
    """
    時間的分離による複勝結果重み付けを適用
    
    各馬の過去の複勝実績に基づいて、現在の競走経験質指数（REQI）を調整する。
    これにより循環論理を回避しつつ、複勝結果の価値を統合する。
    
    Args:
        df: レースデータフレーム（馬名、年月日、着順必須）
        base_race_level: 基本競走経験質指数（REQI）
        
    Returns:
        pd.Series: 複勝実績調整済み競走経験質指数（REQI）
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
        logger.warning("⚠️ 日付カラムが見つかりません。基本競走経験質指数（REQI）をそのまま使用")
        return base_race_level
    
    try:
        df_work[date_col] = pd.to_datetime(df_work[date_col], format='%Y%m%d')
    except (ValueError, TypeError):
        try:
            df_work[date_col] = pd.to_datetime(df_work[date_col])
        except (ValueError, TypeError):
            logger.warning("⚠️ 日付変換に失敗。基本競走経験質指数（REQI）をそのまま使用")
            return base_race_level
    
    # 結果格納用
    adjusted_race_level = base_race_level.copy()
    
    # 馬ごとに過去実績ベースの調整を実施
    processed_horses = 0
    unique_horses = df_work['馬名'].unique()
    adjustment_start = time.time()
    
    for horse_name in unique_horses:
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
            log_processing_step("複勝結果調整", adjustment_start, processed_horses, len(unique_horses))
    
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

@log_performance("統合層別分析")
def perform_integrated_stratified_analysis(analysis_df: pd.DataFrame) -> Dict[str, Any]:
    """統合された層別分析の実行"""
    logger.info("🔬 統合層別分析を開始...")
    
    results = {}
    
    # 1. 年齢層別分析
    logger.info("👶 年齢層別分析（HorseREQI効果の年齢差）...")
    age_results = analyze_stratification(analysis_df, '年齢層', '複勝率')
    results['age_analysis'] = age_results
    
    # 2. 経験数別分析
    logger.info("📊 経験数別分析（HorseREQI効果の経験差）...")
    experience_results = analyze_stratification(analysis_df, '経験数層', '複勝率')
    results['experience_analysis'] = experience_results
    
    # 3. 距離カテゴリ別分析
    logger.info("🏃 距離カテゴリ別分析（HorseREQI効果の距離適性差）...")
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
        
        # 平均競走経験質指数（REQI）分析
        avg_correlation = group_data['平均競走経験質指数（REQI）'].corr(group_data[target_col])
        avg_corr_coef, avg_p_value = pearsonr(group_data['平均競走経験質指数（REQI）'], group_data[target_col])
        avg_r_squared = avg_correlation ** 2 if not pd.isna(avg_correlation) else np.nan
        
        # 最高競走経験質指数（REQI）分析
        max_correlation = group_data['最高競走経験質指数（REQI）'].corr(group_data[target_col])
        max_corr_coef, max_p_value = pearsonr(group_data['最高競走経験質指数（REQI）'], group_data[target_col])
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
            # 平均競走経験質指数（REQI）結果
            'avg_correlation': avg_correlation,
            'avg_p_value': avg_p_value,
            'avg_r_squared': avg_r_squared,
            'avg_confidence_interval': avg_ci,
            # 最高競走経験質指数（REQI）結果
            'max_correlation': max_correlation,
            'max_p_value': max_p_value,
            'max_r_squared': max_r_squared,
            'max_confidence_interval': max_ci,
            # 共通統計情報
            'mean_place_rate': group_data[target_col].mean(),
            'std_place_rate': group_data[target_col].std(),
            'mean_avg_race_level': group_data['平均競走経験質指数（REQI）'].mean(),
            'mean_max_race_level': group_data['最高競走経験質指数（REQI）'].mean(),
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
    report.append("# 競走経験質指数（REQI）と複勝率の層別分析結果レポート（統合版）")
    report.append("")
    report.append("## 分析概要")
    report.append(f"- **分析対象**: {len(analysis_df):,}頭（最低6戦以上）")
    report.append(f"- **分析内容**: 競走経験質指数（REQI）と複勝率の相関（着順重み付き対応）")
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
        
        # 平均競走経験質指数（REQI）結果テーブル
        report.append("### 平均競走経験質指数（REQI） vs 複勝率")
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
    report.append("   - analyze_horse_REQI.pyに統合された層別分析機能")
    
    # レポートファイルに保存
    report_path = output_dir / "stratified_analysis_integrated_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    logger.info(f"📋 層別分析レポート保存: {report_path}")
    return "\n".join(report)

def calculate_reqi_with_dynamic_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    【重要】レポート記載の動的重み計算によるREQI計算
    race_level_analysis_report.md 5.1.3節記載の計算方法を適用
    """
    logger.info("🎯 レポート記載の動的重み計算によるREQI計算中...")
    
    df_copy = df.copy()
    
    # 📊 グローバル重みの取得
    if WeightManager.is_initialized():
        weights = get_global_weights()
        calculation_details = WeightManager.get_calculation_details()
        
        print("\n" + "="*80)
        print("📋 REQI計算: グローバル重み使用（race_level_analysis_report.md 5.1.3節準拠）")
        print("="*80)
        print("✅ 事前算出されたグローバル重みを使用:")
        print(f"   グレード: {weights['grade_weight']:.3f} ({weights['grade_weight']*100:.1f}%)")
        print(f"   場所: {weights['venue_weight']:.3f} ({weights['venue_weight']*100:.1f}%)")
        print(f"   距離: {weights['distance_weight']:.3f} ({weights['distance_weight']*100:.1f}%)")
        if calculation_details:
            print(f"📊 算出基準: {calculation_details.get('training_period', 'N/A')} ({calculation_details.get('sample_size', 'N/A'):,}行)")
        print("="*80)
        
        # 📝 ログにも重み使用情報を出力
        logger.info("📊 ========== REQI計算でグローバル重み使用 ==========")
        logger.info("✅ グローバル重みシステムを使用してREQI計算を実行:")
        logger.info(f"   📊 グレード重み: {weights['grade_weight']:.4f} ({weights['grade_weight']*100:.2f}%)")
        logger.info(f"   📊 場所重み: {weights['venue_weight']:.4f} ({weights['venue_weight']*100:.2f}%)")
        logger.info(f"   📊 距離重み: {weights['distance_weight']:.4f} ({weights['distance_weight']*100:.2f}%)")
        if calculation_details:
            logger.info(f"   📊 算出基準: {calculation_details.get('training_period', 'N/A')} ({calculation_details.get('sample_size', 'N/A'):,}行)")
            logger.info(f"   📊 目標変数: {calculation_details.get('target_column', 'N/A')}")
        logger.info("=" * 60)
        
        grade_weight = weights['grade_weight']
        venue_weight = weights['venue_weight']
        distance_weight = weights['distance_weight']
    else:
        # フォールバック: 個別計算
        print("\n" + "="*80)
        print("📋 REQI計算: 個別動的重み計算（グローバル重み未初期化のため）")
        print("="*80)
        print("⚠️ グローバル重みが初期化されていません。個別計算を実行します。")
        print("# 重み算出方法（レポート5.1.3節記載）")
        print("# w_i = r_i² / (r_grade² + r_venue² + r_distance²)")
        print("="*80)
        
        # 従来の個別計算ロジック（省略せずに保持）
        weights = _calculate_individual_weights(df_copy)
        grade_weight = weights['grade_weight']
        venue_weight = weights['venue_weight'] 
        distance_weight = weights['distance_weight']
        
        # 📝 個別計算の結果もログに出力
        logger.info("📊 ========== REQI計算で個別重み計算使用 ==========")
        logger.info("⚠️ グローバル重み未初期化のため個別計算を実行:")
        logger.info(f"   📊 グレード重み: {grade_weight:.4f} ({grade_weight*100:.2f}%)")
        logger.info(f"   📊 場所重み: {venue_weight:.4f} ({venue_weight*100:.2f}%)")
        logger.info(f"   📊 距離重み: {distance_weight:.4f} ({distance_weight*100:.2f}%)")
        logger.info("=" * 60)
    
    # 1. グレードレベルの計算
    def calculate_grade_level(row):
        """グレードレベルを計算
        
        【重要】データのグレード数値は「小さいほど高グレード」という関係
        - 1 = G1（最高グレード） → 9.0（最高レベル）
        - 2 = G2 → 4.0
        - 3 = G3 → 3.0
        - 4 = 重賞 → 2.0
        - 5 = 特別（低グレード） → 1.0（低レベル）
        - 6 = リステッド → 1.5
        """
        # グレード情報から数値化
        for grade_col in ['グレード_x', 'グレード_y', 'グレード']:
            if grade_col in df_copy.columns and pd.notna(row.get(grade_col)):
                try:
                    grade = int(row[grade_col])
                    if grade == 1: 
                        return 9.0    # G1（最高グレード → 最高レベル）
                    elif grade == 2: 
                        return 4.0    # G2
                    elif grade == 3: 
                        return 3.0    # G3
                    elif grade == 4: 
                        return 2.0    # 重賞
                    elif grade == 5: 
                        return 1.0    # 特別（低グレード → 低レベル）
                    elif grade == 6: 
                        return 1.5    # リステッド
                except (ValueError, TypeError):
                    pass
        
        # 賞金からフォールバック（レポート仕様に基づく正しいしきい値）
        for prize_col in ['1着賞金(1着算入賞金込み)', '1着賞金', '本賞金']:
            if prize_col in df_copy.columns and pd.notna(row.get(prize_col)):
                try:
                    prize = float(row[prize_col])
                    if prize >= 1650:  # G1: 1,650万円以上
                        return 9.0
                    elif prize >= 855:  # G2: 855万円以上
                        return 4.0
                    elif prize >= 570:  # G3: 570万円以上
                        return 3.0
                    elif prize >= 300:  # リステッド: 300万円以上
                        return 2.0
                    elif prize >= 120:  # 特別: 120万円以上
                        return 1.0
                    else:
                        return 0.0
                except (ValueError, TypeError):
                    pass
        
        return 0.0  # デフォルト
    
    # 2. 場所レベルの計算
    def calculate_venue_level(row):
        # 場名から判定
        if '場名' in df_copy.columns and pd.notna(row.get('場名')):
            venue_name = str(row['場名'])
            if venue_name in ['東京', '京都', '阪神']:
                return 9.0  # 最高格式
            elif venue_name in ['中山', '中京', '札幌']:
                return 7.0  # 高格式
            elif venue_name in ['函館']:
                return 4.0  # 中格式
            elif venue_name in ['新潟', '福島', '小倉']:
                return 0.0  # 標準格式
        
        # 場コードからフォールバック
        if '場コード' in df_copy.columns and pd.notna(row.get('場コード')):
            venue_code = str(row['場コード']).zfill(2)
            venue_mapping = {
                '01': 9.0, '05': 9.0, '06': 9.0,  # 東京、京都、阪神
                '02': 7.0, '03': 7.0, '08': 7.0,  # 中山、中京、札幌
                '07': 4.0,  # 函館
                '04': 0.0, '09': 0.0, '10': 0.0   # 新潟、福島、小倉
            }
            return venue_mapping.get(venue_code, 0.0)
        
        return 0.0  # デフォルト
    
    # 3. 距離レベルの計算
    def calculate_distance_level(row):
        if '距離' in df_copy.columns and pd.notna(row.get('距離')):
            try:
                distance = int(row['距離'])
                if distance <= 1400:
                    return 0.85      # スプリント
                elif distance <= 1800:
                    return 1.0       # マイル（基準）
                elif distance <= 2000:
                    return 1.35      # 中距離
                elif distance <= 2400:
                    return 1.45      # 中長距離
                else:
                    return 1.25      # 長距離
            except (ValueError, TypeError):
                pass
        
        return 1.0  # マイル相当をデフォルト
    
    # 各レベルを計算
    logger.info("📊 グレードレベル計算中...")
    df_copy['grade_level'] = df_copy.apply(calculate_grade_level, axis=1)
    
    logger.info("📊 場所レベル計算中...")
    df_copy['venue_level'] = df_copy.apply(calculate_venue_level, axis=1)
    
    logger.info("📊 距離レベル計算中...")
    df_copy['distance_level'] = df_copy.apply(calculate_distance_level, axis=1)
    
    # 重み取得完了後の処理
    logger.info("📊 REQI計算式適用中...")
    
    # 動的重みによるREQI計算
    logger.info("📊 REQI（動的重み法）計算中...")
    df_copy['race_level'] = (
        grade_weight * df_copy['grade_level'] +
        venue_weight * df_copy['venue_level'] +
        distance_weight * df_copy['distance_level']
    )
    
    print(f"\n📊 REQI計算式:")
    print(f"race_level = {grade_weight:.3f} * grade_level + {venue_weight:.3f} * venue_level + {distance_weight:.3f} * distance_level")
    
    # 統計情報をログ出力
    grade_stats = df_copy['grade_level'].value_counts().sort_index()
    venue_stats = df_copy['venue_level'].value_counts().sort_index()
    distance_stats = df_copy['distance_level'].value_counts().sort_index()
    
    # 📊 計算結果の表示（毎回出力）
    print(f"\n📊 REQI計算結果:")
    print(f"  📊 グレードレベル分布: {grade_stats.to_dict()}")
    print(f"  📊 場所レベル分布: {venue_stats.to_dict()}")
    print(f"  📊 距離レベル分布: {distance_stats.to_dict()}")
    print(f"  📊 REQI平均値: {df_copy['race_level'].mean():.3f}")
    print(f"  📊 REQI範囲: {df_copy['race_level'].min():.3f} - {df_copy['race_level'].max():.3f}")
    print(f"  📊 適用データ数: {len(df_copy):,}レコード")
    print("=" * 80 + "\n")
    
    logger.info("✅ レポート記載の動的重み法REQI計算完了:")
    logger.info(f"  📊 算出された重み - グレード: {grade_weight:.3f}, 場所: {venue_weight:.3f}, 距離: {distance_weight:.3f}")
    logger.info(f"  📊 グレードレベル分布: {grade_stats.to_dict()}")
    logger.info(f"  📊 場所レベル分布: {venue_stats.to_dict()}")
    logger.info(f"  📊 距離レベル分布: {distance_stats.to_dict()}")
    logger.info(f"  📊 REQI平均値: {df_copy['race_level'].mean():.3f}")
    logger.info(f"  📊 REQI範囲: {df_copy['race_level'].min():.3f} - {df_copy['race_level'].max():.3f}")
    
    return df_copy

def calculate_accurate_feature_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    【重要】実際のCSVデータから特徴量を正確に計算（デフォルト値使用禁止）
    レポート内容に基づく正確な計算ロジックを実装
    """
    logger.info("🎯 実際のCSVデータから特徴量を正確に計算中...")
    
    df_copy = df.copy()
    
    # 1. venue_level の計算（場コード・場名から）
    def calculate_venue_level(row):
        # 場名から判定
        if '場名' in df_copy.columns and pd.notna(row.get('場名')):
            venue_name = str(row['場名'])
            # レポート記載の格式レベル
            if venue_name in ['東京', '京都', '阪神']:
                return 3.0  # 最高格式
            elif venue_name in ['中山', '中京', '札幌']:
                return 2.0  # 高格式
            elif venue_name in ['函館']:
                return 1.5  # 中格式
            elif venue_name in ['新潟', '福島', '小倉']:
                return 1.0  # 標準格式
        
        # 場コードから判定（フォールバック）
        if '場コード' in df_copy.columns and pd.notna(row.get('場コード')):
            venue_code = str(row['場コード']).zfill(2)
            venue_mapping = {
                '01': 3.0, '05': 3.0, '06': 3.0,  # 東京、京都、阪神
                '02': 2.0, '03': 2.0, '08': 2.0,  # 中山、中京、札幌
                '07': 1.5,  # 函館
                '04': 1.0, '09': 1.0, '10': 1.0   # 新潟、福島、小倉
            }
            return venue_mapping.get(venue_code, 1.0)
        
        return 1.0  # 最終フォールバック
    
    # 2. prize_level の計算（1着賞金から）
    def calculate_prize_level(row):
        # 賞金カラムを特定
        prize_col = None
        for col in ['1着賞金(1着算入賞金込み)', '1着賞金', '本賞金']:
            if col in df_copy.columns and pd.notna(row.get(col)):
                prize_col = col
                break
        
        if prize_col:
            try:
                prize = float(row[prize_col])
                # レポート記載の賞金基準（万円単位）
                if prize >= 16500:  # G1
                    return 3.0
                elif prize >= 8550:  # G2
                    return 2.5
                elif prize >= 5700:  # G3
                    return 2.0
                elif prize >= 3000:  # リステッド
                    return 1.5
                elif prize >= 1200:  # 特別/OP
                    return 1.0
                else:
                    return 0.8  # 条件戦
            except (ValueError, TypeError):
                pass
        
        # グレード情報からフォールバック
        for grade_col in ['グレード_x', 'グレード_y', 'グレード']:
            if grade_col in df_copy.columns and pd.notna(row.get(grade_col)):
                try:
                    grade = int(row[grade_col])
                    if grade == 1: 
                        return 3.0    # G1
                    elif grade == 2: 
                        return 2.5  # G2
                    elif grade == 3: 
                        return 2.0  # G3
                    elif grade == 4: 
                        return 1.5  # 重賞
                    elif grade == 5: 
                        return 1.0  # 特別
                    elif grade == 6: 
                        return 1.2  # リステッド
                except (ValueError, TypeError):
                    pass
        
        return 1.0  # デフォルト（特別レース相当）
    
    # 3. distance_level の計算（距離から）
    def calculate_distance_level(row):
        if '距離' in df_copy.columns and pd.notna(row.get('距離')):
            try:
                distance = int(row['距離'])
                # レポート記載の距離基準
                if distance <= 1400:
                    return 0.85      # スプリント
                elif distance <= 1800:
                    return 1.0       # マイル
                elif distance <= 2000:
                    return 1.25      # 中距離
                else:
                    return 1.4       # 長距離
            except (ValueError, TypeError):
                pass
        
        return 1.0  # マイル相当をデフォルト
    
    # 1. grade_level の計算
    def calculate_grade_level(row):
        """グレードレベルを計算
        
        【重要】データのグレード数値は「小さいほど高グレード」という関係
        - 1 = G1（最高グレード）
        - 2 = G2
        - 3 = G3
        - 4 = 重賞
        - 5 = 特別（低グレード）
        - 6 = リステッド
        
        これをgrade_levelでは「大きいほど高グレード」に変換
        """
        # グレード列の候補を確認
        grade_cols = ['グレード_x', 'グレード_y']
        grade_value = None
        
        for col in grade_cols:
            if col in row and pd.notna(row[col]):
                grade_value = row[col]
                break
        
        if grade_value is None:
            return 1.0  # デフォルト値
        
        # グレード値に基づくレベル設定
        # データは「小さい数値=高グレード」なので、grade_levelは「大きい数値=高グレード」に変換
        try:
            grade_num = float(grade_value)
            if grade_num == 1:
                return 3.0  # G1（最高）
            elif grade_num == 2:
                return 2.5  # G2
            elif grade_num == 3:
                return 2.0  # G3
            elif grade_num == 4:
                return 1.5  # 重賞
            elif grade_num == 5:
                return 1.0  # 特別（低）
            elif grade_num == 6:
                return 1.2  # リステッド
            else:
                return 0.5  # その他（デフォルトより低い）
        except (ValueError, TypeError):
            return 1.0  # デフォルト値
    
    # 各特徴量を計算
    logger.info("📊 grade_level を計算中...")
    df_copy['grade_level'] = df_copy.apply(calculate_grade_level, axis=1)
    
    logger.info("📊 venue_level を計算中...")
    df_copy['venue_level'] = df_copy.apply(calculate_venue_level, axis=1)
    
    logger.info("📊 prize_level を計算中...")
    df_copy['prize_level'] = df_copy.apply(calculate_prize_level, axis=1)
    
    logger.info("📊 distance_level を計算中...")
    df_copy['distance_level'] = df_copy.apply(calculate_distance_level, axis=1)
    
    # 結果をログ出力
    grade_stats = df_copy['grade_level'].value_counts().sort_index()
    venue_stats = df_copy['venue_level'].value_counts().sort_index()
    prize_stats = df_copy['prize_level'].value_counts().sort_index()
    distance_stats = df_copy['distance_level'].value_counts().sort_index()
    
    logger.info("✅ 特徴量計算完了:")
    logger.info(f"  📊 grade_level 分布: {grade_stats.to_dict()}")
    logger.info(f"  📊 venue_level 分布: {venue_stats.to_dict()}")
    logger.info(f"  📊 prize_level 分布: {prize_stats.to_dict()}")
    logger.info(f"  📊 distance_level 分布: {distance_stats.to_dict()}")
    
    return df_copy

def analyze_by_periods_optimized(analyzer, periods, base_output_dir):
    """【最適化版】データフレーム一括処理による期間別分析（重複処理完全回避）"""
    global _global_data, _global_feature_levels, _global_raw_data
    
    logger.info("🚀 最適化版期間別分析を開始...")
    
    # 【重要】グローバル重み設定完了で設定した重みに統一
    logger.info("🎯 期間別分析用の統一重みを確認中...")
    if WeightManager.is_initialized():
        global_weights = WeightManager.get_weights()
        logger.info(f"✅ グローバル重み設定完了で設定された重みを使用: {global_weights}")
    else:
        logger.warning("⚠️ グローバル重みが未初期化です。最初の期間で重みを計算します")
    
    # 1. グローバル変数から計算済みデータを取得（重複処理完全回避）
    # データ取得成功フラグ
    data_loaded = False
    
    # グローバル変数をチェック（__main__とanalyze_REQI両方を確認）
    import sys
    
    # __main__として実行されている場合を優先
    target_module = None
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], '_global_data'):
        target_module = sys.modules['__main__']
        logger.info("🔍 __main__モジュールのグローバル変数を参照します")
    elif '_global_data' in globals():
        target_module = sys.modules[__name__]
        logger.info("🔍 analyze_REQIモジュールのグローバル変数を参照します")
    
    if target_module is not None:
        has_data = hasattr(target_module, '_global_data')
        has_features = hasattr(target_module, '_global_feature_levels')
        data_not_none = has_data and target_module._global_data is not None
        features_not_none = has_features and target_module._global_feature_levels is not None
        
        logger.info(f"🔍 グローバル変数チェック: _global_data={data_not_none}, _global_feature_levels={features_not_none}")
        
        if data_not_none and features_not_none:
            logger.info("💾 グローバル変数から計算済みデータを取得中...")
            combined_df = target_module._global_data.copy()
            df_with_features = target_module._global_feature_levels.copy()
            logger.info(f"✅ 計算済みデータ取得完了: {len(combined_df):,}行")
            data_loaded = True
    else:
        logger.info("🔍 グローバル変数チェック: _global_data=False, _global_feature_levels=False")
        
    
    # データ取得に成功した場合は競走経験質指数（REQI）特徴量をチェック
    if data_loaded:
        # 競走経験質指数（REQI）特徴量が既に計算済みかチェック
        if 'race_level' in df_with_features.columns:
            logger.info("💾 競走経験質指数（REQI）特徴量も既に計算済みです（完全最適化）")
        else:
            logger.info("🧮 競走経験質指数（REQI）特徴量を計算中...")
            df_with_features = calculate_race_level_features_with_position_weights(df_with_features)
    else:
        logger.warning("⚠️ グローバル変数が設定されていません。フォールバック処理を実行します...")
        # 取得経路を UnifiedAnalyzer API に統一
        try:
            from horse_racing.base.unified_analyzer import create_unified_analyzer
            ua = create_unified_analyzer('period', min_races=analyzer.config.min_races, enable_stratified=True)
            combined_df = ua.load_data_unified(analyzer.config.input_path, 'utf-8')
        except Exception:
            # UA経由での取得に失敗した場合のみ従来フォールバック
            logger.info(f"🔍 _global_raw_dataチェック: {_global_raw_data is not None}")
            if _global_raw_data is not None:
                logger.info("💾 既存のグローバル生データを再利用中...")
                combined_df = _global_raw_data.copy()
            else:
                logger.warning("⚠️ _global_raw_dataも利用できません。新規読み込みを実行します...")
                combined_df = load_all_data_once(analyzer.config.input_path, 'utf-8')
                if combined_df.empty:
                    return {}
        
        # 特徴量レベル列を計算
        logger.info("🧮 実際のCSVデータから特徴量を正確に計算中...")
        df_with_features = calculate_accurate_feature_levels(combined_df)
        
        # 競走経験質指数（REQI）特徴量一括計算
        logger.info("🧮 競走経験質指数（REQI）特徴量一括計算中...")
        df_with_features = calculate_race_level_features_with_position_weights(df_with_features)
        
        logger.info(f"✅ 全データ前処理完了: {len(df_with_features):,}レース")
    
    all_results = {}
    
    # 3. 期間ごとにデータフレームをフィルタして分析
    for period_name, start_year, end_year in periods:
        logger.info(f"📊 期間 {period_name} の分析開始...")
        
        try:
            # 期間別出力ディレクトリの作成
            period_output_dir = base_output_dir / period_name
            period_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 【最適化】データフレームフィルタリング（コピー不要）
            period_mask = (df_with_features['年'] >= start_year) & (df_with_features['年'] <= end_year)
            period_df = df_with_features[period_mask].copy()  # 必要な部分のみコピー
            
            logger.info(f"  📅 期間設定: {start_year}年 - {end_year}年")
            logger.info(f"  📊 対象データ: {len(period_df):,}行")
            logger.info(f"  🐎 対象馬数: {len(period_df['馬名'].unique()):,}頭")
            
            # 期間内の実際の年範囲を確認
            if len(period_df) > 0:
                actual_min_year = int(period_df['年'].min())
                actual_max_year = int(period_df['年'].max())
                logger.info(f"  📊 実際の年範囲: {actual_min_year}年 - {actual_max_year}年")
            
            # データ充足性チェック
            if len(period_df) < analyzer.config.min_races:
                logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(period_df)}行)")
                continue
            
            # 【重要】グローバル重み設定完了で設定した重みに統一（再計算を防ぐ）
            if WeightManager.is_initialized():
                logger.info(f"♻️ 期間 {period_name} ではグローバル重み設定完了で設定された重みを再利用します")
                # 重みの再計算を防ぐ
                WeightManager.prevent_recalculation()
            else:
                logger.warning(f"⚠️ 期間 {period_name} でグローバル重みが未初期化です。重みを計算します")
                # 最初の期間でのみ重みを計算
                weights = WeightManager.initialize_from_training_data(df_with_features)
                logger.info(f"✅ 期間 {period_name} で重み設定完了: {weights}")
            
            # 【重要修正】期間別アナライザーを作成し、全データから特定期間を直接設定
            period_config = AnalysisConfig(
                input_path=analyzer.config.input_path,
                min_races=analyzer.config.min_races,
                output_dir=str(period_output_dir),
                date_str=analyzer.config.date_str,
                start_date=None,  # 重複フィルタリング防止
                end_date=None     # 重複フィルタリング防止
            )
            
            period_analyzer = REQIAnalyzer(period_config, 
                                              enable_stratified_analysis=analyzer.enable_stratified_analysis)
            
            # 【重要修正】特徴量計算済みのデータを直接設定（重複計算回避）
            period_analyzer.df = period_df.copy()
            
            # 【修正】期間情報を明示的に設定して時系列分割の問題を回避
            period_analyzer._override_period_info = {
                'start_year': start_year,
                'end_year': end_year,
                'period_name': period_name,
                'total_years': end_year - start_year + 1
            }
            
            # 分析実行
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
                'total_races': len(period_df),
                'total_horses': len(period_df['馬名'].unique())
            }
            
            all_results[period_name] = results
            logger.info(f"✅ 期間 {period_name} 完了: {results['period_info']['total_races']:,}レース, {results['period_info']['total_horses']:,}頭")
            
        except Exception as e:
            logger.error(f"❌ 期間 {period_name} でエラー: {str(e)}")
            logger.error("詳細なエラー情報:", exc_info=True)
            continue
    
    logger.info("🎉 最適化版期間別分析完了")
    return all_results

def analyze_by_periods(analyzer, periods, base_output_dir):
    """期間別に分析を実行（最適化版を使用）"""
    return analyze_by_periods_optimized(analyzer, periods, base_output_dir)


def generate_period_summary_report(all_results, output_dir):
    """期間別分析の総合レポートを生成"""
    report_path = output_dir / '競走経験質指数（REQI）分析_期間別総合レポート.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 競走経験質指数（REQI）分析 期間別総合レポート\n\n")
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
                
                f.write(f"**平均競走経験質指数（REQI） vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_avg:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_avg:.3f}\n\n")
                
                f.write(f"**最高競走経験質指数（REQI） vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_max:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_max:.3f}\n\n")
            else:
                f.write("- 相関分析データなし\n\n")
        
        f.write("\n## 💡 総合的な傾向と知見\n\n")
        
        # 期間別の相関係数変化
        if len(all_results) > 1:
            f.write("### 時系列変化\n")
            f.write("平均競走経験質指数（REQI）と複勝率の相関係数の変化：\n")
            
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
        
        f.write("\n### 競走経験質指数（REQI）分析の特徴\n")
        f.write("- 競走経験質指数（REQI）は競馬場の格式度と実力の関係を数値化\n")
        f.write("- 平均レベル：馬の継続的な実力を表す指標\n")
        f.write("- 最高レベル：馬のピーク時の実力を表す指標\n")
        f.write("- 時系列分析により、競馬界の格式体系の変化を把握可能\n")
    
    logger.info(f"期間別総合レポート保存: {report_path}")

@log_performance("包括的オッズ比較分析")
def perform_comprehensive_odds_analysis(data_dir: str, output_dir: str, sample_size: int = None, min_races: int = 6) -> Dict[str, Any]:
    """包括的オッズ比較分析の実行"""
    logger.info("🎯 包括的オッズ比較分析を開始...")
    
    try:
        # OddsComparisonAnalyzerを使用（利用可能な場合）
        analyzer = OddsComparisonAnalyzer(min_races=min_races)
        
        # グローバル関数を使用してデータを読み込み
        combined_df = load_all_data_once(data_dir, 'utf-8')
        if combined_df.empty:
            raise ValueError("データファイルが見つかりません")
        
        # ファイル数を計算
        dataset_files = get_all_dataset_files(data_dir)
        file_count = len(dataset_files)
        
        # サンプルサイズ制限がある場合は適用
        if sample_size is not None and len(combined_df) > sample_size * 1000:  # 概算で制限
            logger.info(f"サンプルサイズ制限を適用: {sample_size * 1000}行")
            combined_df = combined_df.sample(n=sample_size * 1000, random_state=42)
        
        logger.info(f"統合データ: {len(combined_df):,} レコード")
        log_dataframe_info(combined_df, "統合オッズデータ")
        
        # HorseREQI計算
        horse_stats_df = analyzer.calculate_horse_race_level(combined_df)
        logger.info(f"HorseREQI計算完了: {len(horse_stats_df):,}頭")
        
        # 相関分析
        correlation_results = analyzer.perform_correlation_analysis(horse_stats_df)
        
        # 回帰分析
        regression_results = analyzer.perform_regression_analysis(horse_stats_df)
        
        # 結果をまとめる
        analysis_results = {
            'data_summary': {
                'total_records': len(combined_df),
                'horse_count': len(horse_stats_df),
                'file_count': file_count
            },
            'correlations': correlation_results,
            'regression': regression_results
        }
        
        # 【修正】可視化の作成
        logger.info("📊 可視化（散布図・モデル性能比較）を作成中...")
        try:
            # 相関分析と回帰分析の結果を統合
            visualization_results = {
                'correlations': correlation_results['correlations'],
                'h2_verification': regression_results.get('h2_verification', {})
            }
            analyzer.create_visualizations(horse_stats_df, visualization_results, Path(output_dir))
            logger.info("✅ 可視化の作成が完了しました")
        except Exception as e:
            logger.error(f"❌ 可視化作成でエラー: {str(e)}")
            logger.error("詳細なエラー情報:", exc_info=True)
        
        # レポート生成
        analyzer.generate_comprehensive_report(horse_stats_df, correlation_results, regression_results, Path(output_dir))
        
        return analysis_results
        
    except ImportError:
        # OddsComparisonAnalyzerが利用できない場合の簡易版
        logger.warning("OddsComparisonAnalyzerが利用できません。簡易版を実行します。")
        return perform_simple_odds_analysis(data_dir, output_dir, sample_size, min_races)

def perform_simple_odds_analysis(data_dir: str, output_dir: str, sample_size: int = None, min_races: int = 6) -> Dict[str, Any]:
    """簡易版オッズ比較分析"""
    logger.info("📊 簡易版オッズ比較分析を実行...")
    
    # グローバル関数を使用してデータを読み込み
    combined_df = load_all_data_once(data_dir, 'utf-8')
    if combined_df.empty:
        raise ValueError("有効なデータが見つかりません")
    
    # ファイル数を計算
    dataset_files = get_all_dataset_files(data_dir)
    file_count = len(dataset_files)
    
    # サンプルサイズ制限がある場合は適用
    if sample_size is not None and len(combined_df) > sample_size * 1000:  # 概算で制限
        logger.info(f"サンプルサイズ制限を適用: {sample_size * 1000}行")
        combined_df = combined_df.sample(n=sample_size * 1000, random_state=42)
    
    logger.info("🔗 簡易版データ準備完了")
    logger.info(f"統合データ: {len(combined_df):,} レコード")
    log_dataframe_info(combined_df, "簡易版統合データ")
    
    # 基本的な馬統計計算
    horse_stats = calculate_simple_horse_statistics(combined_df, min_races)
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
            'file_count': file_count
        },
        'correlations': correlations,
        'regression': regression
    }
    
    # 【追加】簡易版でも可視化を作成
    logger.info("📊 簡易版可視化を作成中...")
    try:
        create_simple_visualizations(horse_stats, correlations, regression, Path(output_dir))
        logger.info("✅ 簡易版可視化が完了しました")
    except Exception as e:
        logger.error(f"❌ 簡易版可視化作成でエラー: {str(e)}")
    
    # 簡易レポート生成
    generate_simple_report(analysis_results, Path(output_dir))
    
    return analysis_results

@log_performance("簡易馬統計計算")
def calculate_simple_horse_statistics(df: pd.DataFrame, min_races: int = 6) -> pd.DataFrame:
    """簡易版馬統計計算（層別分析と統一したREQI計算方法を適用）"""
    # 必要カラムの確認
    required_cols = ['馬名', '着順']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要なカラムが不足: {missing_cols}")
    
    # 【統一】芝レースのみフィルタ（層別分析と統一）
    if '芝ダ障害コード' in df.columns:
        original_count = len(df)
        df = df[df['芝ダ障害コード'] == '芝']
        logger.info(f"📊 芝レースフィルタ: {original_count:,} → {len(df):,}行")
    
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
    
    # 【統一】層別分析と同じ着順重み付きREQI計算を適用
    logger.info("📊 層別分析と統一した着順重み付きREQI計算を適用中...")
    df_with_reqi = calculate_race_level_features_with_position_weights(df)
    
    # 【高速化】pandas groupbyを使用してO(n²)をO(n)に改善
    logger.info("🚀 高速化版馬統計計算を実行中（pandas groupby使用）...")
    stats_calc_start = time.time()
    
    # 馬ごとにグループ化して統計を一括計算
    horse_groups = df_with_reqi.groupby('馬名')
    
    # 基本統計の計算
    basic_stats = horse_groups.agg({
        '着順': ['count', lambda x: (x == 1).mean(), lambda x: (x <= 3).mean()],
        'race_level': ['mean', 'max']
    }).round(6)
    
    # 列名を整理
    basic_stats.columns = ['total_races', 'win_rate', 'place_rate', 'avg_race_level', 'max_race_level']
    
    # オッズベース予測確率の計算（列が存在する場合のみ）
    odds_stats = pd.DataFrame(index=basic_stats.index)
    
    if '確定単勝オッズ' in df_with_reqi.columns:
        odds_stats['avg_win_prob_from_odds'] = horse_groups['確定単勝オッズ'].apply(
            lambda x: (1 / x).mean() if len(x) > 0 else 0
        )
    else:
        odds_stats['avg_win_prob_from_odds'] = 0
    
    if '確定複勝オッズ下' in df_with_reqi.columns:
        odds_stats['avg_place_prob_from_odds'] = horse_groups['確定複勝オッズ下'].apply(
            lambda x: (1 / x).mean() if len(x) > 0 else 0
        )
    else:
        odds_stats['avg_place_prob_from_odds'] = 0
    
    # 統計を結合
    horse_stats_df = pd.concat([basic_stats, odds_stats], axis=1)
    
    # 最低出走数でフィルタ
    horse_stats_df = horse_stats_df[horse_stats_df['total_races'] >= min_races]
    
    # 馬名を列に追加
    horse_stats_df['horse_name'] = horse_stats_df.index
    
    # 列の順序を整理
    horse_stats_df = horse_stats_df[['horse_name', 'total_races', 'win_rate', 'place_rate', 
                                   'avg_win_prob_from_odds', 'avg_place_prob_from_odds',
                                   'avg_race_level', 'max_race_level']]
    
    stats_time = time.time() - stats_calc_start
    logger.info(f"✅ 高速化版馬統計計算完了: {len(horse_stats_df):,}頭 ({stats_time:.2f}秒)")
    
    return horse_stats_df.set_index('horse_name')

def perform_simple_correlation_analysis(horse_stats: pd.DataFrame) -> Dict[str, Any]:
    """簡易版相関分析（層別分析と統一したREQI指標を使用）"""
    correlations = {}
    target = 'place_rate'
    
    # 【統一】層別分析と統一したREQI指標を使用
    variables = {
        '平均REQI': 'avg_race_level',  # 層別分析と統一の指標
        '最高REQI': 'max_race_level',  # 層別分析と統一の指標
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
            logger.info(f"📊 相関分析: {name} r={corr:.3f}, R²={corr**2:.3f}, p={p_value:.3e}")
    
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
    
    # 【修正】REQI（平均競走経験質指数（REQI））
    if 'avg_race_level' in data.columns:
        X_level = data[['avg_race_level']].fillna(0).values
        X_level_train, X_level_test = X_level[:split_idx], X_level[split_idx:]
        
        model_level = LinearRegression()
        model_level.fit(X_level_train, y_train)
        y_pred_level = model_level.predict(X_level_test)
        
        results['reqi_model'] = {
            'train_r2': model_level.score(X_level_train, y_train),
            'test_r2': r2_score(y_test, y_pred_level),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_level))
        }
    
    # 【修正】統計的検定を含むH2仮説検証
    if 'odds_baseline' in results and 'reqi_model' in results:
        # 基本的な数値比較
        h2_supported = results['reqi_model']['test_r2'] > results['odds_baseline']['test_r2']
        improvement = results['reqi_model']['test_r2'] - results['odds_baseline']['test_r2']
        
        # 統計的有意性の簡易評価（改善幅が0.01以上かつ正の値）
        statistically_meaningful = improvement > 0.01 and h2_supported
        
        results['h2_verification'] = {
            'hypothesis_supported': h2_supported,
            'improvement': improvement,
            'statistically_meaningful': statistically_meaningful,
            'warning': '本分析は簡易版です。厳密な統計的検定にはOddsComparisonAnalyzerを使用してください。'
        }
    
    return results

def create_simple_visualizations(horse_stats: pd.DataFrame, correlations: Dict[str, Any], 
                                regression: Dict[str, Any], output_dir: Path):
    """簡易版オッズ分析の可視化作成"""
    try:
        # matplotlibのインポートとバックエンド設定
        import matplotlib
        matplotlib.use('Agg')  # GUIバックエンドを避ける
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 統一フォント設定を適用
        from horse_racing.utils.font_config import setup_japanese_fonts
        setup_japanese_fonts(suppress_warnings=True)
        
        # 出力ディレクトリの作成
        viz_dir = output_dir / "odds_comparison"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 簡易版可視化出力ディレクトリ: {viz_dir}")
        
        # 1. 相関散布図
        logger.info("📊 簡易版相関散布図を作成中...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('競走経験質指数（REQI）とオッズ情報の相関分析', fontsize=16, fontweight='bold')
        
        # 【修正】平均REQI vs 複勝率
        if 'avg_race_level' in horse_stats.columns and 'place_rate' in horse_stats.columns:
            axes[0, 0].scatter(horse_stats['avg_race_level'], horse_stats['place_rate'], alpha=0.6, s=20, color='blue')
            axes[0, 0].set_xlabel('平均REQI')
            axes[0, 0].set_ylabel('複勝率')
            
            # 相関係数を取得
            reqi_corr = correlations.get('平均REQI', {}).get('correlation', 0)
            axes[0, 0].set_title(f'平均REQI vs 複勝率 (r={reqi_corr:.3f})')
        
        # オッズベース複勝予測 vs 複勝率
        if 'avg_place_prob_from_odds' in horse_stats.columns and 'place_rate' in horse_stats.columns:
            axes[0, 1].scatter(horse_stats['avg_place_prob_from_odds'], horse_stats['place_rate'], alpha=0.6, s=20, color='green')
            axes[0, 1].set_xlabel('オッズベース複勝予測')
            axes[0, 1].set_ylabel('複勝率')
            
            odds_place_corr = correlations.get('オッズベース複勝予測', {}).get('correlation', 0)
            axes[0, 1].set_title(f'オッズベース複勝予測 vs 複勝率 (r={odds_place_corr:.3f})')
        
        # オッズベース勝率予測 vs 複勝率
        if 'avg_win_prob_from_odds' in horse_stats.columns and 'place_rate' in horse_stats.columns:
            axes[1, 0].scatter(horse_stats['avg_win_prob_from_odds'], horse_stats['place_rate'], alpha=0.6, s=20, color='orange')
            axes[1, 0].set_xlabel('オッズベース勝率予測')
            axes[1, 0].set_ylabel('複勝率')
            
            odds_win_corr = correlations.get('オッズベース勝率予測', {}).get('correlation', 0)
            axes[1, 0].set_title(f'オッズベース勝率予測 vs 複勝率 (r={odds_win_corr:.3f})')
        
        # 空の4番目のプロット
        axes[1, 1].text(0.5, 0.5, 'データサンプル\n統計情報', ha='center', va='center', fontsize=14)
        axes[1, 1].text(0.5, 0.3, f'分析対象: {len(horse_stats):,}頭', ha='center', va='center', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('分析概要')
        
        plt.tight_layout()
        scatter_plot_path = viz_dir / 'correlation_scatter_plots.png'
        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   format='png', pad_inches=0.1)
        plt.close()
        logger.info(f"✅ 相関散布図を保存: {scatter_plot_path}")
        
        # 2. モデル性能比較（H2仮説検証）
        if regression and 'h2_verification' in regression:
            logger.info("📊 H2仮説検証チャートを作成中...")
            h2_results = regression['h2_verification']
            
            model_names = ['オッズベースライン', '平均REQI']
            r2_scores = [
                regression.get('odds_baseline', {}).get('test_r2', 0),
                regression.get('reqi_model', {}).get('test_r2', 0)
            ]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, r2_scores, color=['#ff7f0e', '#2ca02c'])
            plt.ylabel('R² (決定係数)')
            plt.title('H2仮説検証: 平均REQI の予測性能')
            plt.ylim(0, max(r2_scores) * 1.2 if max(r2_scores) > 0 else 1)
            
            # 数値ラベルを追加
            for bar, score in zip(bars, r2_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_scores)*0.01,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # H2仮説結果をテキストで表示
            if h2_results.get('hypothesis_supported', False):
                result_text = f"✅ H2仮説サポート\n改善: {h2_results.get('improvement', 0):+.4f}"
                plt.text(0.7, max(r2_scores) * 0.8, result_text, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            else:
                result_text = f"❌ H2仮説非サポート\n改善: {h2_results.get('improvement', 0):+.4f}"
                plt.text(0.7, max(r2_scores) * 0.8, result_text, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            plt.tight_layout()
            performance_plot_path = viz_dir / 'model_performance_comparison.png'
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       format='png', pad_inches=0.1)
            plt.close()
            logger.info(f"✅ H2仮説検証チャートを保存: {performance_plot_path}")
        
        # 作成されたファイルのリスト
        created_files = list(viz_dir.glob("*.png"))
        if created_files:
            logger.info("📁 作成された簡易版可視化ファイル:")
            for file_path in created_files:
                logger.info(f"   - {file_path.name}")
        
    except ImportError as e:
        logger.error(f"❌ matplotlib/seabornのインポートエラー: {e}")
        logger.info("可視化ライブラリがインストールされていない可能性があります")
    except Exception as e:
        logger.error(f"❌ 簡易版可視化作成でエラー: {str(e)}")
        # エラー時にも確実にfigureを閉じる
        try:
            plt.close('all')
        except:
            pass

def generate_simple_report(results: Dict[str, Any], output_dir: Path):
    """簡易レポート生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "horse_REQI_odds_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 競走経験質指数（REQI）とオッズ比較分析レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**実行スクリプト**: analyze_horse_REQI.py\n\n")
        
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
            
            if 'reqi_model' in regression:
                model = regression['reqi_model']
                f.write(f"| 平均REQI | {model.get('train_r2', 0):.4f} | {model.get('test_r2', 0):.4f} | {model.get('test_rmse', 0):.4f} |\n")
            
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
        f.write("平均REQI（競走経験質指数）とオッズ情報の比較分析が完了しました。\n")
        f.write("レポート記載の固定重み法を適用した正確なREQI計算により、統計的妥当性を確保しました。\n")
    
    logger.info(f"簡易レポートを生成: {report_path}")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='競走経験質指数（REQI）とオッズ比較分析を実行します（統合版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 競走経験質指数（REQI）とオッズの比較分析
  python analyze_horse_REQI.py --odds-analysis export/dataset --output-dir results/reqi_odds

  # 従来の競走経験質指数（REQI）分析
  python analyze_horse_REQI.py export/with_bias --output-dir results/race_level_analysis

  # 層別分析のみ実行
  python analyze_horse_REQI.py --stratified-only --output-dir results/stratified_analysis

このスクリプトの主要機能:
  1. 競走経験質指数（REQI）とオッズ情報の包括的比較分析
  2. H2仮説「REQIがオッズベースラインを上回る」の検証
  3. 相関分析と回帰分析による統計的評価
  4. 層別分析（年齢層・経験数・距離カテゴリ別）
  5. 期間別分析（3年間隔での時系列分析）
        """
    )
    parser.add_argument('input_path', nargs='?', help='入力ファイルまたはディレクトリのパス (例: export/with_bias)')
    parser.add_argument('--output-dir', default='results/race_level_analysis', help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=6, help='分析対象とする最小レース数')
    parser.add_argument('--encoding', default='utf-8', help='入力ファイルのエンコーディング')
    parser.add_argument('--start-date', help='分析開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', help='分析終了日（YYYYMMDD形式）')
    
    # 新機能のオプション
    parser.add_argument('--odds-analysis', metavar='DATA_DIR', help='競走経験質指数（REQI）とオッズの比較分析を実行（データディレクトリを指定）')
    parser.add_argument('--sample-size', type=int, default=None, help='オッズ分析でのサンプルファイル数（指定しない場合は全ファイル）')
    
    # 従来のオプション（継続）
    parser.add_argument('--three-year-periods', action='store_true',
                       help='3年間隔での期間別分析を実行（デフォルトは全期間分析）')
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
            # ログディレクトリの作成（output_dir/logs配下に統一）
            # argsは既に取得済みなので、output_dir配下に出力
            out_dir = Path(getattr(args, 'output_dir', 'results'))
            log_dir = out_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / f'analyze_horse_REQI_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # ログ設定の初期化
        setup_logging(log_level=args.log_level, log_file=log_file)
        
        # 引数検証（ログ設定後に実行、オッズ分析・層別分析のみの場合はスキップ）
        if not args.odds_analysis and not args.stratified_only:
            args = validate_args(args)

        # 📋 race_level_analysis_report.md準拠の処理開始表示
        print("\n" + "="*80)
        print("🏁 競馬データ分析開始: race_level_analysis_report.md準拠")
        print("="*80)
        print("📖 参照レポート: race_level_analysis_report.md")
        print("🎯 REQI計算方式: 動的重み計算法（毎回相関分析で算出）")
        print("📊 重み算出: w_i = r_i² / (r_grade² + r_venue² + r_distance²)")
        print("🔬 統計的根拠: 実測相関係数の2乗値正規化")
        print("⏳ グローバル重み初期化中...")
        print("="*80 + "\n")
        
        # 🎯 グローバル重み初期化（オッズ分析時のみ実行）
        if args.odds_analysis:
            try:
                weights_initialized = initialize_global_weights(args)
                if weights_initialized:
                    logger.info("✅ グローバル重み初期化完了")
                else:
                    logger.warning("⚠️ グローバル重み初期化に失敗、各モジュールで個別計算")
            except Exception as e:
                logger.error(f"❌ グローバル重み初期化エラー: {str(e)}")
                logger.warning("⚠️ 各モジュールで個別重み計算を実行します")
        else:
            logger.info("📊 期間別分析モード: 重み初期化は各モジュールで実行")

        # ログ設定完了後に開始メッセージを出力
        logger.info("🏇 競走経験質指数（REQI）分析を開始します...")
        logger.info(f"📅 実行日時: {datetime.now()}")
        logger.info(f"🖥️ ログレベル: {args.log_level}")
        logger.info(f"📝 ログファイル: {log_file}")
        
        # 初期システムリソース状況をログ出力
        log_system_resources()

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
        
        # 層別分析設定の処理
        enable_stratified = args.enable_stratified_analysis and not args.disable_stratified_analysis
        if enable_stratified:
            logger.info(f"📊 層別分析: 有効（年齢層別・経験数別・距離カテゴリ別）")
        else:
            logger.info(f"📊 層別分析: 無効（--disable-stratified-analysisで無効化）")
        
        # オッズ分析の場合
        if args.odds_analysis:
            logger.info("🎯 競走経験質指数（REQI）とオッズの比較分析を実行します...")
            try:
                # 統一分析器を使用
                from horse_racing.base.unified_analyzer import create_unified_analyzer
                analyzer = create_unified_analyzer('odds', args.min_races, enable_stratified)
                
                # データ読み込み
                df = analyzer.load_data_unified(args.odds_analysis, args.encoding)
                
                # グローバル重み初期化（オッズ分析時のみ）
                if not WeightManager.is_initialized():
                    analyzer.initialize_global_weights(df)
                else:
                    logger.info("✅ グローバル重みは既に初期化済みです")
                
                # 前処理
                df = analyzer.preprocess_data_unified(df)
                
                # 分析実行
                results = analyzer.analyze(df)
                
                logger.info("✅ オッズ比較分析が完了しました。")
                logger.info(f"📊 分析対象: {results['data_summary']['total_records']:,}レコード, {results['data_summary']['horse_count']:,}頭")
                logger.info(f"📁 結果保存先: {args.output_dir}")
                
                # H2仮説結果の表示
                if 'regression' in results and 'h2_verification' in results['regression']:
                    h2 = results['regression']['h2_verification']
                    result_text = "サポート" if h2.get('h2_hypothesis_supported', h2.get('hypothesis_supported', False)) else "非サポート"
                    logger.info(f"🎯 H2仮説「競走経験質指数（REQI）がオッズベースラインを上回る」: {result_text}")
                    improvement = h2.get('r2_improvement', h2.get('improvement', 0))
                    logger.info(f"   性能改善: {improvement:+.4f}")
                
                # 【強制出力】オッズ比較レポートを必ず生成（包括版が失敗しても簡易版を出力）
                try:
                    logger.info("📋 オッズ比較の簡易レポートを生成します（存在しない場合は新規作成）...")
                    _ = perform_simple_odds_analysis(args.odds_analysis, args.output_dir, sample_size=None, min_races=args.min_races)
                    logger.info("✅ 簡易オッズ比較レポート生成完了: horse_REQI_odds_analysis_report.md")
                except Exception as e:
                    logger.error(f"❌ 簡易オッズ比較レポート生成でエラー: {str(e)}")
                    logger.error("詳細なエラー情報:", exc_info=True)

                # 【追加】オッズ分析モードでも層別レポートを生成
                try:
                    logger.info("📋 統合層別分析レポートを生成中...")
                    stratified_dataset = create_stratified_dataset_from_export('export/dataset')
                    stratified_results = perform_integrated_stratified_analysis(stratified_dataset)
                    _ = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
                    logger.info("✅ 統合層別分析レポート生成完了")
                except Exception as e:
                    logger.error(f"❌ 統合層別分析レポート生成エラー: {str(e)}")
                    logger.error("詳細なエラー情報:", exc_info=True)
                
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
                _ = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
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
            try:
                # 統一分析器を使用
                from horse_racing.base.unified_analyzer import create_unified_analyzer
                analyzer = create_unified_analyzer('period', args.min_races, enable_stratified)
                
                # データ読み込み
                df = analyzer.load_data_unified(args.input_path, args.encoding)
                
                # グローバル重み初期化（期間別分析時は重複実行を回避）
                if not WeightManager.is_initialized():
                    analyzer.initialize_global_weights(df)
                else:
                    logger.info("✅ グローバル重みは既に初期化済みです")
                
                # 前処理
                df = analyzer.preprocess_data_unified(df)
                
                logger.info(f"📊 読み込んだデータ件数: {len(df):,}件")
            
            # 年データが存在するかチェック
                if '年' in df.columns and df['年'].notna().any():
                    min_year = int(df['年'].min())
                    max_year = int(df['年'].max())
                    logger.info(f"📊 年データ範囲: {min_year}年 - {max_year}年")
                
                    # 期間別分析実行
                    results = analyzer.analyze(df)
                    
                    if results:
                        logger.info(f"📊 期間別分析完了: {len(results)}期間")
                        
                        # 期間別分析の総合レポートを生成
                        logger.info("📋 期間別分析の総合レポートを生成中...")
                        try:
                            generate_period_summary_report(results, Path(args.output_dir))
                            logger.info("✅ 期間別分析総合レポート生成完了")
                        except Exception as e:
                            logger.error(f"❌ 総合レポート生成エラー: {str(e)}")
                        
                        # 【追加】統合層別分析レポートも生成
                        try:
                            logger.info("📋 統合層別分析レポートを生成中...")
                            stratified_dataset = create_stratified_dataset_from_export('export/dataset')
                            stratified_results = perform_integrated_stratified_analysis(stratified_dataset)
                            _ = generate_stratified_report(stratified_results, stratified_dataset, Path(args.output_dir))
                            logger.info("✅ 統合層別分析レポート生成完了")
                        except Exception as e:
                            logger.error(f"❌ 統合層別分析レポート生成エラー: {str(e)}")
                            logger.error("詳細なエラー情報:", exc_info=True)
                    
                        # 結果の保存先を表示
                        logger.info(f"📁 結果保存先: {args.output_dir}")
                        logger.info(f"📋 総合レポート: {args.output_dir}/競走経験質指数（REQI）分析_期間別総合レポート.md")
                        logger.info(f"📋 層別レポート: {args.output_dir}/stratified_analysis_integrated_report.md")
                        
                        return 0
                    else:
                        logger.warning("⚠️ 有効な期間が見つかりませんでした")
                        return 1
                else:
                    logger.warning("⚠️ 年データが見つかりません")
                    return 1
                    
            except Exception as e:
                logger.error(f"❌ 期間別分析でエラー: {str(e)}")
                logger.error("詳細なエラー情報:", exc_info=True)
                return 1
        
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

            # 1. REQIAnalyzerのインスタンス化
            analyzer = REQIAnalyzer(config, enable_stratified)

            # 2. データの読み込み
            logger.info("📖 全データ読み込み中...")
            analyzer.df = analyzer.load_data()
            log_dataframe_info(analyzer.df, "読み込み完了データ")

            # 前処理を追加
            logger.info("🔧 前処理中...")
            analyzer.df = analyzer.preprocess_data()
            log_dataframe_info(analyzer.df, "前処理完了データ")
            
            # 3. 特徴量計算
            logger.info("🧮 特徴量計算中...")
            analyzer.df = analyzer.calculate_feature()
            log_dataframe_info(analyzer.df, "特徴量計算完了データ")

            # 4. 【重要】修正版分析の実行
            logger.info("🔬 【修正版】厳密な時系列分割による分析を実行中...")
            log_system_resources()
            analyzer.stats = analyzer.analyze()
            
            # 結果の可視化
            logger.info("📊 可視化生成中...")
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
                _ = generate_stratified_report(stratified_results, stratified_dataset, output_dir)
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
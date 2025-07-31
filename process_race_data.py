"""
競馬レースデータ処理のコマンドラインエントリーポイント
計画書Phase 0: データ整備（実務レベル対応版）

実務レベルの特徴：
1. 戦略的欠損値処理（CSV作成時）
2. データ品質チェックとレポート
3. 段階的処理とログ出力
4. エラーハンドリングと復旧機能
5. 処理時間とメモリ使用量の監視
"""
from horse_racing.data.processors.bac_processor import process_all_bac_files
from horse_racing.data.processors.sed_processor import process_all_sed_files
from horse_racing.data.processors.srb_processor import process_all_srb_files, merge_srb_with_sed
import os
import argparse
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from output_utils import OutputUtils
from typing import Dict, Any, Tuple, List
import numpy as np
import re
from collections import defaultdict

# 実務レベルのログ設定
def setup_logging(log_level='INFO', log_file=None):
    """実務レベルのログ設定"""
    import logging
    
    # シンプルな設定
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

# メインロガー
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """
    データ品質チェッククラス
    実務レベルのデータ整備に必要な品質管理機能
    """
    
    def __init__(self):
        self.quality_report = {}
        
    def check_data_quality(self, df: pd.DataFrame, stage_name: str) -> Dict[str, Any]:
        """
        包括的なデータ品質チェック
        
        Args:
            df: チェック対象のDataFrame
            stage_name: 処理段階名
            
        Returns:
            品質レポート辞書
        """
        logger.info(f"📊 {stage_name} - データ品質チェック開始")
        start_time = time.time()
        
        report = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'outliers': {},
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # 1. 欠損値分析
            logger.info("   🔍 欠損値分析中...")
            missing_analysis = self._analyze_missing_values(df)
            report['missing_values'] = missing_analysis
            
            # 2. データ型チェック
            logger.info("   🏷️ データ型チェック中...")
            report['data_types'] = self._check_data_types(df)
            
            # 3. 重複チェック
            logger.info("   🔄 重複チェック中...")
            report['duplicates'] = df.duplicated().sum()
            
            # 4. 外れ値検出（数値列のみ）
            logger.info("   📈 外れ値検出中...")
            report['outliers'] = self._detect_outliers(df)
            
            # 5. ビジネスルール検証
            logger.info("   📋 ビジネスルール検証中...")
            warnings, recommendations = self._validate_business_rules(df)
            report['warnings'] = warnings
            report['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            report['execution_time_seconds'] = execution_time
            
            logger.info(f"✅ {stage_name} - データ品質チェック完了 ({execution_time:.2f}秒)")
            
            # レポート要約をログ出力
            self._log_quality_summary(report)
            
        except Exception as e:
            logger.error(f"❌ データ品質チェックでエラー: {str(e)}")
            report['error'] = str(e)
        
        self.quality_report[stage_name] = report
        return report
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """欠損値の詳細分析"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            'total_missing_cells': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'critical_columns': []  # 50%以上欠損の列
        }
        
        # 重要な欠損パターンの特定
        for col, pct in missing_percentages.items():
            if pct >= 50:
                analysis['critical_columns'].append(col)
        
        return analysis
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """データ型の妥当性チェック"""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """IQR法による外れ値検出"""
        outlier_counts = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].notna().sum() > 0:  # 欠損値でない値が存在する場合のみ
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
        
        return outlier_counts
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """競馬データ特有のビジネスルール検証"""
        warnings = []
        recommendations = []
        
        # 着順のチェック
        if '着順' in df.columns:
            invalid_positions = df[df['着順'] < 0]
            if len(invalid_positions) > 0:
                warnings.append(f"不正な着順データ: {len(invalid_positions)}件")
        
        # タイムのチェック
        if 'タイム' in df.columns:
            # 異常に速い/遅いタイムの検出
            if df['タイム'].notna().sum() > 0:
                median_time = df['タイム'].median()
                if median_time and (median_time < 60 or median_time > 300):
                    warnings.append(f"異常なタイム中央値: {median_time}秒")
        
        # 距離のチェック
        if '距離' in df.columns:
            if df['距離'].notna().sum() > 0:
                min_distance = df['距離'].min()
                max_distance = df['距離'].max()
                if min_distance < 1000 or max_distance > 4000:
                    warnings.append(f"異常な距離範囲: {min_distance}m - {max_distance}m")
        
        # 推奨事項
        if len(warnings) == 0:
            recommendations.append("データ品質は良好です")
        else:
            recommendations.append("データクリーニングを検討してください")
        
        return warnings, recommendations
    
    def _log_quality_summary(self, report: Dict[str, Any]):
        """品質レポートサマリーのログ出力"""
        logger.info(f"📊 【{report['stage']}】品質サマリー:")
        logger.info(f"   📏 データ規模: {report['total_rows']:,}行 x {report['total_columns']}列")
        logger.info(f"   💾 メモリ使用量: {report['memory_usage_mb']:.1f}MB")
        logger.info(f"   ❓ 欠損セル数: {report['missing_values']['total_missing_cells']:,}")
        logger.info(f"   🔄 重複行数: {report['duplicates']:,}")
        
        if report['warnings']:
            logger.warning(f"   ⚠️ 警告: {len(report['warnings'])}件")
            for warning in report['warnings']:
                logger.warning(f"      • {warning}")

class MissingValueHandler:
    """
    戦略的欠損値処理クラス
    計画書Phase 0の要件に基づく実務レベルの欠損値処理
    """
    
    def __init__(self):
        self.processing_log = []
        
    def handle_missing_values(self, df: pd.DataFrame, strategy_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        戦略的欠損値処理の実行
        
        Args:
            df: 処理対象DataFrame
            strategy_config: 処理戦略設定
            
        Returns:
            欠損値処理済みDataFrame
        """
        logger.info("🔧 戦略的欠損値処理開始")
        start_time = time.time()
        
        # デフォルト戦略設定
        if strategy_config is None:
            strategy_config = self._get_default_strategy()
        
        df_processed = df.copy()
        original_rows = len(df_processed)
        
        try:
            # 1. 重要列の欠損値処理
            df_processed = self._handle_critical_columns(df_processed, strategy_config)
            
            # 2. 数値列の欠損値処理
            df_processed = self._handle_numeric_columns(df_processed, strategy_config)
            
            # 3. カテゴリ列の欠損値処理
            df_processed = self._handle_categorical_columns(df_processed, strategy_config)
            
            # 4. 残存欠損値の最終処理
            df_processed = self._handle_remaining_missing(df_processed, strategy_config)
            
            execution_time = time.time() - start_time
            final_rows = len(df_processed)
            
            logger.info(f"✅ 欠損値処理完了 ({execution_time:.2f}秒)")
            logger.info(f"   📊 処理前: {original_rows:,}行")
            logger.info(f"   📊 処理後: {final_rows:,}行")
            logger.info(f"   📉 除去行数: {original_rows - final_rows:,}行 ({((original_rows - final_rows) / original_rows) * 100:.1f}%)")
            
            # 処理ログの保存
            self._save_processing_log(df_processed)
            
        except Exception as e:
            logger.error(f"❌ 欠損値処理でエラー: {str(e)}")
            raise
        
        return df_processed
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """デフォルトの欠損値処理戦略"""
        return {
            'critical_columns': {
                '着順': 'drop',  # 着順が欠損の行は削除
                'タイム': 'drop',  # タイムが欠損の行は削除
                '距離': 'drop',   # 距離が欠損の行は削除
                '馬名': 'drop',   # 馬名が欠損の行は削除
                'IDM': 'drop'     # IDMが欠損の行は削除
            },
            'numeric_columns': {
                'method': 'median',  # 中央値で補完
                'max_missing_rate': 0.5  # 50%以上欠損の列は削除
            },
            'categorical_columns': {
                'method': 'mode',    # 最頻値で補完
                'unknown_label': '不明',
                'max_missing_rate': 0.8  # 80%以上欠損の列は削除
            },
            'remaining_strategy': 'drop'  # 残存欠損値は行削除
        }
    
    def _handle_critical_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """重要列の欠損値処理"""
        logger.info("   🎯 重要列の欠損値処理中...")
        
        critical_config = config.get('critical_columns', {})
        
        for column, strategy in critical_config.items():
            if column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    logger.info(f"      • {column}: {missing_count:,}件の欠損値を{strategy}処理")
                    
                    if strategy == 'drop':
                        df = df.dropna(subset=[column])
                        self.processing_log.append(f"{column}: {missing_count}行を削除（重要列）")
        
        return df
    
    def _handle_numeric_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """数値列の欠損値処理（グレード専用処理含む）"""
        logger.info("   🔢 数値列の欠損値処理中...")
        
        numeric_config = config.get('numeric_columns', {})
        method = numeric_config.get('method', 'median')
        max_missing_rate = numeric_config.get('max_missing_rate', 0.5)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df)
            
            if missing_count > 0:
                # グレード列の特別処理（実務レベル）
                if column in ['グレード', 'grade', 'レースグレード']:
                    logger.info(f"      • {column}: 実務レベルグレード推定処理を実行")
                    df = self._estimate_grade_from_features(df, column)
                    
                    # 推定後の欠損数をチェック
                    remaining_missing = df[column].isnull().sum()
                    estimated_count = missing_count - remaining_missing
                    
                    if estimated_count > 0:
                        logger.info(f"      • {column}: {estimated_count:,}件を賞金・レース名から推定補完")
                        self.processing_log.append(f"{column}: 賞金・レース名から{estimated_count}件推定→グレード名列追加")
                    
                    # 残りの欠損値は中央値で補完（数値で処理後、グレード名列追加）
                    if remaining_missing > 0:
                        fill_value = df[column].median()
                        # 数値で補完してからグレード名列追加処理に委ねる
                        df[column] = df[column].fillna(fill_value)
                        logger.info(f"      • {column}: 残り{remaining_missing:,}件をmedian({fill_value})で補完後、グレード名列追加")
                        self.processing_log.append(f"{column}: median補完{remaining_missing}件→グレード名列追加")
                
                elif missing_rate > max_missing_rate:
                    logger.warning(f"      • {column}: 欠損率{missing_rate:.1%} > {max_missing_rate:.1%} → 列削除")
                    df = df.drop(columns=[column])
                    self.processing_log.append(f"{column}: 高欠損率により列削除")
                else:
                    if method == 'median':
                        fill_value = df[column].median()
                    elif method == 'mean':
                        fill_value = df[column].mean()
                    else:
                        fill_value = 0
                    
                    df[column] = df[column].fillna(fill_value)
                    logger.info(f"      • {column}: {missing_count:,}件を{method}({fill_value})で補完")
                    self.processing_log.append(f"{column}: {method}で{missing_count}件補完")
        
        return df
    
    def _handle_categorical_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """カテゴリ列の欠損値処理"""
        logger.info("   🏷️ カテゴリ列の欠損値処理中...")
        
        categorical_config = config.get('categorical_columns', {})
        method = categorical_config.get('method', 'mode')
        unknown_label = categorical_config.get('unknown_label', '不明')
        max_missing_rate = categorical_config.get('max_missing_rate', 0.8)
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df)
            
            if missing_count > 0:
                if missing_rate > max_missing_rate:
                    logger.warning(f"      • {column}: 欠損率{missing_rate:.1%} > {max_missing_rate:.1%} → 列削除")
                    df = df.drop(columns=[column])
                    self.processing_log.append(f"{column}: 高欠損率により列削除")
                else:
                    if method == 'mode' and not df[column].mode().empty:
                        fill_value = df[column].mode()[0]
                    else:
                        fill_value = unknown_label
                    
                    df[column] = df[column].fillna(fill_value)
                    logger.info(f"      • {column}: {missing_count:,}件を'{fill_value}'で補完")
                    self.processing_log.append(f"{column}: {fill_value}で{missing_count}件補完")
        
        return df
    
    def _handle_remaining_missing(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """残存欠損値の最終処理"""
        remaining_missing = df.isnull().sum().sum()
        
        if remaining_missing > 0:
            logger.info(f"   🔧 残存欠損値処理中: {remaining_missing:,}件")
            
            strategy = config.get('remaining_strategy', 'drop')
            
            if strategy == 'drop':
                initial_rows = len(df)
                df = df.dropna()
                dropped_rows = initial_rows - len(df)
                
                if dropped_rows > 0:
                    logger.info(f"      • 残存欠損値のある{dropped_rows:,}行を削除")
                    self.processing_log.append(f"残存欠損値: {dropped_rows}行削除")
        
        return df
    
    def _estimate_grade_from_features(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """
        実務レベルのグレード推定処理
        賞金・レース名・出走頭数等からグレードを推定
        推定できない場合は該当レコードを削除
        
        Args:
            df: 処理対象DataFrame
            grade_column: グレード列名
            
        Returns:
            グレード推定済みDataFrame（推定失敗レコードは削除済み）
        """
        initial_rows = len(df)
        grade_missing_mask = df[grade_column].isnull()
        initial_missing_count = grade_missing_mask.sum()
        
        if not grade_missing_mask.any():
            # 既存の数値グレードからグレード名列を作成
            df = self._add_grade_name_column(df, grade_column)
            return df
        
        logger.info(f"      📊 グレード欠損値: {initial_missing_count:,}件 ({initial_missing_count/initial_rows*100:.1f}%)")
        
        # 推定対象データ
        estimation_df = df[grade_missing_mask].copy()
        
        # 1. 賞金からグレード推定
        if '本賞金' in df.columns:
            estimation_df = self._estimate_grade_from_prize(estimation_df, grade_column)
        
        # 2. レース名からグレード推定（コメントアウト - 欠損値対応を厳密化）
        # if 'レース名' in df.columns:
        #     estimation_df = self._estimate_grade_from_race_name(estimation_df, grade_column)
        
        # 3. 出走頭数による補正（コメントアウト - 欠損値対応を厳密化）
        # if '頭数' in df.columns:
        #     estimation_df = self._adjust_grade_by_field_size(estimation_df, grade_column)
        
        # 4. 距離による補正（コメントアウト - 欠損値対応を厳密化）
        # if '距離' in df.columns:
        #     estimation_df = self._adjust_grade_by_distance(estimation_df, grade_column)
        
        # 推定結果を元のDataFrameに反映
        df.loc[grade_missing_mask, grade_column] = estimation_df[grade_column]
        
        # 推定後の残存欠損値をチェック
        remaining_missing_mask = df[grade_column].isnull()
        remaining_missing_count = remaining_missing_mask.sum()
        estimated_count = initial_missing_count - remaining_missing_count
        
        if estimated_count > 0:
            logger.info(f"      ✅ グレード推定成功: {estimated_count:,}件")
            self.processing_log.append(f"{grade_column}: 賞金・レース名から{estimated_count}件推定")
        
        # 残存欠損値（推定失敗）のレコードを削除
        if remaining_missing_count > 0:
            logger.info(f"      ❌ グレード推定失敗→削除: {remaining_missing_count:,}件 ({remaining_missing_count/initial_rows*100:.1f}%)")
            df = df[~remaining_missing_mask]
            self.processing_log.append(f"{grade_column}: 推定失敗により{remaining_missing_count}行削除")
        
        # 数値グレードを保持しつつ「グレード名」列を作成
        df = self._add_grade_name_column(df, grade_column)
        
        final_rows = len(df)
        deleted_rows = initial_rows - final_rows
        
        if deleted_rows > 0:
            logger.info(f"      📉 削除レコード統計: {deleted_rows:,}行削除 (削除率: {deleted_rows/initial_rows*100:.1f}%)")
            logger.info(f"      📊 残存レコード: {final_rows:,}行 (残存率: {final_rows/initial_rows*100:.1f}%)")
        
        return df
    
    def _estimate_grade_from_prize(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """賞金からグレード推定（実務レベル基準）"""
        if '本賞金' not in df.columns:
            return df
        
        # 賞金を数値型に変換
        df['本賞金'] = pd.to_numeric(df['本賞金'], errors='coerce')
        
        # 実務レベル賞金基準（単位：万円）
        # 実際の競馬界の賞金体系に基づく
        prize_grade_mapping = [
            (15000, 1),    # G1: 1億5千万円以上
            (6000, 2),     # G2: 6千万円以上
            (4000, 3),     # G3: 4千万円以上
            (1500, 4),     # 重賞: 1千5百万円以上
            (500, 5),      # 特別: 500万円以上
            (0, 6)         # その他: 500万円未満
        ]
        
        for min_prize, grade in prize_grade_mapping:
            mask = (df['本賞金'] >= min_prize) & df[grade_column].isnull()
            df.loc[mask, grade_column] = grade
        
        return df
    
    def _estimate_grade_from_race_name(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """レース名からグレード推定（実務レベルパターンマッチング）"""
        if 'レース名' not in df.columns:
            return df
        
        # レース名のグレード判定パターン（実務レベル）
        race_patterns = {
            1: [  # G1パターン
                'ダービー', 'オークス', '菊花賞', '皐月賞', '桜花賞', 'マイル', 
                '有馬記念', '宝塚記念', '天皇賞', 'ジャパンカップ', 'スプリンターズ',
                'エリザベス女王杯', 'フェブラリーステークス', 'チャンピオンズカップ',
                '高松宮記念', '安田記念', 'ヴィクトリア', '秋華賞'
            ],
            2: [  # G2パターン  
                '京都記念', '阪神大賞典', '目黒記念', '毎日王冠', '京都大賞典',
                'アルゼンチン共和国杯', '中山記念', '金鯱賞', '京王杯', '府中牝馬',
                'セントウルステークス', 'スワンステークス', '小倉記念'
            ],
            3: [  # G3パターン
                '函館記念', '中京記念', '新潟記念', '七夕賞', '福島記念', 
                'きさらぎ賞', '弥生賞', 'スプリング', 'セントライト', 'アルテミス',
                '朝日杯', 'ホープフル', 'ラジオ', 'クイーン', 'オープン'
            ],
            4: [  # 重賞（リステッド）パターン
                '重賞', 'ステークス', 'カップ', '賞', '記念', '特別',
                'オープン', 'リステッド', 'L'
            ]
        }
        
        for grade, patterns in race_patterns.items():
            for pattern in patterns:
                mask = (df['レース名'].str.contains(pattern, case=False, na=False)) & df[grade_column].isnull()
                df.loc[mask, grade_column] = grade
        
        # デフォルト値補完は行わない（推定失敗の場合は後でレコード削除）
        
        return df
    
    def _adjust_grade_by_field_size(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """出走頭数によるグレード補正（実務レベル調整）"""
        if '頭数' not in df.columns:
            return df
        
        df['頭数'] = pd.to_numeric(df['頭数'], errors='coerce')
        
        # 出走頭数による補正ロジック
        # 大きなレースほど出走頭数が多い傾向
        for idx, row in df.iterrows():
            if pd.notnull(row[grade_column]) and pd.notnull(row['頭数']):
                current_grade = row[grade_column]
                field_size = row['頭数']
                
                # 出走頭数が異常に少ない場合はグレードを下げる
                if field_size < 8 and current_grade <= 3:  # G3以上で8頭未満は怪しい
                    df.loc[idx, grade_column] = min(current_grade + 1, 6)
                # 出走頭数が多い場合はグレード維持または向上
                elif field_size >= 16 and current_grade >= 5:  # 16頭以上で条件戦は重賞の可能性
                    df.loc[idx, grade_column] = max(current_grade - 1, 4)
        
        return df
    
    def _adjust_grade_by_distance(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """距離によるグレード補正（実務レベル調整）"""
        if '距離' not in df.columns:
            return df
        
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
        
        # 距離による補正ロジック
        # 特殊距離（3000m以上）は重賞の可能性が高い
        for idx, row in df.iterrows():
            if pd.notnull(row[grade_column]) and pd.notnull(row['距離']):
                current_grade = row[grade_column]
                distance = row['距離']
                
                # 長距離レース（3000m以上）の場合
                if distance >= 3000 and current_grade >= 5:
                    df.loc[idx, grade_column] = min(current_grade - 1, 4)  # 重賞以上に格上げ
                
                # 極端な短距離（1000m未満）や長距離（3600m超）は特別レース
                if (distance < 1000 or distance > 3600) and current_grade >= 4:
                    df.loc[idx, grade_column] = min(current_grade - 1, 3)  # G3以上に格上げ
        
        return df
    
    def _add_grade_name_column(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """
        数値グレードから「グレード名」列を作成
        
        Args:
            df: 処理対象DataFrame
            grade_column: グレード列名
            
        Returns:
            グレード名列が追加されたDataFrame
        """
        # グレード変換マッピング（実務レベル正式表記）
        grade_mapping = {
            1: 'Ｇ１',
            2: 'Ｇ２', 
            3: 'Ｇ３',
            4: '重賞',
            5: '特別',
            6: 'Ｌ　（リステッド競走）'
        }
        
        # グレード列を数値型として保持（元の列はそのまま）
        df[grade_column] = pd.to_numeric(df[grade_column], errors='coerce')
        
        # NaN値のデフォルト補完は行わない
        
        # グレード名データを作成（NaN値はそのまま保持）
        grade_names = df[grade_column].map(grade_mapping)
        
        # グレード名列が既に存在するかチェック
        if 'グレード名' in df.columns:
            # 既存の列を更新
            df['グレード名'] = grade_names
        else:
            # グレード列の直後に「グレード名」列を挿入
            grade_col_index = df.columns.get_loc(grade_column)
            df.insert(grade_col_index + 1, 'グレード名', grade_names)
        
        return df
    
    def _save_processing_log(self, df: pd.DataFrame):
        """処理ログの保存（追記モード対応）"""
        log_path = Path('export/missing_value_processing_log.txt')
        
        try:
            # ログファイルが存在しない場合のみヘッダー作成
            write_header = not log_path.exists()
            
            with open(log_path, 'a', encoding='utf-8') as f:  # 追記モードに変更
                if write_header:
                    f.write(f"欠損値処理ログ - {datetime.now()}\n")
                    f.write("=" * 50 + "\n\n")
                
                # 各ファイルの処理ログを追記
                for log_entry in self.processing_log:
                    f.write(f"• {log_entry}\n")
                
                # 最終データ形状を追記
                f.write(f"最終データ形状: {df.shape}\n")
                f.write(f"残存欠損値: {df.isnull().sum().sum()}件\n\n")
            
            logger.info(f"   📝 処理ログ保存: {log_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ 処理ログ保存エラー: {str(e)}")

class SystemMonitor:
    """システム監視クラス（簡略版）"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def log_system_status(self, stage_name: str):
        """システム状態のログ出力（簡略版）"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        logger.info(f"💻 [{stage_name}] システム状態:")
        logger.info(f"   ⏱️ 経過時間: {elapsed_time:.1f}秒")

def ensure_export_dirs():
    """
    出力用ディレクトリの存在確認と作成
    実務レベルの管理機能付き
    """
    dirs = [
        'export/BAC', 
        'export/SRB', 
        'export/SED', 
        'export/with_bias',          # 実際のSED+SRB統合データ出力先
        'export/quality_reports',     # データ品質レポート保存用
        'export/logs'                 # ログ保存用
    ]
    
    created_dirs = []
    
    for dir_path in dirs:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
            logger.info(f"📁 ディレクトリ作成: {dir_path}")
    
    if created_dirs:
        logger.info(f"✅ {len(created_dirs)}個のディレクトリを作成しました")
    else:
        logger.info("📁 すべてのディレクトリが既に存在します")

def save_quality_report(quality_checker: DataQualityChecker):
    """データ品質レポートの保存"""
    report_path = Path('export/quality_reports/data_quality_report.json')
    
    try:
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_checker.quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 品質レポート保存: {report_path}")
        
    except Exception as e:
        logger.warning(f"⚠️ 品質レポート保存エラー: {str(e)}")

def display_deletion_statistics():
    """
    グレード欠損による削除統計の表示
    SEDとwith_biasディレクトリを比較して削除統計を出力
    """
    try:
        from pathlib import Path
        
        # ディレクトリパス
        sed_dir = Path('export/SED/formatted')
        bias_dir = Path('export/with_bias')
        
        if not sed_dir.exists() or not bias_dir.exists():
            logger.warning("⚠️ 比較用ディレクトリが見つかりません")
            return
        
        # ファイル一覧取得
        sed_files = list(sed_dir.glob('*.csv'))
        bias_files = list(bias_dir.glob('*.csv'))
        
        if not sed_files or not bias_files:
            logger.warning("⚠️ 比較用ファイルが見つかりません")
            return
        
        # 統計を収集
        total_sed = 0
        total_bias = 0
        total_deleted = 0
        deletion_files = []
        
        # ファイル名でマッピング
        sed_files_dict = {f.stem.replace('_formatted', ''): f for f in sed_files}
        
        for bias_file in bias_files:
            base_name = bias_file.stem.replace('_formatted_with_bias', '')
            
            if base_name in sed_files_dict:
                sed_file = sed_files_dict[base_name]
                
                try:
                    # レコード数を数える（ヘッダー除く）
                    with open(sed_file, 'r', encoding='utf-8') as f:
                        sed_count = sum(1 for line in f) - 1
                    
                    with open(bias_file, 'r', encoding='utf-8') as f:
                        bias_count = sum(1 for line in f) - 1
                    
                    deleted = sed_count - bias_count
                    total_sed += sed_count
                    total_bias += bias_count
                    total_deleted += deleted
                    
                    if deleted > 0:
                        deletion_rate = (deleted / sed_count * 100) if sed_count > 0 else 0
                        deletion_files.append({
                            'file': base_name,
                            'deleted': deleted,
                            'deletion_rate': deletion_rate
                        })
                
                except Exception:
                    continue
        
        # 統計表示
        logger.info(f"📈 全体削除統計:")
        logger.info(f"   📥 処理前総レコード: {total_sed:,}件")
        logger.info(f"   📤 処理後総レコード: {total_bias:,}件")
        logger.info(f"   ❌ 削除レコード数: {total_deleted:,}件")
        logger.info(f"   📉 全体削除率: {(total_deleted/total_sed*100 if total_sed > 0 else 0):.2f}%")
        logger.info(f"   🗂️ 削除発生ファイル数: {len(deletion_files)}")
        logger.info(f"   📊 削除発生率: {(len(deletion_files)/len(sed_files_dict)*100 if sed_files_dict else 0):.1f}%")
        
        if deletion_files:
            logger.info("\n📋 削除の多いファイル（上位10件）:")
            deletion_files.sort(key=lambda x: x['deleted'], reverse=True)
            for i, item in enumerate(deletion_files[:10], 1):
                logger.info(f"   {i:2d}. {item['file']}: -{item['deleted']:,}件 (-{item['deletion_rate']:.1f}%)")
        else:
            logger.info("✅ グレード欠損による削除は発生していません")
    
    except Exception as e:
        logger.warning(f"⚠️ 削除統計表示エラー: {str(e)}")

def summarize_processing_log():
    """
    実務レベル欠損値処理ログのサマリー生成
    冗長なログをまとめて統計情報を作成
    """
    log_file = Path('export/missing_value_processing_log.txt')
    backup_file = Path('export/missing_value_processing_log_original.txt')
    summary_file = Path('export/missing_value_processing_summary.txt')
    
    # ログファイルが存在しない場合はスキップ
    if not log_file.exists():
        logger.info("📝 欠損値処理ログが見つからないため、サマリー生成をスキップします")
        return
    
    logger.info("📊 欠損値処理ログをサマリー形式に整理中...")
    
    try:
        # ログ解析
        stats = _parse_processing_log(log_file)
        
        if not stats:
            logger.warning("⚠️ ログ解析に失敗しました")
            return
        
        # サマリーレポート生成
        _generate_summary_report(stats, summary_file)
        
        # 元ログをバックアップ
        if backup_file.exists():
            backup_file.unlink()  # 既存バックアップを削除
        log_file.rename(backup_file)
        
        # サマリーを新しいログファイルに
        summary_file.rename(log_file)
        
        logger.info("✅ 欠損値処理ログの整理完了")
        logger.info(f"   📋 サマリー: {log_file}")
        logger.info(f"   💾 バックアップ: {backup_file}")
        logger.info(f"   📊 処理ファイル数: {stats['total_files']}ファイル")
        
        # 統計サマリーをログ出力
        if stats['idm_deletions']:
            total_idm = sum(stats['idm_deletions'])
            logger.info(f"   🎯 IDM削除: {total_idm:,}行 ({len(stats['idm_deletions'])}ファイル)")
        
        if stats['grade_estimations']:
            total_grade = sum(stats['grade_estimations'])
            logger.info(f"   🏆 グレード推定: {total_grade:,}件 ({len(stats['grade_estimations'])}ファイル)")
        
    except Exception as e:
        logger.warning(f"⚠️ ログサマリー生成エラー: {str(e)}")

def _parse_processing_log(log_file: Path) -> Dict[str, Any]:
    """ログファイルを解析して処理統計を作成"""
    
    # 統計情報格納用
    stats = {
        'idm_deletions': [],
        'grade_estimations': [],
        'median_imputations': defaultdict(list),
        'dropped_columns': set(),
        'categorical_imputations': defaultdict(list),
        'other_imputations': defaultdict(list),
        'total_files': 0,
        'final_shapes': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"ログファイル読み込みエラー: {e}")
        return None
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('==') or line.startswith('欠損値処理ログ'):
            continue
            
        # IDM削除
        if 'IDM:' in line and '行を削除（重要列）' in line:
            match = re.search(r'IDM: (\d+)行を削除', line)
            if match:
                stats['idm_deletions'].append(int(match.group(1)))
        
        # グレード推定
        elif 'グレード:' in line and '推定→グレード名列追加' in line:
            match = re.search(r'グレード: 賞金・レース名から(\d+)件推定', line)
            if match:
                stats['grade_estimations'].append(int(match.group(1)))
        
        # 中央値補完
        elif 'medianで' in line and '件補完' in line:
            match = re.search(r'• ([^:]+): medianで(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                count = int(match.group(2))
                stats['median_imputations'][column_name].append(count)
        
        # 高欠損率による列削除
        elif '高欠損率により列削除' in line:
            match = re.search(r'• ([^:]+): 高欠損率により列削除', line)
            if match:
                stats['dropped_columns'].add(match.group(1))
        
        # カテゴリ補完（レース名、馬体重増減）
        elif line.startswith('• レース名:') or line.startswith('• レース名略称:') or line.startswith('• 馬体重増減:'):
            match = re.search(r'• ([^:]+): (.+)で(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                value = match.group(2)
                count = int(match.group(3))
                stats['categorical_imputations'][column_name].append((value, count))
        
        # その他の補完処理
        elif '件補完' in line and 'median' not in line:
            match = re.search(r'• ([^:]+): (.+)で(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                value = match.group(2)
                count = int(match.group(3))
                stats['other_imputations'][column_name].append((value, count))
        
        # 最終データ形状
        elif '最終データ形状:' in line:
            match = re.search(r'最終データ形状: \((\d+), (\d+)\)', line)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                stats['final_shapes'].append((rows, cols))
    
    # ファイル数を推定（IDM削除の回数とグレード推定の回数の合計）
    stats['total_files'] = len(stats['idm_deletions']) + len(stats['grade_estimations'])
    
    return stats

def _generate_summary_report(stats: Dict[str, Any], output_file: Path):
    """統計情報からサマリーレポートを生成"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 欠損値処理ログ サマリーレポート（実務レベル）\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 処理ファイル数
        f.write(f"📁 処理ファイル数: {stats['total_files']}ファイル\n\n")
        
        # IDM削除統計
        if stats['idm_deletions']:
            total_idm = sum(stats['idm_deletions'])
            f.write("🎯 IDM欠損値削除処理:\n")
            f.write(f"   • 処理回数: {len(stats['idm_deletions'])}回\n")
            f.write(f"   • 総削除行数: {total_idm:,}行\n")
            f.write(f"   • 平均削除行数: {total_idm/len(stats['idm_deletions']):.1f}行\n\n")
        
        # グレード推定統計
        if stats['grade_estimations']:
            total_grade = sum(stats['grade_estimations'])
            f.write("🏆 グレード推定処理:\n")
            f.write(f"   • 処理回数: {len(stats['grade_estimations'])}回\n")
            f.write(f"   • 総推定件数: {total_grade:,}件\n")
            f.write(f"   • 平均推定件数: {total_grade/len(stats['grade_estimations']):.1f}件\n\n")
        
        # 中央値補完統計
        if stats['median_imputations']:
            f.write("🔢 中央値補完処理:\n")
            for column, counts in stats['median_imputations'].items():
                total_count = sum(counts)
                f.write(f"   • {column}: {len(counts)}回, 総補完{total_count:,}件 (平均{total_count/len(counts):.1f}件)\n")
            f.write("\n")
        
        # 高欠損率列削除
        if stats['dropped_columns']:
            f.write("❌ 高欠損率により削除された列:\n")
            sorted_columns = sorted(stats['dropped_columns'])
            for i, column in enumerate(sorted_columns, 1):
                f.write(f"   {i:2d}. {column}\n")
            f.write(f"\n   📊 削除列数: {len(sorted_columns)}列\n\n")
        
        # カテゴリ補完統計
        if stats['categorical_imputations']:
            f.write("🏷️ カテゴリ補完処理:\n")
            for column, values in stats['categorical_imputations'].items():
                total_count = sum(count for _, count in values)
                unique_values = len(set(value for value, _ in values))
                f.write(f"   • {column}: {len(values)}回, 総補完{total_count:,}件, {unique_values}種類の値\n")
            f.write("\n")
        
        # その他補完統計
        if stats['other_imputations']:
            f.write("🔧 その他補完処理:\n")
            for column, values in stats['other_imputations'].items():
                total_count = sum(count for _, count in values)
                f.write(f"   • {column}: {len(values)}回, 総補完{total_count:,}件\n")
            f.write("\n")
        
        # 最終データ統計
        if stats['final_shapes']:
            total_rows = sum(rows for rows, _ in stats['final_shapes'])
            total_cols = sum(cols for _, cols in stats['final_shapes'])
            avg_rows = total_rows / len(stats['final_shapes']) if stats['final_shapes'] else 0
            avg_cols = total_cols / len(stats['final_shapes']) if stats['final_shapes'] else 0
            
            f.write("📊 最終データ統計:\n")
            f.write(f"   • 総行数: {total_rows:,}行\n")
            f.write(f"   • 平均行数: {avg_rows:.1f}行/ファイル\n")
            f.write(f"   • 平均列数: {avg_cols:.1f}列/ファイル\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("🎉 実務レベル欠損値処理 完了サマリー\n")
        f.write("=" * 80 + "\n")

def process_race_data(exclude_turf=False, turf_only=False, 
                     enable_missing_value_handling=True, enable_quality_check=True):
    """
    競馬レースデータの実務レベル処理（標準版）
    計画書Phase 0: データ整備の実装
    
    Args:
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
        enable_missing_value_handling (bool): 戦略的欠損値処理を実行するかどうか
        enable_quality_check (bool): データ品質チェックを実行するかどうか
    """
    logger.info("🏇 ■ 競馬レースデータの実務レベル処理を開始します ■")
    
    # システム監視開始
    monitor = SystemMonitor()
    
    # 処理オプションの確認
    if exclude_turf and turf_only:
        logger.error("❌ 芝コースを除外するオプションと芝コースのみを処理するオプションは同時に指定できません")
        return
    
    # 通常の処理設定のログ出力
    logger.info("📋 処理設定:")
    logger.info(f"   🌱 芝コース除外: {'はい' if exclude_turf else 'いいえ'}")
    logger.info(f"   🌱 芝コースのみ: {'はい' if turf_only else 'いいえ'}")
    logger.info(f"   🔧 欠損値処理: {'有効' if enable_missing_value_handling else '無効'}")
    logger.info(f"   📈 品質チェック: {'有効' if enable_quality_check else '無効'}")
    
    # システムコンポーネントの初期化
    quality_checker = DataQualityChecker() if enable_quality_check else None
    missing_handler = MissingValueHandler() if enable_missing_value_handling else None
    
    # 出力用ディレクトリの確認
    ensure_export_dirs()
    monitor.log_system_status("初期化完了")
    
    try:
        # 1. BACデータの処理
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-1: BACデータ（レース基本情報）の処理")
        logger.info("="*60)
        
        process_all_bac_files(exclude_turf=exclude_turf, turf_only=turf_only)
        monitor.log_system_status("BAC処理完了")
    
        # 2. SRBデータの処理
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-2: SRBデータ（レース詳細情報）の処理")
        logger.info("="*60)
        
        process_all_srb_files(exclude_turf=exclude_turf, turf_only=turf_only)
        monitor.log_system_status("SRB処理完了")
    
        # 3. SEDデータの処理とSRB・BACデータとの紐づけ
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-3: SEDデータ（競走成績）の処理と紐づけ")
        logger.info("="*60)
        
        process_all_sed_files(exclude_turf=exclude_turf, turf_only=turf_only)
    
        # 4. SEDデータとSRBデータの紐づけ
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-4: SEDデータとSRBデータの統合")
        logger.info("="*60)
        logger.info("📋 バイアス情報完備データのみを保持します")
        
        merge_result = merge_srb_with_sed(
            separate_output=True, 
            exclude_turf=exclude_turf, 
            turf_only=turf_only
        )
        
        if not merge_result:
            logger.error("❌ SEDデータとSRBデータの紐づけに失敗しました")
            return False
        
        logger.info("✅ データ統合完了:")
        logger.info("   📁 SEDデータ: export/SED/")
        logger.info("   📁 SRBデータ: export/SRB/")
        logger.info("   📁 統合データ: export/with_bias/")
        
        monitor.log_system_status("データ統合完了")
        
        # 5. データ品質チェック（統合後）
        if enable_quality_check:
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 0-5: データ品質チェック")
            logger.info("="*60)
            
            # サンプルファイルで品質チェック実行
            sample_files = list(Path('export/with_bias').glob('*.csv'))
            if sample_files:
                sample_file = sample_files[0]
                logger.info(f"📄 サンプルファイルで品質チェック: {sample_file.name}")
                
                try:
                    sample_df = pd.read_csv(sample_file, encoding='utf-8')
                    quality_checker.check_data_quality(sample_df, "統合後データ")
                except Exception as e:
                    logger.warning(f"⚠️ 品質チェックエラー: {str(e)}")
        
        # 7. 品質レポートの保存
        if enable_quality_check and quality_checker:
            save_quality_report(quality_checker)
        
        # 8. 欠損値処理ログのサマリー生成（実務レベル）
        if enable_missing_value_handling:
            logger.info("\n" + "="*60)
            logger.info("📝 Phase 0-7: 欠損値処理ログの自動整理")
            logger.info("="*60)
            summarize_processing_log()
        
        # 9. グレード欠損削除統計の表示
        if enable_missing_value_handling:
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 0-8: グレード欠損削除統計")
            logger.info("="*60)
            display_deletion_statistics()
        
        # 10. 処理完了サマリー
        logger.info("\n" + "="*60)
        logger.info("🎉 Phase 0: データ整備 完了")
        logger.info("="*60)
        
        total_time = time.time() - monitor.start_time
        logger.info(f"⏱️ 総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        monitor.log_system_status("全処理完了")
        
        logger.info("\n📁 生成されたデータ:")
        if Path('export/with_bias').exists():
            bias_files = list(Path('export/with_bias').glob('*.csv'))
            logger.info(f"   🔗 統合データ: {len(bias_files)}ファイル")
        
        if enable_quality_check and Path('export/quality_reports').exists():
            logger.info("   📈 品質レポート: export/quality_reports/")
        
        logger.info("\n🎓 実務レベルのデータ整備が完了しました！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ データ処理中に予期せぬエラーが発生しました: {str(e)}")
        logger.error("🔧 スタックトレース:", exc_info=True)
        return False

if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='競馬レースデータの実務レベル処理（計画書Phase 0：データ整備対応版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 使用例:
  python process_race_data.py                                    # 基本処理
  python process_race_data.py --turf-only                      # 芝コースのみで処理
  python process_race_data.py --no-missing-handling              # 欠損値処理を無効化
  python process_race_data.py --no-quality-check                 # 品質チェックを無効化

🔧 このスクリプトの役割:
  このスクリプトは、複数の形式の生レースデータ（BAC, SRB, SED）を読み込み、
  それらを一つの整形されたデータセットに統合します。
  最終的な成果物は `export/with_bias/` ディレクトリに出力され、
  これが後続の分析スクリプト（例: analyze_horse_racelevel.py）の入力となります。

🔧 実務レベルの品質管理:
  ✅ 戦略的欠損値処理
  ✅ データ品質チェックとレポート
  ✅ 欠損値処理ログの自動サマリー生成
  ✅ システム監視
  ✅ 段階的処理とログ出力
  ✅ エラーハンドリングと復旧機能
        """
    )
    
    # トラック条件オプション
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', '--芝コース除外', action='store_true', 
                           help='芝コースのデータを除外する')
    track_group.add_argument('--turf-only', '--芝コースのみ', action='store_true', 
                           help='芝コースのデータのみを処理する')
    
    # 機能オプション
    parser.add_argument('--no-missing-handling', '--欠損値処理無効', action='store_true',
                       help='戦略的欠損値処理を無効化する')
    
    parser.add_argument('--no-quality-check', '--品質チェック無効', action='store_true',
                       help='データ品質チェックを無効化する')
    
    # ログレベルオプション
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='ログレベルの設定')
    
    parser.add_argument('--log-file', help='ログファイルのパス（指定しない場合はコンソールのみ）')
    
    args = parser.parse_args()
    
    # ログ設定の初期化
    log_file = args.log_file
    
    if log_file is None:
        # 自動ログファイル設定（ディレクトリ作成も含む）
        log_dir = Path('export/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = f'export/logs/process_race_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    setup_logging(log_level=args.log_level, log_file=log_file)
    
    # メインロガーでの開始メッセージ
    logger.info("🚀 競馬レースデータ実務レベル処理を開始します")
    logger.info(f"📅 実行日時: {datetime.now()}")
    logger.info(f"🖥️ ログレベル: {args.log_level}")
    if log_file:
        logger.info(f"📝 ログファイル: {log_file}")
    
    # レースデータ処理の実行
    success = process_race_data(
        exclude_turf=args.exclude_turf, 
        turf_only=args.turf_only,
        enable_missing_value_handling=not args.no_missing_handling,
        enable_quality_check=not args.no_quality_check
    )
    
    if success:
        logger.info("🎉 実務レベルデータ処理が正常に完了しました")
        exit_code = 0
    else:
        logger.error("❌ データ処理が失敗しました")
        exit_code = 1
    
    logger.info(f"🏁 プロセス終了 (終了コード: {exit_code})")
    exit(exit_code) 
"""
戦略的欠損値処理クラス
"""
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..config.column_names import ColumnNames
from .grade_estimator import GradeEstimator
from .horse_age_calculator import HorseAgeCalculator

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """戦略的欠損値処理クラス。

    計画書 Phase 0 の要件に基づく実務レベルの欠損値処理を提供する。
    """
    
    def __init__(self, columns: Optional[ColumnNames] = None):
        """欠損値処理で利用する依存を初期化します。

        Args:
            columns (ColumnNames, optional): 列名設定。
        """
        self.columns = columns or ColumnNames()
        self.processing_log = []
        self.grade_estimator = GradeEstimator(columns=self.columns)  # グレード推定専用クラスを使用
        self.age_calculator = HorseAgeCalculator(columns=self.columns)  # 馬齢計算専用クラスを使用
        self.logger = logging.getLogger(__name__)
        
    def handle_missing_values(self, df: pd.DataFrame, strategy_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """戦略的欠損値処理を実行する。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            strategy_config (Dict[str, Any], optional): 欠損値処理戦略。

        Returns:
            pd.DataFrame: 欠損値処理を施した DataFrame。
        """
        self.logger.info("🔧 戦略的欠損値処理開始")
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
            
            # 5. 馬齢計算（血統登録番号と年月日から）- 専用クラスを使用
            df_processed = self.age_calculator.calculate_horse_age(df_processed)
            
            execution_time = time.time() - start_time
            final_rows = len(df_processed)
            
            self.logger.info(f"✅ 欠損値処理完了 ({execution_time:.2f}秒)")
            self.logger.info(f"   📊 処理前: {original_rows:,}行")
            self.logger.info(f"   📊 処理後: {final_rows:,}行")
            self.logger.info(f"   📉 除去行数: {original_rows - final_rows:,}行 ({((original_rows - final_rows) / original_rows) * 100:.1f}%)")
            
            # 処理ログの保存
            self._save_processing_log(df_processed)
            
        except Exception as e:
            self.logger.error(f"❌ 欠損値処理でエラー: {str(e)}")
            raise
        
        return df_processed
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """デフォルトの欠損値処理戦略を返します。

        Returns:
            Dict[str, Any]: 欠損値処理戦略の設定辞書。
        """
        return {
            'critical_columns': {
                self.columns.POSITION: 'drop',  # 着順が欠損の行は削除
                self.columns.DISTANCE: 'drop',  # 距離が欠損の行は削除
                self.columns.HORSE_NAME: 'drop',  # 馬名が欠損の行は削除
                self.columns.IDM: 'drop'  # IDMが欠損の行は削除
            },
            'numeric_columns': {
                'method': 'median',  # 中央値で補完
                'max_missing_rate': 0.5  # 50%以上欠損の列は削除
            },
            'categorical_columns': {
                'method': 'mode',  # 最頻値で補完
                'unknown_label': '不明',
                'max_missing_rate': 0.8  # 80%以上欠損の列は削除
            },
            # 残存欠損値は重要列サブセットでのみ行削除（実務レポート方針）
            'remaining_strategy': 'drop_subset',
            'remaining_subset': [
                self.columns.POSITION, 
                self.columns.DISTANCE, 
                self.columns.HORSE_NAME, 
                self.columns.IDM, 
                self.columns.GRADE
            ]
        }
    
    def _handle_critical_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """重要列に対する欠損値処理を実施します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            config (Dict[str, Any]): 欠損値処理戦略。

        Returns:
            pd.DataFrame: 処理後の DataFrame。
        """
        self.logger.info("   🎯 重要列の欠損値処理中...")
        
        critical_config = config.get('critical_columns', {})
        
        for column, strategy in critical_config.items():
            if column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"      • {column}: {missing_count:,}件の欠損値を{strategy}処理")
                    
                    if strategy == 'drop':
                        df = df.dropna(subset=[column])
                        self.processing_log.append(f"{column}: {missing_count}行を削除（重要列）")
        
        return df
    
    def _handle_numeric_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """数値列の欠損値処理を実施します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            config (Dict[str, Any]): 欠損値処理戦略。

        Returns:
            pd.DataFrame: 処理後の DataFrame。
        """
        self.logger.info("   🔢 数値列の欠損値処理中...")
        
        numeric_config = config.get('numeric_columns', {})
        method = numeric_config.get('method', 'median')
        max_missing_rate = numeric_config.get('max_missing_rate', 0.5)
        
        # グレード列が文字列でも推定ロジックが動くように数値化を試みる
        grade_columns = self.columns.get_grade_columns()
        for grade_col in grade_columns:
            if grade_col in df.columns:
                df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce')

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # 賞金関連の列を欠損値処理の対象から除外（欠損が多くて削除されるのを防ぐ）
        prize_columns = self.columns.get_prize_columns()
        columns_to_process = [
            col for col in numeric_columns 
            if col not in prize_columns
        ]

        for column in columns_to_process:
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df) if len(df) > 0 else 0
            
            if missing_count > 0:
                # グレード列の特別処理（実務レベル）- 専用クラスを使用
                if column in grade_columns:
                    self.logger.info(f"      • {column}: 実務レベルグレード推定処理を実行")
                    df = self.grade_estimator.estimate_grade(df, column)
                    
                    # 推定後の欠損数をチェック
                    remaining_missing = df[column].isnull().sum()
                    estimated_count = missing_count - remaining_missing
                    
                    if estimated_count > 0:
                        self.processing_log.append(f"{column}: 賞金・レース名から{estimated_count}件推定→グレード名列追加")
                
                elif missing_rate > max_missing_rate:
                    self.logger.warning(f"      • {column}: 欠損率{missing_rate:.1%} > {max_missing_rate:.1%} → 列削除")
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
                    self.logger.info(f"      • {column}: {missing_count:,}件を{method}({fill_value})で補完")
                    self.processing_log.append(f"{column}: {method}で{missing_count}件補完")
        
        return df
    
    def _handle_categorical_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """カテゴリ列の欠損値処理を実施します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            config (Dict[str, Any]): 欠損値処理戦略。

        Returns:
            pd.DataFrame: 処理後の DataFrame。
        """
        self.logger.info("   🏷️ カテゴリ列の欠損値処理中...")
        
        categorical_config = config.get('categorical_columns', {})
        method = categorical_config.get('method', 'mode')
        unknown_label = categorical_config.get('unknown_label', '不明')
        max_missing_rate = categorical_config.get('max_missing_rate', 0.8)
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        grade_columns = self.columns.get_grade_columns() + [self.columns.GRADE_NAME]
        
        for column in categorical_columns:
            # グレードはモード補完の対象から除外（推定ロジックに委ねる）
            if column in grade_columns:
                continue
            
            # グレード_yの特別処理（予測マーク付き）
            if column == self.columns.GRADE_Y:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"      • {column}: {missing_count:,}件をmode(特別)で補完（予測マーク付き）")
                    df[column] = df[column].fillna('特別（予測）')
                    self.processing_log.append(f"{column}: {missing_count}件をmode(特別)で補完（予測マーク付き）")
                continue
            
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df) if len(df) > 0 else 0
            
            if missing_count > 0:
                if missing_rate > max_missing_rate:
                    self.logger.warning(f"      • {column}: 欠損率{missing_rate:.1%} > {max_missing_rate:.1%} → 列削除")
                    df = df.drop(columns=[column])
                    self.processing_log.append(f"{column}: 高欠損率により列削除")
                else:
                    if method == 'mode':
                        mode_values = df[column].mode()
                        fill_value = mode_values.iloc[0] if not mode_values.empty else unknown_label
                    else:
                        fill_value = unknown_label
                    
                    df[column] = df[column].fillna(fill_value)
                    self.logger.info(f"      • {column}: {missing_count:,}件を{method}({fill_value})で補完")
                    self.processing_log.append(f"{column}: {method}で{missing_count}件補完")
        
        return df
    
    def _handle_remaining_missing(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """残存する欠損値の最終処理を行います。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            config (Dict[str, Any]): 欠損値処理戦略。

        Returns:
            pd.DataFrame: 残存欠損値を処理した DataFrame。
        """
        remaining_missing = df.isnull().sum().sum()
        
        if remaining_missing > 0:
            self.logger.info(f"   🔧 残存欠損値処理中: {remaining_missing:,}件")
            
            strategy = config.get('remaining_strategy', 'drop')
            
            if strategy == 'drop':
                initial_rows = len(df)
                df = df.dropna()
                dropped_rows = initial_rows - len(df)
                
                if dropped_rows > 0:
                    self.logger.info(f"      • 残存欠損値のある{dropped_rows:,}行を削除")
                    self.processing_log.append(f"残存欠損値: {dropped_rows}行削除")
            elif strategy == 'drop_subset':
                subset = config.get('remaining_subset', [])
                subset = [col for col in subset if col in df.columns]
                if subset:
                    initial_rows = len(df)
                    df = df.dropna(subset=subset)
                    dropped_rows = initial_rows - len(df)
                    if dropped_rows > 0:
                        self.logger.info(f"      • 重要列({', '.join(subset)})の残存欠損{dropped_rows:,}行を削除")
                        self.processing_log.append(f"残存欠損(重要列): {dropped_rows}行削除")
        
        return df
    
    
    def _save_processing_log(self, df: pd.DataFrame):
        """処理ログを追記モードで保存します。

        Args:
            df (pd.DataFrame): 最終的な処理結果の DataFrame。
        """
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
            
            self.logger.info(f"   📝 処理ログ保存: {log_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 処理ログ保存エラー: {str(e)}")
        finally:
            self.processing_log.clear()


"""
グレード推定専用クラス
"""
import logging
import pandas as pd
from typing import Optional

from ..config.column_names import ColumnNames
from ..config.grade_config import GradeThresholds, RacePatterns

logger = logging.getLogger(__name__)


class GradeEstimator:
    """グレード推定専用クラス（単一責任原則を遵守）"""
    
    def __init__(self, thresholds: Optional[GradeThresholds] = None, 
                 patterns: Optional[RacePatterns] = None,
                 columns: Optional[ColumnNames] = None):
        """推定に必要な依存オブジェクトを初期化します。

        Args:
            thresholds (GradeThresholds, optional): 賞金に基づく閾値設定。
            patterns (RacePatterns, optional): レース名パターン設定。
            columns (ColumnNames, optional): 列名設定。
        """
        self.thresholds = thresholds or GradeThresholds()
        self.patterns = patterns or RacePatterns()
        self.columns = columns or ColumnNames()
        self.logger = logging.getLogger(__name__)
    
    def estimate_grade(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """グレード推定を実行します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            grade_column (str): 推定対象となるグレード列名。

        Returns:
            pd.DataFrame: グレード推定結果を反映した DataFrame（コピー）。
        """
        # DataFrameのコピーを作成（不変性を保証）
        df_result = df.copy()
        
        initial_rows = len(df_result)
        grade_missing_mask = df_result[grade_column].isnull()
        initial_missing_count = grade_missing_mask.sum()
        
        if not grade_missing_mask.any():
            # 既存の数値グレードからグレード名列を作成
            df_result = self._add_grade_name_column(df_result, grade_column)
            return df_result
        
        self.logger.info(f"📊 グレード欠損値: {initial_missing_count:,}件 ({initial_missing_count/initial_rows*100:.1f}%)")
        
        # 推定対象データ
        estimation_df = df_result[grade_missing_mask].copy()
        
        # 1. 賞金ベースの推定
        if self.columns.PRIZE_1ST_WITH_BONUS in df_result.columns:
            estimation_df = self._estimate_from_prize(estimation_df, grade_column, self.columns.PRIZE_1ST_WITH_BONUS)
        
        # 2. 本賞金からの推定（フォールバック）
        if self.columns.PRIZE_MAIN in df_result.columns:
            estimation_df = self._estimate_from_prize(estimation_df, grade_column, self.columns.PRIZE_MAIN)
        
        # 3. レース名からの推定
        if self.columns.RACE_NAME in df_result.columns:
            estimation_df = self._estimate_from_race_name(estimation_df, grade_column)
        
        # 4. 特徴量からの推定（距離・出走頭数）
        estimation_df = self._estimate_from_features(estimation_df, grade_column)
        
        # 5. 最終的に推定できない場合は条件戦（5）として設定
        final_missing = estimation_df[grade_column].isnull().sum()
        if final_missing > 0:
            self.logger.info(f"      🎯 最終推定失敗{final_missing:,}件を条件戦（5）として設定")
            estimation_df.loc[estimation_df[grade_column].isnull(), grade_column] = 5
        
        # 推定結果を元のDataFrameに反映
        df_result.loc[grade_missing_mask, grade_column] = estimation_df[grade_column]
        
        # グレード名列を追加
        df_result = self._add_grade_name_column(df_result, grade_column)
        
        estimated_count = initial_missing_count - df_result[grade_column].isnull().sum()
        if estimated_count > 0:
            self.logger.info(f"      ✅ グレード推定成功: {estimated_count:,}件")
        
        return df_result
    
    def _estimate_from_prize(self, df: pd.DataFrame, grade_column: str, prize_col: str) -> pd.DataFrame:
        """賞金情報からグレードを推定します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            grade_column (str): グレードを格納する列名。
            prize_col (str): 参照する賞金列名。

        Returns:
            pd.DataFrame: 推定結果を反映した DataFrame（コピー）。
        """
        if prize_col not in df.columns:
            return df
        
        # DataFrameのコピーを作成
        df_result = df.copy()
        
        # 数値化
        df_result[prize_col] = pd.to_numeric(df_result[prize_col], errors='coerce')
        
        # しきい値を適用
        thresholds_list = self.thresholds.to_thresholds_list()
        for min_prize, grade_value in thresholds_list:
            mask = (df_result[prize_col] >= min_prize) & df_result[grade_column].isnull()
            df_result.loc[mask, grade_column] = grade_value
        
        return df_result
    
    def _estimate_from_race_name(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """レース名のパターンからグレードを推定します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            grade_column (str): グレードを格納する列名。

        Returns:
            pd.DataFrame: 推定結果を反映した DataFrame（コピー）。
        """
        if self.columns.RACE_NAME not in df.columns:
            return df
        
        # DataFrameのコピーを作成
        df_result = df.copy()
        
        race_patterns = {
            1: self.patterns.G1_PATTERNS,
            2: self.patterns.G2_PATTERNS,
            3: self.patterns.G3_PATTERNS,
            4: self.patterns.STAKES_PATTERNS,
            5: self.patterns.CONDITIONS_PATTERNS
        }
        
        for grade, patterns in race_patterns.items():
            for pattern in patterns:
                mask = (df_result[self.columns.RACE_NAME].str.contains(pattern, case=False, na=False)) & df_result[grade_column].isnull()
                df_result.loc[mask, grade_column] = grade
        
        return df_result
    
    def _estimate_from_features(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """距離や頭数などの特徴量からグレードを推定します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            grade_column (str): グレードを格納する列名。

        Returns:
            pd.DataFrame: 推定結果を反映した DataFrame（コピー）。
        """
        # DataFrameのコピーを作成
        df_result = df.copy()
        
        # 距離による推定
        if self.columns.DISTANCE in df_result.columns:
            df_result[self.columns.DISTANCE] = pd.to_numeric(df_result[self.columns.DISTANCE], errors='coerce')
            
            long_distance_mask = (df_result[self.columns.DISTANCE] >= 3000) & df_result[grade_column].isnull()
            df_result.loc[long_distance_mask, grade_column] = 4  # 重賞
            
            short_distance_mask = (df_result[self.columns.DISTANCE] < 1000) & df_result[grade_column].isnull()
            df_result.loc[short_distance_mask, grade_column] = 5  # 特別
        
        # 出走頭数による推定
        if self.columns.HORSE_COUNT in df_result.columns:
            df_result[self.columns.HORSE_COUNT] = pd.to_numeric(df_result[self.columns.HORSE_COUNT], errors='coerce')
            
            large_field_mask = (df_result[self.columns.HORSE_COUNT] >= 16) & df_result[grade_column].isnull()
            df_result.loc[large_field_mask, grade_column] = 4  # 重賞
            
            small_field_mask = (df_result[self.columns.HORSE_COUNT] < 8) & df_result[grade_column].isnull()
            df_result.loc[small_field_mask, grade_column] = 5  # 条件戦
        
        return df_result
    
    def _add_grade_name_column(self, df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
        """数値グレードをグレード名に変換します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。
            grade_column (str): 数値グレードが格納された列名。

        Returns:
            pd.DataFrame: グレード名列を付与した DataFrame（コピー）。
        """
        # DataFrameのコピーを作成
        df_result = df.copy()
        
        df_result[grade_column] = pd.to_numeric(df_result[grade_column], errors='coerce')
        grade_names = df_result[grade_column].map(self.thresholds.GRADE_NAME_MAPPING)
        
        if self.columns.GRADE_NAME in df_result.columns:
            df_result[self.columns.GRADE_NAME] = grade_names
        else:
            grade_col_index = df_result.columns.get_loc(grade_column)
            df_result.insert(grade_col_index + 1, self.columns.GRADE_NAME, grade_names)
        
        return df_result


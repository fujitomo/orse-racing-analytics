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
import argparse
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import re
from collections import defaultdict
from dataclasses import dataclass

# モジュール共通ロガー
logger = logging.getLogger(__name__)

# =====================================
# 列名の定義（既存コードとの互換用）
# =====================================

class ColumnNames:
    """データ列名の集中定義とユーティリティ。

    既存の日本語列名を対象に、推定ロジックが参照する列名を提供する。
    """
    # 基本列
    RACE_NAME = 'レース名'
    DISTANCE = '距離'
    HORSE_COUNT = '頭数'
    POSITION = '着順'
    HORSE_NAME = '馬名'
    HORSE_AGE = '馬齢'
    IDM = 'IDM'
    GRADE = 'グレード'
    GRADE_Y = 'グレード_y'
    GRADE_NAME = 'グレード名'

    # 日付・識別
    REGISTRATION_NUMBER = '血統登録番号'
    RACE_DATE = '年月日'

    # 賞金関連
    PRIZE_1ST_WITH_BONUS = '1着賞金(1着算入賞金込み)'
    PRIZE_MAIN = '本賞金'

    def get_grade_columns(self):
        return [self.GRADE, 'grade', 'レースグレード']

    def get_prize_columns(self):
        return [
            '2着賞金', '3着賞金', '4着賞金', '5着賞金',
            '1着算入賞金', '2着算入賞金',
            self.PRIZE_1ST_WITH_BONUS, '2着賞金(2着算入賞金込み)', '平均賞金',
            self.PRIZE_MAIN
        ]

# =====================================
# グレード推定用の設定クラス（マジックナンバー解消）
# =====================================

@dataclass
class GradeThresholds:
    """グレード推定用の賞金閾値設定（formattedデータ分析結果に基づく実証的基準）"""
    G1_MIN: int = 3407    # G1: 3,407万円以上（G1レース平均）
    G2_MIN: int = 2177    # G2: 2,177万円以上（G2レース平均）
    G3_MIN: int = 1438    # G3: 1,438万円以上（G3レース平均）
    LISTED_MIN: int = 903  # L（リステッド）: 903万円以上（Lレース平均）
    SPECIAL_MIN: int = 552 # 特別/OP: 552万円以上（特別レース平均）
    
    # グレード名マッピング
    GRADE_NAME_MAPPING: Dict[int, str] = None
    
    def __post_init__(self):
        if self.GRADE_NAME_MAPPING is None:
            object.__setattr__(self, 'GRADE_NAME_MAPPING', {
                1: 'Ｇ１',
                2: 'Ｇ２', 
                3: 'Ｇ３',
                4: '重賞',
                5: '特別',
                6: 'Ｌ（リステッド）'
            })
    
    def to_thresholds_list(self) -> List[Tuple[int, int]]:
        """賞金しきい値を降順リストに変換します。

        Returns:
            List[Tuple[int, int]]: 最低賞金と対応するグレード値のタプルのリスト。
        """
        return [
            (self.G1_MIN, 1),
            (self.G2_MIN, 2),
            (self.G3_MIN, 3),
            (self.LISTED_MIN, 6),
            (self.SPECIAL_MIN, 5)
        ]

@dataclass
class RacePatterns:
    """レース名パターン定義"""
    G1_PATTERNS: List[str] = None
    G2_PATTERNS: List[str] = None
    G3_PATTERNS: List[str] = None
    STAKES_PATTERNS: List[str] = None
    CONDITIONS_PATTERNS: List[str] = None
    
    def __post_init__(self):
        if self.G1_PATTERNS is None:
            self.G1_PATTERNS = [
                'ジャパンカップ', '有馬記念', '大阪杯', '東京優駿',
                '天皇賞', '宝塚記念', '皐月賞', '菊花賞',
                '安田記念', 'マイルチャンピオンシップ',
                '高松宮記念', 'スプリンターズステークス',
                '優駿牝馬', '桜花賞', 'ヴィクトリアマイル',
                'エリザベス女王杯', 'ジャパンカップダート',
                'ＮＨＫマイルカップ', 'チャンピオンズカップ',
                'フェブラリーステークス', '秋華賞', 'ＪＢＣクラシック',
                '中山グランドジャンプ', '中山大障害',
                '朝日杯フューチュリティステークス', 'ＪＢＣスプリント',
                'ダービー', 'オークス', 'マイル', 'フューチュリティ'
            ]
        
        if self.G2_PATTERNS is None:
            self.G2_PATTERNS = ['札幌記念', '阪神カップ', '記念', '大賞典']
        
        if self.G3_PATTERNS is None:
            self.G3_PATTERNS = ['賞', '特別']
        
        if self.STAKES_PATTERNS is None:
            self.STAKES_PATTERNS = ['重賞', 'リステッド', 'L']
        
        if self.CONDITIONS_PATTERNS is None:
            self.CONDITIONS_PATTERNS = ['条件', '新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス']

# =====================================
# グレード推定専用クラス（SRP遵守）
# =====================================

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

# =====================================
# 馬齢計算専用クラス（SRP遵守）
# =====================================

class HorseAgeCalculator:
    """馬齢計算専用クラス"""
    
    DEFAULT_HORSE_AGE = 3  # 日本競馬の一般的なデビュー年齢
    VALID_AGE_RANGE = (2, 20)  # 競走馬の妥当な年齢範囲
    
    def __init__(self, columns: Optional[ColumnNames] = None):
        """
        Args:
            columns: 列名設定
        """
        self.columns = columns or ColumnNames()
        self.logger = logging.getLogger(__name__)
    
    def calculate_horse_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """血統登録番号と年月日から馬齢を算出します。

        Args:
            df (pd.DataFrame): 処理対象の DataFrame。

        Returns:
            pd.DataFrame: 馬齢列を追加した DataFrame（コピー）。
        """
        try:
            # DataFrameのコピーを作成（不変性を保証）
            df_result = df.copy()
            
            # 必要な列の確認
            if self.columns.REGISTRATION_NUMBER not in df_result.columns or self.columns.RACE_DATE not in df_result.columns:
                self.logger.warning("⚠️ 血統登録番号または年月日列が見つかりません")
                return df_result
            
            # レース日で安定ソートし、初出走レースを確実に取得
            if self.columns.RACE_DATE in df_result.columns:
                df_result = df_result.sort_values(by=self.columns.RACE_DATE, kind='stable')

            # 馬齢列を初期化
            df_result[self.columns.HORSE_AGE] = None
            
            # 馬ごとに最初のレース情報を取得
            horse_first_race = df_result.groupby(self.columns.HORSE_NAME, sort=False).first()
            
            horse_age_map = {}
            
            for horse_name, row in horse_first_race.iterrows():
                try:
                    registration_raw = row[self.columns.REGISTRATION_NUMBER]
                    race_date_raw = row[self.columns.RACE_DATE]

                    registration_number = re.sub(r'\D', '', str(registration_raw))
                    if len(registration_number) < 2:
                        self.logger.debug(f"⚠️ 血統登録番号形式エラー: {horse_name}")
                        horse_age_map[horse_name] = self.DEFAULT_HORSE_AGE
                        continue

                    birth_year = int(registration_number[:2])
                    birth_year = birth_year + 2000 if birth_year <= 30 else birth_year + 1900

                    if pd.isna(race_date_raw):
                        self.logger.debug(f"⚠️ 日付欠損: {horse_name}")
                        horse_age_map[horse_name] = self.DEFAULT_HORSE_AGE
                        continue

                    race_date_digits = re.sub(r'\D', '', str(race_date_raw))
                    if len(race_date_digits) != 8:
                        self.logger.debug(f"⚠️ 日付形式エラー: {horse_name}")
                        horse_age_map[horse_name] = self.DEFAULT_HORSE_AGE
                        continue

                    race_year = int(race_date_digits[:4])

                    # 馬齢計算（日本競馬では1月1日に全馬が加齢）
                    age = race_year - birth_year

                    if self.VALID_AGE_RANGE[0] <= age <= self.VALID_AGE_RANGE[1]:
                        horse_age_map[horse_name] = age
                    else:
                        self.logger.debug(f"⚠️ 異常な年齢: {horse_name} (計算年齢:{age})")
                        horse_age_map[horse_name] = self.DEFAULT_HORSE_AGE
                        
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"⚠️ 年齢計算エラー: {horse_name} - {str(e)}")
                    horse_age_map[horse_name] = self.DEFAULT_HORSE_AGE
            
            # 馬齢列に値を設定
            df_result[self.columns.HORSE_AGE] = df_result[self.columns.HORSE_NAME].map(horse_age_map)
            
            # 統計情報をログ出力
            age_counts = {}
            for age in horse_age_map.values():
                age_counts[age] = age_counts.get(age, 0) + 1
            
            self.logger.info(f"✅ 馬齢計算完了: {len(horse_age_map)}頭")
            self.logger.info(f"📊 年齢分布: {dict(sorted(age_counts.items()))}")
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"❌ 馬齢計算エラー: {str(e)}")
            return df

# 実務レベルのログ設定
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """実務レベルのログ設定を初期化する。

    Args:
        log_level (str): ログレベル（例: ``INFO``, ``DEBUG``）。
        log_file (str, optional): ログ出力ファイルパス。``None`` の場合はコンソールのみ。
    """
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

class DataQualityChecker:
    """データ品質チェッククラス。

    実務レベルのデータ整備に必要な品質管理機能を提供する。
    """
    
    def __init__(self):
        """インスタンスを初期化する。"""
        self.quality_report = {}  # 各処理段階のデータ品質レポートを格納する辞書
        self.logger = logging.getLogger(__name__)
        
    def check_data_quality(self, df: pd.DataFrame, stage_name: str) -> Dict[str, Any]:
        """包括的なデータ品質チェックを実行します。

        Args:
            df (pd.DataFrame): チェック対象の DataFrame。
            stage_name (str): 処理段階名（例: ``BAC処理後``）。

        Returns:
            Dict[str, Any]: 品質レポートを格納した辞書。
        """
        self.logger.info(f"📊 {stage_name} - データ品質チェック開始")
        start_time = time.time()
        
        report = {
            'stage': stage_name,  # 処理段階名（例：'BAC処理後', '統合後'）
            'timestamp': datetime.now().isoformat(),  # 品質チェック実行時刻（ISO形式）
            'total_rows': len(df),  # データ行数（レコード数）
            'total_columns': len(df.columns),  # データ列数（カラム数）
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,  # メモリ使用量（MB）
            'missing_values': {},  # 欠損値分析結果（列別の欠損数・割合）
            'data_types': {},  # データ型情報（列名とデータ型のマッピング）
            'duplicates': 0,  # 重複行数
            'outliers': {},  # 外れ値検出結果（列別の外れ値数）
            'warnings': [],  # 品質警告リスト（異常値、不正データなど）
            'recommendations': []  # 改善推奨事項リスト
        }
        
        try:
            # 1. 欠損値分析
            self.logger.info("   🔍 欠損値分析中...")
            missing_analysis = self._analyze_missing_values(df)
            report['missing_values'] = missing_analysis
            
            # 2. データ型チェック
            self.logger.info("   🏷️ データ型チェック中...")
            report['data_types'] = self._check_data_types(df)
            
            # 3. 重複チェック
            self.logger.info("   🔄 重複チェック中...")
            report['duplicates'] = int(df.duplicated().sum())
            
            # 4. 外れ値検出（数値列のみ）
            self.logger.info("   📈 外れ値検出中...")
            report['outliers'] = self._detect_outliers(df)
            
            # 5. ビジネスルール検証
            self.logger.info("   📋 ビジネスルール検証中...")
            warnings, recommendations = self._validate_business_rules(df)
            report['warnings'] = warnings
            report['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            report['execution_time_seconds'] = execution_time
            
            self.logger.info(f"✅ {stage_name} - データ品質チェック完了 ({execution_time:.2f}秒)")
            
            # レポート要約をログ出力
            self._log_quality_summary(report)
            
        except Exception as e:
            self.logger.error(f"❌ データ品質チェックでエラー: {str(e)}")
            report['error'] = str(e)
        
        self.quality_report[stage_name] = report
        return report
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """欠損値の詳細分析を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, Any]: 欠損セル総数や列別内訳などの分析結果。
        """
        missing_counts = df.isnull().sum()
        # 欠損値のパーセンテージ
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            'total_missing_cells': int(missing_counts.sum()),
            'columns_with_missing': {k: int(v) for k, v in missing_counts[missing_counts > 0].to_dict().items()},
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'critical_columns': []  # 50%以上欠損の列
        }
        
        # 重要な欠損パターンの特定
        for col, pct in missing_percentages.items():
            if pct >= 50:
                analysis['critical_columns'].append(col)
        
        return analysis
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """データ型の妥当性チェックを行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, str]: 列名とデータ型のマッピング。
        """
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """IQR 法による外れ値検出を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, int]: 列別の外れ値件数。
        """
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
                outlier_counts[col] = int(len(outliers))
        
        return outlier_counts
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """競馬データ特有のビジネスルール検証を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Tuple[List[str], List[str]]: 警告リストと推奨リスト。
        """
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
        """品質レポートサマリーをログ出力する。

        Args:
            report (Dict[str, Any]): 品質レポート辞書。
        """
        self.logger.info(f"📊 【{report['stage']}】品質サマリー:")
        self.logger.info(f"   📏 データ規模: {report['total_rows']:,}行 x {report['total_columns']}列")
        self.logger.info(f"   💾 メモリ使用量: {report['memory_usage_mb']:.1f}MB")
        self.logger.info(f"   ❓ 欠損セル数: {report['missing_values']['total_missing_cells']:,}")
        self.logger.info(f"   🔄 重複行数: {report['duplicates']:,}")
        
        if report['warnings']:
            self.logger.warning(f"   ⚠️ 警告: {len(report['warnings'])}件")
            for warning in report['warnings']:
                self.logger.warning(f"      • {warning}")

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

class SystemMonitor:
    """システム監視クラス（簡略版）"""
    
    def __init__(self):
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
    
    def log_system_status(self, stage_name: str):
        """システム状態をログに出力します。

        Args:
            stage_name (str): 出力対象の処理段階名。
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        logger.info(f"💻 [{stage_name}] システム状態:")
        logger.info(f"   ⏱️ 経過時間: {elapsed_time:.1f}秒")

def ensure_export_dirs():
    """出力用ディレクトリの存在確認と作成を行う。"""
    logger = logging.getLogger(__name__)
    
    dirs = [
        'export/BAC', 
        'export/SRB', 
        'export/SED', 
        'export/dataset',          # 実際のSED+SRB統合データ出力先
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
    """データ品質レポートを JSON として保存します。

    Args:
        quality_checker (DataQualityChecker): 品質レポートを保持するオブジェクト。
    """
    import json
    
    logger = logging.getLogger(__name__)
    report_path = Path('export/quality_reports/data_quality_report.json')
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_checker.quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 品質レポート保存: {report_path}")
        
    except Exception as e:
        logger.warning(f"⚠️ 品質レポート保存エラー: {str(e)}")

def display_deletion_statistics():
    """グレード欠損による削除統計を表示する。"""
    logger = logging.getLogger(__name__)
    
    try:
        def _count_csv_rows(file_path: Path) -> int:
            buffer_size = 1024 * 1024
            newline_count = 0
            last_char = b'\n'

            with file_path.open('rb') as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    newline_count += chunk.count(b'\n')
                    last_char = chunk[-1:]

            line_count = newline_count
            if last_char not in (b'\n', b''):
                line_count += 1

            return max(line_count - 1, 0)

        # ディレクトリパス
        sed_dir = Path('export/SED/formatted')
        bias_dir = Path('export/dataset')
        
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
            base_name = bias_file.stem.replace('_formatted_dataset', '')
            
            if base_name in sed_files_dict:
                sed_file = sed_files_dict[base_name]
                
                try:
                    # レコード数を数える（ヘッダー除く）
                    sed_count = _count_csv_rows(sed_file)
                    bias_count = _count_csv_rows(bias_file)
                    
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
        logger.info("📈 全体削除統計:")
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
    """欠損値処理ログのサマリーを生成する。"""
    logger = logging.getLogger(__name__)
    
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

def _parse_processing_log(log_file: Path) -> Optional[Dict[str, Any]]:
    """欠損値処理ログを解析して統計を生成します。

    Args:
        log_file (Path): 解析対象のログファイルパス。

    Returns:
        Optional[Dict[str, Any]]: ログ解析結果の統計情報。
    """
    logger = logging.getLogger(__name__)
    
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
        return {}
    
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
    """統計情報からサマリーレポートを生成します。

    Args:
        stats (Dict[str, Any]): ログ解析によって得られた統計情報。
        output_file (Path): 出力先のファイルパス。
    """
    
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

def process_race_data(exclude_turf: bool = False, turf_only: bool = False, 
                     enable_missing_value_handling: bool = True, enable_quality_check: bool = True) -> bool:
    """競馬レースデータの実務レベル処理（標準版）。

    計画書 Phase 0: データ整備の実装。
    
    この関数はRaceDataProcessorクラスのシンラッパーです。
    後方互換性のために残されています。

    Args:
        exclude_turf (bool): 芝コースを除外するかどうか。
        turf_only (bool): 芝コースのみを処理するかどうか。
        enable_missing_value_handling (bool): 戦略的欠損値処理を実行するかどうか。
        enable_quality_check (bool): データ品質チェックを実行するかどうか。

    Returns:
        bool: 成功時 ``True``、失敗時 ``False``。
    """
    logger.info("🏇 ■ 競馬レースデータの実務レベル処理を開始します ■")
    
    # システム監視開始
    monitor = SystemMonitor()
    
    # 処理オプションの確認
    if exclude_turf and turf_only:
        logger.error("❌ 芝コースを除外するオプションと芝コースのみを処理するオプションは同時に指定できません")
        return False
    
    # 通常の処理設定のログ出力
    logger.info("📋 処理設定:")
    logger.info(f"   🌱 芝コース除外: {'はい' if exclude_turf else 'いいえ'}")
    logger.info(f"   🌱 芝コースのみ: {'はい' if turf_only else 'いいえ'}")
    logger.info(f"   🔧 欠損値処理: {'有効' if enable_missing_value_handling else '無効'}")
    logger.info(f"   📈 品質チェック: {'有効' if enable_quality_check else '無効'}")
    
    # システムコンポーネントの初期化
    quality_checker = DataQualityChecker() if enable_quality_check else None
    
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
        logger.info("   📁 統合データ: export/dataset/")
        
        monitor.log_system_status("データ統合完了")
        
        # 5. データ品質チェック（統合後）
        if enable_quality_check:
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 0-5: データ品質チェック")
            logger.info("="*60)
            
            # サンプルファイルで品質チェック実行
            sample_files = list(Path('export/dataset').glob('*.csv'))
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
        if Path('export/dataset').exists():
            bias_files = list(Path('export/dataset').glob('*.csv'))
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
  最終的な成果物は `export/dataset/` ディレクトリに出力され、
  これが後続の分析スクリプト（例: analyze_horse_REQI.py）の入力となります。

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
    main_logger = logging.getLogger(__name__)
    main_logger.info("🚀 競馬レースデータ実務レベル処理を開始します")
    main_logger.info(f"📅 実行日時: {datetime.now()}")
    main_logger.info(f"🖥️ ログレベル: {args.log_level}")
    if log_file:
        main_logger.info(f"📝 ログファイル: {log_file}")

    try:
        success = process_race_data(
            exclude_turf=args.exclude_turf,
            turf_only=args.turf_only,
            enable_missing_value_handling=not args.no_missing_handling,
            enable_quality_check=not args.no_quality_check,
        )
    except Exception as e:
        main_logger.error(f"❌ 予期せぬエラー: {str(e)}")
        main_logger.error("🔧 スタックトレース:", exc_info=True)
        success = False

    if success:
        main_logger.info("🎉 実務レベルデータ処理が正常に完了しました")
        exit_code = 0
    else:
        main_logger.error("❌ データ処理が失敗しました")
        exit_code = 1

    main_logger.info(f"🏁 プロセス終了 (終了コード: {exit_code})")
    exit(exit_code)
"""
馬齢計算専用クラス
"""
import logging
import pandas as pd
import re
from typing import Optional

from ..config.column_names import ColumnNames

logger = logging.getLogger(__name__)


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


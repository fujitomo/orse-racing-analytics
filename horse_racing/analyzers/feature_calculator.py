"""
REQI特徴量計算専用モジュール
グレード・開催場・距離レベルの算出と競走経験質指数（REQI）計算を担当
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict

# 外部モジュールと定数
from ..core.weight_manager import WeightManager, get_global_weights
from ..data.constants.features import (
    GRADE_LEVEL_MAPPING,
    PRIZE_MONEY_THRESHOLDS,
    PRIZE_TO_GRADE_LEVEL,
    VENUE_GROUPS,
    VENUE_LEVELS,
    DISTANCE_THRESHOLDS,
    DISTANCE_LEVELS,
    DEFAULT_REQI_WEIGHTS,
    VENUE_CODE_GROUPS,
)

logger = logging.getLogger(__name__)


def _get_fallback_weights() -> Dict[str, float]:
    """レポート5.1.3節の固定重みを返します。"""
    logger.warning("⚠️ フォールバック重みを使用します: " + str(DEFAULT_REQI_WEIGHTS))
    return DEFAULT_REQI_WEIGHTS.copy()

def _calculate_individual_weights(df: pd.DataFrame) -> Dict[str, float]:
    """
    個別データから動的重みを計算します。
    (元 analyze_REQI.py/_calculate_individual_weights)
    """
    try:
        logger.info("🔍 個別動的重み計算を開始...")
        
        required_cols = ['馬名', 'grade_level', 'venue_level', 'distance_level', '着順']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ 個別重み計算に必要なカラムが不足。フォールバックします。")
            return _get_fallback_weights()

        df['is_win'] = (pd.to_numeric(df['着順'], errors='coerce') == 1).astype(int)
        
        horse_stats = df.groupby('馬名').agg(
            win_rate=('is_win', 'mean'),
            race_count=('grade_level', 'count')
        ).reset_index()
        
        horse_stats = horse_stats[horse_stats['race_count'] >= 6].copy()
        if len(horse_stats) < 100:
           logger.error(f"❌ サンプル数不足 ({len(horse_stats)}頭)。フォールバックします。")
           return _get_fallback_weights()
        
        for col in ['grade_level', 'venue_level', 'distance_level']:
            avg_feature = df.groupby('馬名')[col].mean().reset_index()
            horse_stats = horse_stats.merge(avg_feature.rename(columns={col: f'avg_{col}'}), on='馬名', how='left')
        
        from scipy.stats import pearsonr
        correlations = {}
        feature_mapping = {'avg_grade_level': 'grade', 'avg_venue_level': 'venue', 'avg_distance_level': 'distance'}
        
        for feature_col, name in feature_mapping.items():
            clean_data = horse_stats[[feature_col, 'win_rate']].dropna()
            if len(clean_data) > 1:
                corr, _ = pearsonr(clean_data[feature_col], clean_data['win_rate'])
                correlations[name] = {'squared': corr ** 2}
            else:
                correlations[name] = {'squared': 0}

        total_squared = sum(stats['squared'] for stats in correlations.values())
        if total_squared == 0:
           logger.warning("⚠️ 総寄与度が0。フォールバックします。")
           return _get_fallback_weights()

        weights = {f"{name}_weight": stats['squared'] / total_squared for name, stats in correlations.items()}
        
        final_weights = _get_fallback_weights()
        final_weights.update(weights)

        logger.info(f"✅ 個別動的重み計算完了: {final_weights}")
        return final_weights

    except Exception as e:
        logger.error(f"❌ 個別重み計算エラー: {e}。フォールバックします。")
        return _get_fallback_weights()


class FeatureCalculator:
    """REQI特徴量計算専用クラス（単一責任原則を遵守）。"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_reqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        REQI（競走経験質指数）を計算するメインのメソッド。
        特徴量レベルの計算から重み付けまでを一貫して行います。
        """
        self.logger.info("🎯 REQI計算処理を開始 (ベクトル化)...")
        df_copy = df.copy()

        # 1. 各レベルをベクトル化計算
        df_copy = self._calculate_feature_levels_vectorized(df_copy)

        # 2. REQI計算用の重みを取得
        if WeightManager.is_initialized():
            weights = get_global_weights()
            self.logger.info(f"✅ グローバル重みを使用: {weights}")
        else:
            self.logger.warning("⚠️ グローバル重み未初期化。データから個別動的重みを計算します。")
            # 個別重み計算のために、一度デフォルト重みで仮計算
            temp_df = self._apply_reqi_calculation(df_copy, DEFAULT_REQI_WEIGHTS)
            weights = _calculate_individual_weights(temp_df)
        
        # 3. 最終的なREQIを計算
        df_copy = self._apply_reqi_calculation(df_copy, weights)
        
        self.logger.info("✅ REQI計算処理完了。")
        self._log_feature_distributions(df_copy)
        
        return df_copy

    def _calculate_feature_levels_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ての特徴量レベルをベクトル化処理で計算します。"""
        self.logger.info("📊 グレード、場所、距離レベルをベクトル化計算中...")
        df = self._calculate_grade_level_vectorized(df)
        df = self._calculate_venue_level_vectorized(df)
        df = self._calculate_distance_level_vectorized(df)
        return df

    def _apply_reqi_calculation(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """REQIの最終的な重み付き計算を適用するヘルパー関数。"""
        w = {
            'grade': weights.get('grade_weight', DEFAULT_REQI_WEIGHTS['grade_weight']),
            'venue': weights.get('venue_weight', DEFAULT_REQI_WEIGHTS['venue_weight']),
            'distance': weights.get('distance_weight', DEFAULT_REQI_WEIGHTS['distance_weight']),
        }
        self.logger.info(f"📊 REQI計算式適用: race_level = {w['grade']:.3f}*grade + {w['venue']:.3f}*venue + {w['distance']:.3f}*distance")

        df['race_level'] = (
            w['grade'] * df['grade_level'] +
            w['venue'] * df['venue_level'] +
            w['distance'] * df['distance_level']
        )
        return df

    def _calculate_grade_level_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """グレードレベルをベクトル化処理で計算します。"""
        df['grade_level'] = 0.0
        grade_col = next((col for col in ['グレード_x', 'グレード_y', 'グレード'] if col in df.columns), None)
        
        if grade_col:
            grades = pd.to_numeric(df[grade_col], errors='coerce')
            df['grade_level'] = grades.map(GRADE_LEVEL_MAPPING).fillna(0.0)

        needs_fallback = df['grade_level'] == 0.0
        if needs_fallback.any():
            prize_col = next((col for col in ['1着賞金(1着算入賞金込み)', '1着賞金', '本賞金'] if col in df.columns), None)
            if prize_col:
                prizes = pd.to_numeric(df.loc[needs_fallback, prize_col], errors='coerce').fillna(0)
                conditions = [
                    prizes >= PRIZE_MONEY_THRESHOLDS["G1"], prizes >= PRIZE_MONEY_THRESHOLDS["G2"],
                    prizes >= PRIZE_MONEY_THRESHOLDS["G3"], prizes >= PRIZE_MONEY_THRESHOLDS["LISTED"],
                    prizes >= PRIZE_MONEY_THRESHOLDS["SPECIAL"],
                ]
                choices = [
                    PRIZE_TO_GRADE_LEVEL["G1"], PRIZE_TO_GRADE_LEVEL["G2"],
                    PRIZE_TO_GRADE_LEVEL["G3"], PRIZE_TO_GRADE_LEVEL["LISTED"],
                    PRIZE_TO_GRADE_LEVEL["SPECIAL"],
                ]
                df.loc[needs_fallback, 'grade_level'] = np.select(conditions, choices, default=0.0)
        return df

    def _calculate_venue_level_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """場所レベルをベクトル化処理で計算します。"""
        df['venue_level'] = 0.0
        if '場名' in df.columns:
            df['venue_level'] = np.select(
                [df['場名'].isin(VENUE_GROUPS["group1"]), df['場名'].isin(VENUE_GROUPS["group2"]), df['場名'].isin(VENUE_GROUPS["group3"])],
                [VENUE_LEVELS["group1"], VENUE_LEVELS["group2"], VENUE_LEVELS["group3"]],
                default=0.0
            )
        elif '場コード' in df.columns:
            codes = df['場コード'].astype(str).str.zfill(2)
            df['venue_level'] = np.select(
                [codes.isin(VENUE_CODE_GROUPS["group1"]), codes.isin(VENUE_CODE_GROUPS["group2"]), codes.isin(VENUE_CODE_GROUPS["group3"])],
                [VENUE_LEVELS["group1"], VENUE_LEVELS["group2"], VENUE_LEVELS["group3"]],
                default=0.0
            )
        return df

    def _calculate_distance_level_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離レベルをベクトル化処理で計算します。"""
        if '距離' in df.columns:
            dist = pd.to_numeric(df['距離'], errors='coerce').fillna(DISTANCE_THRESHOLDS["mile"])
            conditions = [
                dist <= DISTANCE_THRESHOLDS["sprint"], dist <= DISTANCE_THRESHOLDS["mile"],
                dist <= DISTANCE_THRESHOLDS["intermediate"], dist <= DISTANCE_THRESHOLDS["long"],
            ]
            choices = [
                DISTANCE_LEVELS["sprint"], DISTANCE_LEVELS["mile"],
                DISTANCE_LEVELS["intermediate"], DISTANCE_LEVELS["long"],
            ]
            df['distance_level'] = np.select(conditions, choices, default=DISTANCE_LEVELS["extended"])
        else:
            df['distance_level'] = DISTANCE_LEVELS["mile"]
        return df

    def _log_feature_distributions(self, df: pd.DataFrame) -> None:
        """特徴量の分布をログ出力します。"""
        feature_cols = ['grade_level', 'venue_level', 'distance_level', 'race_level']
        self.logger.info("✅ 特徴量計算完了:")
        for col in feature_cols:
            if col in df.columns:
                stats = df[col].describe()
                self.logger.info(f"  📊 {col} 分布: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    # --- 後方互換性のためのラッパーメソッド ---
    def calculate_accurate_feature_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """[互換性] 実際のCSVデータから特徴量を正確に計算します。"""
        self.logger.info("🔄 `calculate_accurate_feature_levels` は新しい `calculate_reqi` を呼び出します。")
        return self.calculate_reqi(df)

    def calculate_race_level_with_position_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """[互換性] 複勝実績を反映したREQI特徴量を算出します。"""
        self.logger.info("🔄 `calculate_race_level_with_position_weights` は新しい `calculate_reqi` を呼び出します。")
        return self.calculate_reqi(df)


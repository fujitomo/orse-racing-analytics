"""
重み初期化サービス
グローバル重みの計算とキャッシュ管理を提供
"""

import logging
import pandas as pd
from typing import Optional
from .data_cache_service import get_global_cache
from horse_racing.core.weight_manager import WeightManager

logger = logging.getLogger(__name__)


class WeightInitializationService:
    """重み初期化を担当するサービスクラス。"""
    
    TRAINING_START_YEAR = 2010
    TRAINING_END_YEAR = 2020
    
    def __init__(self):
        """重み初期化サービスを初期化します。"""
        self.cache = get_global_cache()
        self.logger = logging.getLogger(__name__)
    
    def initialize_weights(self, combined_df: pd.DataFrame, 
                          feature_calculator) -> bool:
        """グローバル重みを初期化し、関連データをキャッシュします。
        
        Args:
            combined_df (pd.DataFrame): 全期間のデータ。
            feature_calculator: 特徴量計算器インスタンス。
            
        Returns:
            bool: 初期化に成功した場合は True。
        """
        try:
            self.logger.info("🎯 グローバル重み初期化開始...")
            
            # 年の範囲を確認
            if '年' not in combined_df.columns:
                self.logger.warning("⚠️ 年列が見つかりません。全データを使用します")
                df = combined_df
            else:
                year_range = f"{combined_df['年'].min()}-{combined_df['年'].max()}年"
                self.logger.info(f"📅 全データ期間: {year_range}")
                
                # 訓練期間データを抽出
                training_data = combined_df[
                    (combined_df['年'] >= self.TRAINING_START_YEAR) & 
                    (combined_df['年'] <= self.TRAINING_END_YEAR)
                ]
                
                if len(training_data) > 0:
                    df = training_data
                    training_year_range = f"{training_data['年'].min()}-{training_data['年'].max()}年"
                    self.logger.info(f"📊 重み計算用訓練期間データ: {len(training_data):,}行 ({training_year_range})")
                else:
                    self.logger.warning("⚠️ 訓練期間データが見つかりませんでした")
                    df = combined_df
                
                self.logger.info(f"📊 全データ期間: {len(combined_df):,}行 ({combined_df['年'].min()}-{combined_df['年'].max()}年)")
            
            # 特徴量レベル列を計算（重み計算のため）
            self.logger.info("🧮 重み計算用特徴量レベル列を計算中（訓練期間）...")
            df = feature_calculator.calculate_accurate_feature_levels(df)
            
            # グローバルキャッシュに保存
            self.cache.set_combined_data(combined_df)
            
            # 全データで特徴量レベル列を計算
            self.logger.info("🧮 全データで特徴量レベル列を計算中（期間別分析用）...")
            df_all_features = feature_calculator.calculate_accurate_feature_levels(combined_df)
            
            # REQI特徴量も事前計算
            self.logger.info("🚀 競走経験質指数（REQI）特徴量を事前計算中...")
            feature_levels = feature_calculator.calculate_race_level_with_position_weights(df_all_features)
            self.cache.set_feature_levels(feature_levels)
            
            self.logger.info("💾 計算済みデータをグローバルキャッシュに保存しました")
            
            combined_cached = self.cache.get_combined_data(copy=False)
            feature_cached = self.cache.get_feature_levels(copy=False)
            
            if combined_cached is not None:
                self.logger.info(f"📊 グローバルデータ: {len(combined_cached):,}行（全期間）")
            self.logger.info(f"📊 重み計算用データ: {len(df):,}行（訓練期間{self.TRAINING_START_YEAR}-{self.TRAINING_END_YEAR}年）")
            if feature_cached is not None:
                self.logger.info(f"📊 期間別分析用データ: {len(feature_cached):,}行（全期間）")
            self.logger.info("🚀 競走経験質指数（REQI）特徴量も事前計算済み（期間別分析高速化）")
            
            # グローバル重みを初期化
            weights = WeightManager.initialize_from_training_data(df)
            self.logger.info(f"✅ グローバル重み設定完了: {weights}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ グローバル重み初期化エラー: {str(e)}")
            return False

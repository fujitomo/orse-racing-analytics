"""
データキャッシュサービス
グローバルキャッシュの統一インターフェイスを提供
"""

import logging
import pandas as pd
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataCacheService:
    """グローバルに共有するデータフレームを管理するキャッシュサービス。
    
    Attributes:
        raw_data: 生データ（CSV読み込み直後）。
        combined_data: 統合済みデータ（全期間）。
        feature_levels: 特徴量計算済みデータ。
    """

    raw_data: Optional[pd.DataFrame] = None
    combined_data: Optional[pd.DataFrame] = None
    feature_levels: Optional[pd.DataFrame] = None

    def reset(self) -> None:
        """キャッシュ済みデータをすべてクリアします。"""
        self.raw_data = None
        self.combined_data = None
        self.feature_levels = None

    def set_raw_data(self, df: pd.DataFrame, copy: bool = True) -> None:
        """生データをキャッシュに保存します。
        
        Args:
            df (pd.DataFrame): 保存対象のデータフレーム。
            copy (bool): コピーを作成するかどうか。
        """
        self.raw_data = df.copy() if copy else df

    def get_raw_data(self, copy: bool = True) -> Optional[pd.DataFrame]:
        """生データのキャッシュを取得します。
        
        Args:
            copy (bool): コピーを返すかどうか。
            
        Returns:
            Optional[pd.DataFrame]: キャッシュされた生データ。存在しない場合は None。
        """
        if self.raw_data is None:
            return None
        return self.raw_data.copy() if copy else self.raw_data

    def set_combined_data(self, df: pd.DataFrame, copy: bool = True) -> None:
        """統合済みデータをキャッシュに保存します。
        
        Args:
            df (pd.DataFrame): 保存対象のデータフレーム。
            copy (bool): コピーを作成するかどうか。
        """
        self.combined_data = df.copy() if copy else df

    def get_combined_data(self, copy: bool = True) -> Optional[pd.DataFrame]:
        """統合済みデータのキャッシュを取得します。
        
        Args:
            copy (bool): コピーを返すかどうか。
            
        Returns:
            Optional[pd.DataFrame]: キャッシュされた統合データ。存在しない場合は None。
        """
        if self.combined_data is None:
            return None
        return self.combined_data.copy() if copy else self.combined_data

    def set_feature_levels(self, df: pd.DataFrame, copy: bool = True) -> None:
        """特徴量計算済みデータをキャッシュに保存します。
        
        Args:
            df (pd.DataFrame): 保存対象のデータフレーム。
            copy (bool): コピーを作成するかどうか。
        """
        self.feature_levels = df.copy() if copy else df

    def get_feature_levels(self, copy: bool = True) -> Optional[pd.DataFrame]:
        """特徴量計算済みデータのキャッシュを取得します。
        
        Args:
            copy (bool): コピーを返すかどうか。
            
        Returns:
            Optional[pd.DataFrame]: キャッシュされた特徴量データ。存在しない場合は None。
        """
        if self.feature_levels is None:
            return None
        return self.feature_levels.copy() if copy else self.feature_levels

    def has_feature_levels(self) -> bool:
        """特徴量キャッシュが存在するか判定します。
        
        Returns:
            bool: 存在する場合 True。
        """
        return self.feature_levels is not None

    def has_combined_data(self) -> bool:
        """統合済みデータキャッシュが存在するか判定します。
        
        Returns:
            bool: 存在する場合 True。
        """
        return self.combined_data is not None

    def has_raw_data(self) -> bool:
        """生データキャッシュが存在するか判定します。
        
        Returns:
            bool: 存在する場合 True。
        """
        return self.raw_data is not None


# グローバルキャッシュインスタンス
_GLOBAL_CACHE = DataCacheService()


def get_global_cache() -> DataCacheService:
    """グローバルキャッシュインスタンスを取得します。
    
    Returns:
        DataCacheService: グローバルキャッシュ。
    """
    return _GLOBAL_CACHE

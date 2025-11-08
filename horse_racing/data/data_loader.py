"""
データ読み込み専用モジュール
CSV読み込み・統合・キャッシュ管理を担当
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GlobalDataCache:
    """グローバルに共有するデータフレームを管理するキャッシュ。
    
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
GLOBAL_DATA_CACHE = GlobalDataCache()


class DataLoader:
    """CSV読み込みとキャッシュ管理を担当するクラス（SRP遵守）。"""
    
    def __init__(self, cache: Optional[GlobalDataCache] = None):
        """データローダーを初期化します。
        
        Args:
            cache (GlobalDataCache, optional): 使用するキャッシュインスタンス。
        """
        self.cache = cache or GLOBAL_DATA_CACHE
        self.logger = logging.getLogger(__name__)
    
    def load_csv_files(self, input_path: str, encoding: str = 'utf-8', 
                       use_cache: bool = True) -> pd.DataFrame:
        """CSVファイルを読み込み、結果をキャッシュします。
        
        Args:
            input_path (str): CSVファイル、またはそれらを含むディレクトリのパス。
            encoding (str): 読み込み時に使用する文字エンコーディング。
            use_cache (bool): キャッシュを利用するかどうか。
            
        Returns:
            pd.DataFrame: 入力ソースを結合した生データフレーム。
        """
        # キャッシュチェック
        if use_cache:
            cached_raw = self.cache.get_raw_data()
            if cached_raw is not None:
                self.logger.info("💾 グローバルキャッシュから生データを取得中...")
                return cached_raw
        
        self.logger.info("📖 全CSVファイルを初回読み込み中...")
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            return self._load_single_file(input_path_obj, encoding)
        else:
            return self._load_directory(input_path_obj, encoding)
    
    def _load_single_file(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """単一CSVファイルを読み込みます。
        
        Args:
            file_path (Path): ファイルパス。
            encoding (str): 文字エンコーディング。
            
        Returns:
            pd.DataFrame: 読み込んだデータフレーム。
        """
        df = pd.read_csv(file_path, encoding=encoding)
        self.logger.info(f"📊 単一ファイル読み込み: {len(df):,}行")
        self.cache.set_raw_data(df)
        return df
    
    def _load_directory(self, dir_path: Path, encoding: str) -> pd.DataFrame:
        """ディレクトリ内の全CSVファイルを読み込み統合します。
        
        Args:
            dir_path (Path): ディレクトリパス。
            encoding (str): 文字エンコーディング。
            
        Returns:
            pd.DataFrame: 統合されたデータフレーム。
        """
        csv_files = list(dir_path.glob("*.csv"))
        if not csv_files:
            self.logger.error(f"❌ CSVファイルが見つかりません: {dir_path}")
            return pd.DataFrame()
        
        self.logger.info(f"📊 全CSVファイルを統合中... ({len(csv_files)}ファイル)")
        all_dfs = []
        
        for i, csv_file in enumerate(csv_files):
            try:
                df_temp = pd.read_csv(csv_file, encoding=encoding)
                all_dfs.append(df_temp)
                
                # 進捗表示（100ファイルごと）
                if (i + 1) % 100 == 0:
                    self.logger.info(f"   読み込み進捗: {i + 1}/{len(csv_files)}ファイル")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ ファイル読み込みエラー（スキップ）: {csv_file.name} - {str(e)}")
                continue
        
        if all_dfs:
            self.logger.info("🔄 データフレーム統合中...")
            combined_df = pd.concat(all_dfs, ignore_index=True)
            self.logger.info(f"✅ 統合完了: {len(combined_df):,}行のデータ")
            
            # キャッシュに保存
            self.cache.set_raw_data(combined_df)
            self.logger.info("💾 生データをグローバルキャッシュに保存しました")
            self.logger.info(f"🔍 キャッシュ確認: raw_data_cached={self.cache.has_raw_data()}")
            return combined_df
        else:
            self.logger.error("❌ 有効なCSVファイルが見つかりませんでした")
            return pd.DataFrame()


# 後方互換性のためのユーティリティ関数
def cache_raw_data(df: pd.DataFrame, copy: bool = True) -> None:
    """生データをグローバルキャッシュに保存します（後方互換用）。
    
    Args:
        df (pd.DataFrame): 保存対象のデータフレーム。
        copy (bool): コピーを作成するかどうか。
    """
    GLOBAL_DATA_CACHE.set_raw_data(df, copy=copy)


def cache_combined_data(df: pd.DataFrame, copy: bool = True) -> None:
    """統合済みデータをグローバルキャッシュに保存します（後方互換用）。
    
    Args:
        df (pd.DataFrame): 保存対象のデータフレーム。
        copy (bool): コピーを作成するかどうか。
    """
    GLOBAL_DATA_CACHE.set_combined_data(df, copy=copy)


def cache_feature_levels(df: pd.DataFrame, copy: bool = True) -> None:
    """特徴量計算済みデータをグローバルキャッシュに保存します（後方互換用）。
    
    Args:
        df (pd.DataFrame): 保存対象のデータフレーム。
        copy (bool): コピーを作成するかどうか。
    """
    GLOBAL_DATA_CACHE.set_feature_levels(df, copy=copy)


def get_cached_raw_data(copy: bool = True) -> Optional[pd.DataFrame]:
    """キャッシュしている生データを取得します（後方互換用）。
    
    Args:
        copy (bool): コピーを返すかどうか。
        
    Returns:
        Optional[pd.DataFrame]: キャッシュされた生データ。
    """
    return GLOBAL_DATA_CACHE.get_raw_data(copy=copy)


def get_cached_combined_data(copy: bool = True) -> Optional[pd.DataFrame]:
    """キャッシュしている統合済みデータを取得します（後方互換用）。
    
    Args:
        copy (bool): コピーを返すかどうか。
        
    Returns:
        Optional[pd.DataFrame]: キャッシュされた統合データ。
    """
    return GLOBAL_DATA_CACHE.get_combined_data(copy=copy)


def get_cached_feature_levels(copy: bool = True) -> Optional[pd.DataFrame]:
    """キャッシュしている特徴量計算済みデータを取得します（後方互換用）。
    
    Args:
        copy (bool): コピーを返すかどうか。
        
    Returns:
        Optional[pd.DataFrame]: キャッシュされた特徴量データ。
    """
    return GLOBAL_DATA_CACHE.get_feature_levels(copy=copy)


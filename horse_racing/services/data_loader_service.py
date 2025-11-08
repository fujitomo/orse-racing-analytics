"""
データ読み込みサービス
CSV読み込みとキャッシュ統合を提供
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from .data_cache_service import get_global_cache

logger = logging.getLogger(__name__)


class DataLoaderService:
    """CSV読み込みとキャッシュ管理を担当するサービスクラス。"""
    
    def __init__(self):
        """データローダーサービスを初期化します。"""
        self.cache = get_global_cache()
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

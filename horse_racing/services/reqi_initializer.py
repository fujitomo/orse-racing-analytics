"""
REQI初期化統合サービス
データ読み込み・特徴量計算・重み初期化を統合管理
"""

import logging
import pandas as pd
from typing import Optional, Callable
from .data_loader_service import DataLoaderService
from .weight_initialization_service import WeightInitializationService
from horse_racing.analyzers.feature_calculator import FeatureCalculator

logger = logging.getLogger(__name__)


class REQIInitializer:
    """REQI分析の初期化を統合管理するクラス。"""
    
    def __init__(self):
        """REQI初期化サービスを初期化します。"""
        self.loader = DataLoaderService()
        self.feature_calculator = FeatureCalculator()
        self.weight_service = WeightInitializationService()
        self.logger = logging.getLogger(__name__)
    
    def initialize_from_args(self, args, 
                           feature_calc_func: Optional[Callable] = None,
                           reqi_calc_func: Optional[Callable] = None) -> bool:
        """コマンドライン引数からREQI分析を初期化します。
        
        Args:
            args: argparse.Namespace オブジェクト。
            feature_calc_func: 特徴量計算関数（互換性用）。
            reqi_calc_func: REQI計算関数（互換性用）。
            
        Returns:
            bool: 初期化に成功した場合は True。
        """
        try:
            self.logger.info("🎯 REQI分析初期化開始...")
            
            # データ読み込み
            combined_df = self.loader.load_csv_files(args.input_path, args.encoding)
            
            # 日付フィルタ適用
            combined_df = self._filter_by_date_range(
                combined_df, 
                getattr(args, 'start_date', None), 
                getattr(args, 'end_date', None)
            )
            
            if combined_df.empty:
                self.logger.error("❌ データが空です")
                return False
            
            # 重み初期化
            success = self.weight_service.initialize_weights(combined_df, self.feature_calculator)
            
            if success:
                self.logger.info("✅ REQI分析初期化完了")
            else:
                self.logger.warning("⚠️ REQI分析初期化に一部失敗しました")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ REQI分析初期化エラー: {str(e)}")
            return False
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: Optional[str], 
                              end_date: Optional[str]) -> pd.DataFrame:
        """日付範囲でデータをフィルタします。
        
        Args:
            df (pd.DataFrame): 入力データ。
            start_date (Optional[str]): 開始日（YYYYMMDD形式）。
            end_date (Optional[str]): 終了日（YYYYMMDD形式）。
            
        Returns:
            pd.DataFrame: フィルタ後のデータ。
        """
        if df is None or len(df) == 0:
            return df
        
        # 日付列がある場合
        if '年月日' in df.columns:
            df_copy = df.copy()
            try:
                df_copy['__date'] = pd.to_datetime(df_copy['年月日'], format='%Y%m%d', errors='coerce')
            except Exception:
                df_copy['__date'] = pd.to_datetime(df_copy['年月日'], errors='coerce')
            
            mask = pd.Series(True, index=df_copy.index)
            
            if start_date:
                try:
                    from datetime import datetime
                    sd = datetime.strptime(start_date, '%Y%m%d')
                    mask &= df_copy['__date'] >= sd
                except Exception:
                    pass
            
            if end_date:
                try:
                    from datetime import datetime
                    ed = datetime.strptime(end_date, '%Y%m%d')
                    mask &= df_copy['__date'] <= ed
                except Exception:
                    pass
            
            filtered = df_copy.loc[mask].drop(columns=['__date'])
            if len(filtered) != len(df):
                self.logger.info(f"🧹 日付フィルタ適用: {len(df):,} → {len(filtered):,}")
            return filtered
        
        # 年列がある場合
        if '年' in df.columns:
            df_copy = df.copy()
            mask = pd.Series(True, index=df_copy.index)
            
            if start_date and len(start_date) >= 4:
                try:
                    start_year = int(start_date[:4])
                    mask &= pd.to_numeric(df_copy['年'], errors='coerce') >= start_year
                except Exception:
                    pass
            
            if end_date and len(end_date) >= 4:
                try:
                    end_year = int(end_date[:4])
                    mask &= pd.to_numeric(df_copy['年'], errors='coerce') <= end_year
                except Exception:
                    pass
            
            filtered = df_copy.loc[mask]
            if len(filtered) != len(df):
                self.logger.info(f"🧹 年フィルタ適用: {len(df):,} → {len(filtered):,}")
            return filtered
        
        return df

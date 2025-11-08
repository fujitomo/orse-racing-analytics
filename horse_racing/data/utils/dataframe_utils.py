# horse_racing/data/utils/dataframe_utils.py
"""
Pandas DataFrame用のユーティリティ関数
"""
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def filter_by_date_range(df: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """日付範囲で DataFrame をフィルタする（``年月日`` または ``年`` ベース）。

    Args:
        df (pd.DataFrame): 入力データ。
        start_date (str | None): ``YYYYMMDD`` 形式の開始日（含む）。
        end_date (str | None): ``YYYYMMDD`` 形式の終了日（含む）。

    Returns:
        pd.DataFrame: フィルタ後のデータフレーム。
    """
    try:
        if df is None or len(df) == 0:
            return df
        
        df_copy = df.copy()
        
        # 日付列がある場合はそれを優先
        if '年月日' in df_copy.columns:
            df_copy['__date'] = pd.to_datetime(df_copy['年月日'], format='%Y%m%d', errors='coerce')
            mask = pd.Series(True, index=df_copy.index)
            if start_date:
                mask &= df_copy['__date'] >= datetime.strptime(start_date, '%Y%m%d')
            if end_date:
                mask &= df_copy['__date'] <= datetime.strptime(end_date, '%Y%m%d')
            
            filtered = df_copy.loc[mask].drop(columns=['__date'])
            if len(filtered) != len(df):
                logger.info(f"🧹 日付フィルタ適用(年月日): {len(df):,} → {len(filtered):,}")
            return filtered

        # 年列がある場合は年でフィルタ
        if '年' in df_copy.columns:
            mask = pd.Series(True, index=df_copy.index)
            if start_date and len(start_date) >= 4:
                start_year = int(start_date[:4])
                mask &= pd.to_numeric(df_copy['年'], errors='coerce') >= start_year
            if end_date and len(end_date) >= 4:
                end_year = int(end_date[:4])
                mask &= pd.to_numeric(df_copy['年'], errors='coerce') <= end_year
            
            filtered = df_copy.loc[mask]
            if len(filtered) != len(df):
                logger.info(f"🧹 年フィルタ適用(年): {len(df):,} → {len(filtered):,}")
            return filtered
            
        # フィルタ不可
        return df
    except Exception as e:
        logger.warning(f"⚠️ 日付フィルタ適用中に例外: {str(e)}")
        return df

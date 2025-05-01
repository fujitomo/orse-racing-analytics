"""
基本統計分析モジュール
"""
from typing import Dict, Any
import pandas as pd
from ..base.analyzer import BaseAnalyzer

class BasicStatsAnalyzer(BaseAnalyzer):
    """基本統計分析クラス"""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """基本統計量の計算"""
        stats = {
            "race_count": len(df),
            "horse_count": df["馬名"].nunique(),
            "avg_distance": df["距離"].mean(),
            "avg_prize": df["本賞金"].mean() if "本賞金" in df.columns else None
        }
        return stats 
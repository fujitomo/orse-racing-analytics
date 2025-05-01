"""
型定義モジュール
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LoaderConfig:
    """データローダーの設定"""
    input_path: str
    encoding: str = "utf-8"
    use_cache: bool = True
    cache_dir: Optional[str] = None

class RaceAnalysisError(Exception):
    """レース分析固有のエラー"""
    pass

class DataLoadError(RaceAnalysisError):
    """データ読み込みエラー"""
    pass 
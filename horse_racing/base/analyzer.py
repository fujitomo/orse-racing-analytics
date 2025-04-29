import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """分析の設定を保持するデータクラス"""
    input_path: str
    output_dir: str = 'export/analysis'
    date_str: str = ''
    min_races: int = 6
    confidence_level: float = 0.95

class BaseAnalyzer(ABC):
    """基底分析クラス"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df = pd.DataFrame()
        self.stats = {}
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """出力ディレクトリの設定"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """データ読み込みの抽象メソッド"""
        pass

    @abstractmethod
    def preprocess_data(self) -> pd.DataFrame:
        """データ前処理の抽象メソッド"""
        pass

    @abstractmethod
    def calculate_feature(self) -> pd.DataFrame:
        """特徴量計算の抽象メソッド"""
        pass

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """分析実行の抽象メソッド"""
        pass

    @abstractmethod
    def visualize(self) -> None:
        """可視化の抽象メソッド"""
        pass

    def run(self) -> Dict[str, Any]:
        """分析の実行"""
        self.df = self.load_data()
        self.df = self.preprocess_data()
        self.df = self.calculate_feature()
        self.stats = self.analyze()
        self.visualize()
        return self.stats

    def save_results(self, prefix: str) -> None:
        """分析結果の保存"""
        if not self.stats:
            return

        for name, data in self.stats.items():
            if isinstance(data, pd.DataFrame):
                output_path = self.output_dir / f"{prefix}_{name}_{self.config.date_str}.csv"
                data.to_csv(output_path, index=False, encoding="utf-8")

    @staticmethod
    def calculate_confidence_interval(
        correlation: float,
        n: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """信頼区間の計算"""
        from scipy import stats
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        z_lower = z - z_score * se
        z_upper = z + z_score * se
        return np.tanh(z_lower), np.tanh(z_upper)

    def normalize_values(self, series: pd.Series) -> pd.Series:
        """値の正規化（0-10のスケール）"""
        if series.max() == series.min():
            return series.map(lambda x: 5.0)
        return (series - series.min()) / (series.max() - series.min()) * 10 
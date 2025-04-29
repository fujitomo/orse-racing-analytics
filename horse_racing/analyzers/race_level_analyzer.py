from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from ..base.analyzer import BaseAnalyzer, AnalysisConfig
from ..data.loader import RaceDataLoader
from ..visualization.plotter import RacePlotter

class RaceLevelAnalyzer(BaseAnalyzer):
    """レースレベル分析クラス"""

    # グレード定義
    GRADE_LEVELS = {
        1: {"name": "G1", "weight": 10.0, "base_level": 9},
        2: {"name": "G2", "weight": 8.0, "base_level": 8},
        3: {"name": "G3", "weight": 7.0, "base_level": 7},
        4: {"name": "重賞", "weight": 6.0, "base_level": 6},
        5: {"name": "特別", "weight": 5.0, "base_level": 5},
        6: {"name": "L", "weight": 5.5, "base_level": 5.5}
    }

    # レースレベル計算の重み付け定義
    LEVEL_WEIGHTS = {
        "grade_weight": 0.60,
        "prize_weight": 0.40,
        "field_size_weight": 0.10,
        "competition_weight": 0.20,
    }

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.plotter = RacePlotter(self.output_dir)
        self.loader = RaceDataLoader(config.input_path)

    def load_data(self) -> pd.DataFrame:
        """データの読み込み"""
        return self.loader.load()

    def preprocess_data(self) -> pd.DataFrame:
        """データの前処理"""
        df = self.df.copy()

        # 芝レースのみをフィルタリング
        df = df[df["芝ダ障害コード"] == "芝"]

        # レース回数3回以上の馬のみを抽出
        race_counts = df['馬名'].value_counts()
        horses_with_min_races = race_counts[race_counts >= self.config.min_races].index
        df = df[df['馬名'].isin(horses_with_min_races)]

        # グレードの判定
        if "グレード" not in df.columns:
            df["グレード"] = df.apply(self._determine_grade, axis=1)

        # 必要な列の抽出と型変換
        required_columns = [
            "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
            "レース名", "種別", "芝ダ障害コード", "馬番", "グレード", "本賞金"
        ]
        df = df[required_columns]

        # データ型の変換
        df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df["グレード"] = pd.to_numeric(df["グレード"], errors="coerce")
        df["種別"] = pd.to_numeric(df["種別"], errors="coerce")
        df["1着賞金"] = df["本賞金"]  # 本賞金を1着賞金として使用

        return df

    def calculate_feature(self) -> pd.DataFrame:
        """レースレベルの計算"""
        df = self.df.copy()
        df["race_level"] = 0.0
        df["is_win"] = df["着順"] == 1
        df["is_placed"] = df["着順"] <= 3

        # 基本レベルの計算
        grade_level = self._calculate_grade_level(df)
        prize_level = self._calculate_prize_level(df)

        # 重み付け合算
        df["race_level"] = (
            grade_level * self.LEVEL_WEIGHTS["grade_weight"] +
            prize_level * self.LEVEL_WEIGHTS["prize_weight"]
        )

        # レースレベルの正規化
        df["race_level"] = self.normalize_values(df["race_level"])

        # 距離による補正
        df = self._apply_distance_correction(df)

        return df

    def analyze(self) -> Dict[str, Any]:
        """分析の実行"""
        # 馬ごとの基本統計
        horse_stats = self._calculate_horse_stats()
        
        # グレード別の統計
        grade_stats = self._calculate_grade_stats()
        
        # 相関分析
        correlation_stats = self._perform_correlation_analysis(horse_stats)
        
        return {
            "horse_stats": horse_stats,
            "grade_stats": grade_stats,
            "correlation_stats": correlation_stats
        }

    def visualize(self) -> None:
        """分析結果の可視化"""
        if not self.stats:
            return

        horse_stats = self.stats["horse_stats"]
        grade_stats = self.stats["grade_stats"]
        correlation_stats = self.stats["correlation_stats"]

        # グレード別統計の可視化
        self.plotter.plot_grade_stats(grade_stats, self.GRADE_LEVELS)

        # 相関分析の可視化
        if correlation_stats:
            self.plotter.plot_correlation_analysis(
                horse_stats,
                correlation_stats["correlation"],
                correlation_stats["model"],
                correlation_stats["r2"],
                "レースレベル"
            )

        # レースレベルの分布分析
        self.plotter.plot_distribution_analysis(
            self.df[self.df["is_win"]],
            "race_level"
        )

        # トレンド分析
        level_year_stats = self._calculate_level_year_stats()
        self.plotter.plot_trend_analysis(level_year_stats, "レースレベル")

        # 距離とレースレベルのヒートマップ
        distance_level_pivot = self._calculate_distance_level_pivot()
        self.plotter.plot_heatmap(
            distance_level_pivot,
            "距離別・レースレベル別の勝率ヒートマップ",
            "distance_level_heatmap.png"
        )

    def _determine_grade(self, row: pd.Series) -> int:
        """グレードの判定"""
        race_name = str(row["レース名"]) if pd.notna(row["レース名"]) else ""
        race_type = row["種別"] if pd.notna(row["種別"]) else 99

        # レース名からグレードを判定
        if "G1" in race_name or "Ｇ１" in race_name:
            return 1
        elif "G2" in race_name or "Ｇ２" in race_name:
            return 2
        elif "G3" in race_name or "Ｇ３" in race_name:
            return 3
        elif "重賞" in race_name:
            return 4
        elif "L" in race_name or "Ｌ" in race_name:
            return 6

        # 賞金からグレードを判定
        if "本賞金" in row.index:
            prize_grade = self._determine_grade_by_prize(row)
            if prize_grade is not None:
                return prize_grade

        # 種別コードに基づく判定
        if race_type in [11, 12]:  # 2歳・3歳戦
            return 5
        elif race_type in [13, 14]:  # 3歳以上・4歳以上
            return 5 if "特別" in race_name else 5
        elif race_type == 20:  # 障害
            if "J.G1" in race_name:
                return 1
            elif "J.G2" in race_name:
                return 2
            elif "J.G3" in race_name:
                return 3
            else:
                return 5
        else:
            return 5

    def _determine_grade_by_prize(self, row: pd.Series) -> Optional[int]:
        """賞金額からグレードを判定"""
        prize = row["本賞金"] if pd.notna(row["本賞金"]) else None

        if prize is None:
            return None

        if prize >= 10000:    # 1億円以上
            return 1
        elif prize >= 7000:   # 7000万円以上
            return 2
        elif prize >= 4500:   # 4500万円以上
            return 3
        elif prize >= 3500:   # 3500万円以上
            return 4
        elif prize >= 2000:   # 2000万円以上
            return 6
        else:
            return 5

    def _calculate_grade_level(self, df: pd.DataFrame) -> pd.Series:
        """グレードに基づくレベルを計算"""
        grade_level = df["グレード"].map(
            lambda x: self.GRADE_LEVELS[x]["base_level"] if pd.notna(x) else 5.0
        )

        for grade, values in self.GRADE_LEVELS.items():
            mask = df["グレード"] == grade
            grade_level.loc[mask & df["is_win"]] += values["weight"]
            grade_level.loc[mask & df["is_placed"] & ~df["is_win"]] += values["weight"] * 0.5

        return self.normalize_values(grade_level)

    def _calculate_prize_level(self, df: pd.DataFrame) -> pd.Series:
        """賞金に基づくレベルを計算"""
        prize_level = np.log1p(df["1着賞金"]) / np.log1p(df["1着賞金"].max()) * 9.95
        return self.normalize_values(prize_level)

    def _apply_distance_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離による補正を適用"""
        distance_weights = {
            (0, 1400): 1.02,     # スプリント
            (1401, 1800): 1.12,  # マイル
            (1801, 2000): 1.27,  # 中距離
            (2001, 2400): 1.37,  # 中長距離
            (2401, 9999): 1.12,  # 長距離
        }

        for (min_dist, max_dist), weight in distance_weights.items():
            mask = (df["距離"] >= min_dist) & (df["距離"] <= max_dist)
            df.loc[mask, "race_level"] *= weight

        # 2000m特別ボーナス
        mask_2000m = (df["距離"] >= 1900) & (df["距離"] <= 2100)
        df.loc[mask_2000m, "race_level"] *= 1.09

        return df

    def _calculate_horse_stats(self) -> pd.DataFrame:
        """馬ごとの基本統計を計算"""
        # 馬ごとの基本統計
        horse_stats = self.df.groupby("馬名").agg({
            "race_level": ["max", "mean"],
            "is_win": "sum",
            "is_placed": "sum",
            "着順": "count",
            "グレード": lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
        }).reset_index()

        # カラム名の整理
        horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数", "主戦グレード"]
        
        # レース回数が3回以上の馬のみをフィルタリング
        min_races = self.config.min_races if hasattr(self.config, 'min_races') else 6
        horse_stats = horse_stats[horse_stats["出走回数"] >= min_races]
        
        # 勝率と複勝率の計算
        horse_stats["win_rate"] = horse_stats["勝利数"] / horse_stats["出走回数"]
        horse_stats["place_rate"] = horse_stats["複勝数"] / horse_stats["出走回数"]
        
        return horse_stats

    def _calculate_grade_stats(self) -> pd.DataFrame:
        """グレード別の統計を計算"""
        grade_stats = self.df.groupby("グレード").agg({
            "is_win": ["mean", "count"],
            "is_placed": "mean",
            "race_level": ["mean", "std"]
        }).reset_index()

        grade_stats.columns = [
            "グレード", "勝率", "レース数", "複勝率",
            "平均レベル", "レベル標準偏差"
        ]

        return grade_stats

    def _perform_correlation_analysis(self, horse_stats: pd.DataFrame) -> Dict[str, Any]:
        """相関分析を実行"""
        analysis_data = horse_stats.dropna(subset=['最高レベル', 'win_rate'])
        
        if len(analysis_data) == 0:
            return {}

        # 相関係数の計算
        correlation = np.corrcoef(
            analysis_data['最高レベル'],
            analysis_data['win_rate']
        )[0, 1]

        # 回帰分析
        X = analysis_data['最高レベル'].values.reshape(-1, 1)
        y = analysis_data['win_rate'].values
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        return {
            "correlation": correlation,
            "model": model,
            "r2": r2
        }

    def _calculate_level_year_stats(self) -> pd.DataFrame:
        """レースレベル別・年別の統計を計算"""
        df = self.df.copy()
        
        # NA値を0で置換し、無限大の値を最大有限値で置換
        df['race_level'] = df['race_level'].fillna(0)
        df['race_level'] = df['race_level'].replace([np.inf, -np.inf], df['race_level'].replace([np.inf, -np.inf], np.nan).max())
        
        # レースレベル区分を計算（小数点以下を四捨五入して整数に）
        df['レースレベル区分'] = np.round(df['race_level']).astype(int)
        
        # レースレベル別・年別の勝率推移を計算
        level_year_stats = df.groupby(
            ["レースレベル区分", "年"]
        )["is_win"].agg(["count", "mean"]).reset_index()
        
        level_year_stats.columns = [
            "レースレベル区分", "年", "レース数", "勝率"
        ]
        
        return level_year_stats

    def _calculate_distance_level_pivot(self) -> pd.DataFrame:
        """距離とレースレベルのピボットテーブルを計算"""
        df = self.df.copy()
        
        # NA値と無限大値の処理
        df['race_level'] = df['race_level'].fillna(0)
        df['race_level'] = df['race_level'].replace([np.inf, -np.inf], df['race_level'].replace([np.inf, -np.inf], np.nan).max())
        
        # 距離区分とレースレベル区分の計算
        df['距離区分'] = pd.cut(
            df['距離'],
            bins=[0, 1400, 1800, 2000, 2400, 9999],
            labels=['スプリント', 'マイル', '中距離', '中長距離', '長距離']
        )
        df['レースレベル区分'] = np.round(df['race_level']).astype(int)
        
        # ピボットテーブルの作成
        return df.pivot_table(
            index='距離区分',
            columns='レースレベル区分',
            values='is_win',
            aggfunc='mean'
        ) 
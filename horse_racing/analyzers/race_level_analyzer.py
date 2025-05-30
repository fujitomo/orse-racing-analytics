"""
レースレベル分析モジュール
レースのグレードや賞金額などからレースレベルを分析します。
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from horse_racing.base.analyzer import BaseAnalyzer, AnalysisConfig
from horse_racing.data.loader import RaceDataLoader
from horse_racing.visualization.plotter import RacePlotter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

# データローダーのログレベルを調整
loader_logger = logging.getLogger('horse_racing.data.loader')
loader_logger.setLevel(logging.WARNING)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

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
        """初期化"""
        super().__init__(config)
        self.plotter = RacePlotter(self.output_dir)
        self.loader = RaceDataLoader(config.input_path)

    @staticmethod
    def determine_grade_by_prize(row: pd.Series) -> int:
        """賞金からグレードを判定する関数"""
        prize = row.get("本賞金")
        if pd.isna(prize):
            return None
            
        match prize:
            case p if p >= 10000: return 1
            case p if p >= 7000: return 2
            case p if p >= 4500: return 3
            case p if p >= 3500: return 4
            case p if p >= 2000: return 6
            case _: return 5

    @staticmethod
    def determine_grade(row: pd.Series) -> int:
        """レース名と種別コードからグレードを判定する"""
        race_name = str(row.get("レース名", "")).upper().replace("Ｇ", "G").replace("Ｌ", "L")
        race_type = row.get("種別", 99)

        # キーワードとグレードのマッピング
        keyword_to_grade = {
            "G1": 1,
            "G2": 2,
            "G3": 3,
            "重賞": 4,
            "L": 6,
        }

        # マッチするキーワードを探す
        for keyword, grade in keyword_to_grade.items():
            if keyword in race_name:
                return grade
        
        # マッチするグレードがない場合は賞金による判定
        if "本賞金" in row.index:
            prize_grade = RaceLevelAnalyzer.determine_grade_by_prize(row)
            if prize_grade is not None:
                return prize_grade
        
        # 種別コードによる判定
        match race_type:
            case 11 | 12: return 5
            case 13 | 14: return 5
            case 20:
                if "J.G1" in race_name: return 1
                if "J.G2" in race_name: return 2
                if "J.G3" in race_name: return 3
                return 5
            case _: return 5

    def load_data(self) -> pd.DataFrame:
        """データの読み込み"""
        try:
            return self.loader.load()
        except FileNotFoundError as e:
            logger.error(f"指定されたパスにファイルが見つかりません: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """データの前処理"""
        try:
            df = self.df.copy()
            
            # データフレームの構造を確認
            logger.info("データフレームのカラム一覧:")
            logger.info(df.columns.tolist())
            logger.info("\nデータフレームの先頭5行:")
            logger.info(df.head())

            # 日付フィルタリング
            if self.config.start_date or self.config.end_date:
                # 日付カラムの作成
                try:
                    df['date'] = pd.to_datetime(
                        df['年'].astype(str) + 
                        df['回'].astype(str).str.zfill(2) + 
                        df['日'].astype(str).str.zfill(2),
                        format='%Y%m%d'
                    )
                except KeyError as e:
                    logger.error(f"日付カラムの作成に失敗しました。利用可能なカラム: {df.columns.tolist()}")
                    raise ValueError(f"日付カラム（年、回、日）が見つかりません: {str(e)}")
                
                # 開始日のフィルタリング
                if self.config.start_date:
                    df = df[df['date'] >= pd.to_datetime(self.config.start_date)]
                
                # 終了日のフィルタリング
                if self.config.end_date:
                    df = df[df['date'] <= pd.to_datetime(self.config.end_date)]
                
                # 日付カラムを削除
                df = df.drop('date', axis=1)

            # 芝レースのみをフィルタリング
            df = df[df["芝ダ障害コード"] == "芝"]

            # レース回数が最小レース数以上の馬のみを抽出
            race_counts = df['馬名'].value_counts()
            horses_with_min_races = race_counts[race_counts >= self.config.min_races].index
            df = df[df['馬名'].isin(horses_with_min_races)]

            if len(df) == 0:
                raise ValueError(f"条件を満たすデータが見つかりません（最小レース数: {self.config.min_races}）")

            # グレードの判定
            if "グレード" not in df.columns:
                df["グレード"] = df.apply(self.determine_grade, axis=1)

            # 必要な列の抽出と型変換
            required_columns = [
                "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
                "レース名", "種別", "芝ダ障害コード", "馬番", "グレード", "本賞金"
            ]

            # 必要なカラムの存在チェック
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"必要なカラムが不足しています: {', '.join(missing_columns)}")

            df = df[required_columns]

            # データ型の変換
            try:
                df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
                df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
                df["グレード"] = pd.to_numeric(df["グレード"], errors="coerce")
                df["種別"] = pd.to_numeric(df["種別"], errors="coerce")
                df["1着賞金"] = df["本賞金"]
            except Exception as e:
                logger.error(f"データ型の変換中にエラーが発生しました: {str(e)}")
                raise

            return df

        except Exception as e:
            logger.error(f"データの前処理中にエラーが発生しました: {str(e)}")
            raise

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

        # 距離による基本補正
        distance_weights = {
            (0, 1400): 0.85,     # スプリント
            (1401, 1800): 1.00,  # マイル
            (1801, 2000): 1.35,  # 中距離
            (2001, 2400): 1.45,  # 中長距離
            (2401, 9999): 1.25,  # 長距離
        }

        # 距離帯による基本補正を適用
        for (min_dist, max_dist), weight in distance_weights.items():
            mask = (df["距離"] >= min_dist) & (df["距離"] <= max_dist)
            df.loc[mask, "race_level"] *= weight

        # 2000m特別ボーナス
        mask_2000m = (df["距離"] >= 1900) & (df["距離"] <= 2100)
        df.loc[mask_2000m, "race_level"] *= 1.35

        # グレードと距離の相互作用を考慮
        high_grade_mask = df["グレード"].isin([1, 2, 3])  # G1, G2, G3
        optimal_distance_mask = (df["距離"] >= 1800) & (df["距離"] <= 2400)
        df.loc[high_grade_mask & optimal_distance_mask, "race_level"] *= 1.15

        # 最終的な正規化（0-10の範囲に収める）
        df["race_level"] = self.normalize_values(df["race_level"])

        return df

    def _perform_logistic_regression_analysis(self) -> Dict[str, Any]:
        """ロジスティック回帰分析を実行"""
        df = self.df.copy()
        df['is_win_or_place'] = df['着順'].apply(lambda x: 1 if x in [1, 2] else 0)
        df['is_placed_only'] = df['着順'].apply(lambda x: 1 if x <= 3 else 0)
        
        # NA値と無限大値の処理
        df['race_level'] = df['race_level'].fillna(0)
        df['race_level'] = df['race_level'].replace([np.inf, -np.inf], df['race_level'].replace([np.inf, -np.inf], np.nan).max())
        
        # 勝率のモデル
        X_win = df[['race_level']].values
        y_win = df['is_win_or_place'].values
        
        # 複勝率のモデル
        X_place = df[['race_level']].values
        y_place = df['is_placed_only'].values
        
        # 標準化
        scaler_win = StandardScaler()
        X_win_scaled = scaler_win.fit_transform(X_win)
        
        scaler_place = StandardScaler()
        X_place_scaled = scaler_place.fit_transform(X_place)
        
        # データ分割（層化サンプリング）
        X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(
            X_win_scaled, y_win, test_size=0.3, random_state=42, stratify=y_win
        )
        
        X_place_train, X_place_test, y_place_train, y_place_test = train_test_split(
            X_place_scaled, y_place, test_size=0.3, random_state=42, stratify=y_place
        )
        
        # クラスバランスを考慮したロジスティック回帰（勝率）
        model_win = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        model_win.fit(X_win_train, y_win_train)
        
        # クラスバランスを考慮したロジスティック回帰（複勝率）
        model_place = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        model_place.fit(X_place_train, y_place_train)
        
        # 予測と評価（勝率）
        y_win_pred = model_win.predict(X_win_test)
        accuracy_win = accuracy_score(y_win_test, y_win_pred)
        report_win = classification_report(y_win_test, y_win_pred, zero_division=0)
        conf_matrix_win = confusion_matrix(y_win_test, y_win_pred)
        
        # 予測と評価（複勝率）
        y_place_pred = model_place.predict(X_place_test)
        accuracy_place = accuracy_score(y_place_test, y_place_pred)
        report_place = classification_report(y_place_test, y_place_pred, zero_division=0)
        conf_matrix_place = confusion_matrix(y_place_test, y_place_pred)
        
        return {
            "win": {
                "model": model_win,
                "scaler": scaler_win,
                "accuracy": accuracy_win,
                "report": report_win,
                "conf_matrix": conf_matrix_win,
            },
            "place": {
                "model": model_place,
                "scaler": scaler_place,
                "accuracy": accuracy_place,
                "report": report_place,
                "conf_matrix": conf_matrix_place,
            },
            "data": df
        }

    def analyze(self) -> Dict[str, Any]:
        """分析の実行"""
        # 馬ごとの基本統計
        horse_stats = self._calculate_horse_stats()
        
        # グレード別の統計
        grade_stats = self._calculate_grade_stats()
        
        # 相関分析
        correlation_stats = self._perform_correlation_analysis(horse_stats)
        
        # ロジスティック回帰分析
        logistic_stats = self._perform_logistic_regression_analysis()
        
        return {
            "horse_stats": horse_stats,
            "grade_stats": grade_stats,
            "correlation_stats": correlation_stats,
            "logistic_stats": logistic_stats
        }

    def visualize(self) -> None:
        """分析結果の可視化"""
        if not hasattr(self, 'stats') or not self.stats:
            return

        horse_stats = self.stats["horse_stats"]
        grade_stats = self.stats["grade_stats"]
        correlation_stats = self.stats["correlation_stats"]
        logistic_stats = self.stats.get("logistic_stats", {})

        # グレード別統計の可視化
        self.plotter.plot_grade_stats(grade_stats, self.GRADE_LEVELS)

        # 勝率の相関分析の可視化
        if correlation_stats:
            # 最高レベルでの分析
            self.plotter.plot_correlation_analysis(
                data=horse_stats,
                correlation=correlation_stats["correlation_win_max"],
                model=correlation_stats["model_win_max"],
                r2=correlation_stats["r2_win_max"],
                feature_name="レースレベル（最高）と勝率の関係",
                x_column="最高レースレベル　※馬が過去に勝った最も高いレベルのレースを示す"
            )
            
            # 複勝率の相関分析の可視化（最高レベル）
            self.plotter.plot_correlation_analysis(
                data=horse_stats,
                correlation=correlation_stats["correlation_place_max"],
                model=correlation_stats["model_place_max"],
                r2=correlation_stats["r2_place_max"],
                feature_name="レースレベル（最高）と複勝率の関係",
                x_column="最高レースレベル　※馬が過去に勝った最も高いレベルのレースを示す",
                y_column="place_rate"
            )
            
            # 平均レベルでの勝率分析
            # horse_statsのデータを平均レベル用に調整
            horse_stats_avg = horse_stats.copy()
            horse_stats_avg['最高レベル'] = horse_stats_avg['平均レベル']  # plotterが最高レベルを期待するため
            
            self.plotter.plot_correlation_analysis(
                data=horse_stats_avg,
                correlation=correlation_stats["correlation_win_avg"],
                model=correlation_stats["model_win_avg"],
                r2=correlation_stats["r2_win_avg"],
                feature_name="レースレベル（平均）と勝率の関係",
                x_column="平均レースレベル　※馬の全出走レースの平均レベルを示す"
            )
            
            # 平均レベルでの複勝率分析
            self.plotter.plot_correlation_analysis(
                data=horse_stats_avg,
                correlation=correlation_stats["correlation_place_avg"],
                model=correlation_stats["model_place_avg"],
                r2=correlation_stats["r2_place_avg"],
                feature_name="レースレベル（平均）と複勝率の関係",
                x_column="平均レースレベル　※馬の全出走レースの平均レベルを示す",
                y_column="place_rate"
            )

        # レースレベルの分布分析
        if "race_level" in self.df.columns:
            self.plotter.plot_distribution_analysis(
                data=self.df[self.df["is_win"]],
                feature_name="race_level",
                bins=30,
                title="勝利時のレースレベル分布"
            )
            
            # 複勝時のレースレベル分布も追加
            self.plotter.plot_distribution_analysis(
                data=self.df[self.df["is_placed"]],
                feature_name="race_level",
                bins=30,
                title="複勝時のレースレベル分布"
            )

        # トレンド分析
        level_year_stats = self._calculate_level_year_stats()
        if level_year_stats is not None:
            self.plotter.plot_trend_analysis(
                data=level_year_stats,
                feature_name="レースレベル"
            )

        # 距離とレースレベルのヒートマップ
        distance_level_pivot = self._calculate_distance_level_pivot()
        if distance_level_pivot is not None:
            self.plotter.plot_heatmap(
                pivot_table=distance_level_pivot,
                title="距離別・レースレベル別の勝率ヒートマップ",
                filename="distance_level_win_heatmap.png"
            )
            
            # 複勝のヒートマップも追加
            distance_level_place_pivot = self._calculate_distance_level_pivot(value_column="is_placed")
            if distance_level_place_pivot is not None:
                self.plotter.plot_heatmap(
                    pivot_table=distance_level_place_pivot,
                    title="距離別・レースレベル別の複勝率ヒートマップ",
                    filename="distance_level_place_heatmap.png"
                )

        # ロジスティック回帰の可視化
        if logistic_stats and 'data' in logistic_stats:
            data = logistic_stats['data']
            if 'race_level' in data.columns:
                # 勝率の可視化
                if 'is_win_or_place' in data.columns:
                    X = data['race_level'].values.reshape(-1, 1)
                    y = data['is_win_or_place'].values
                    y_pred_proba = logistic_stats['win']['model'].predict_proba(
                        logistic_stats['win']['scaler'].transform(X)
                    )[:, 1]
                    y_pred = logistic_stats['win']['model'].predict(
                        logistic_stats['win']['scaler'].transform(X)
                    )
                    self.plotter.plot_logistic_regression(
                        X=data['race_level'].values,
                        y=y,
                        y_pred_proba=y_pred_proba,
                        y_pred=y_pred,
                        feature_name="レースレベルと勝率の関係（ロジスティック回帰）"
                    )

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
        
        # レース回数がmin_races回以上の馬のみをフィルタリング
        min_races = self.config.min_races if hasattr(self.config, 'min_races') else 3
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
        # TODO:欠損値のついて調査予定
        analysis_data = horse_stats.dropna(subset=['最高レベル', '平均レベル', 'win_rate', 'place_rate'])
        
        if len(analysis_data) == 0:
            return {}

        # 標準偏差が0の場合の処理
        # TODO:標準偏差が0の場合の処理を調査予定
        stddev = analysis_data[['最高レベル', '平均レベル', 'win_rate', 'place_rate']].std()
        if (stddev == 0).any():
            return {
                "correlation_win_max": 0.0,
                "correlation_place_max": 0.0,
                "correlation_win_avg": 0.0,
                "correlation_place_avg": 0.0,
                "model_win_max": None,
                "model_place_max": None,
                "model_win_avg": None,
                "model_place_avg": None,
                "r2_win_max": 0.0,
                "r2_place_max": 0.0,
                "r2_win_avg": 0.0,
                "r2_place_avg": 0.0
            }

        # 最高レベル - 勝率の相関係数と回帰分析
        correlation_win_max = analysis_data[['最高レベル', 'win_rate']].corr().iloc[0, 1]
        X_win_max = analysis_data['最高レベル'].values.reshape(-1, 1)
        y_win = analysis_data['win_rate'].values
        model_win_max = LinearRegression()
        model_win_max.fit(X_win_max, y_win)
        r2_win_max = model_win_max.score(X_win_max, y_win)

        # 最高レベル - 複勝率の相関係数と回帰分析
        correlation_place_max = analysis_data[['最高レベル', 'place_rate']].corr().iloc[0, 1]
        X_place_max = analysis_data['最高レベル'].values.reshape(-1, 1)
        y_place = analysis_data['place_rate'].values
        model_place_max = LinearRegression()
        model_place_max.fit(X_place_max, y_place)
        r2_place_max = model_place_max.score(X_place_max, y_place)

        # 平均レベル - 勝率の相関係数と回帰分析
        correlation_win_avg = analysis_data[['平均レベル', 'win_rate']].corr().iloc[0, 1]
        X_win_avg = analysis_data['平均レベル'].values.reshape(-1, 1)
        model_win_avg = LinearRegression()
        model_win_avg.fit(X_win_avg, y_win)
        r2_win_avg = model_win_avg.score(X_win_avg, y_win)

        # 平均レベル - 複勝率の相関係数と回帰分析
        correlation_place_avg = analysis_data[['平均レベル', 'place_rate']].corr().iloc[0, 1]
        X_place_avg = analysis_data['平均レベル'].values.reshape(-1, 1)
        model_place_avg = LinearRegression()
        model_place_avg.fit(X_place_avg, y_place)
        r2_place_avg = model_place_avg.score(X_place_avg, y_place)

        return {
            # 最高レベル系
            "correlation_win_max": correlation_win_max,
            "correlation_place_max": correlation_place_max,
            "model_win_max": model_win_max,
            "model_place_max": model_place_max,
            "r2_win_max": r2_win_max,
            "r2_place_max": r2_place_max,
            # 平均レベル系
            "correlation_win_avg": correlation_win_avg,
            "correlation_place_avg": correlation_place_avg,
            "model_win_avg": model_win_avg,
            "model_place_avg": model_place_avg,
            "r2_win_avg": r2_win_avg,
            "r2_place_avg": r2_place_avg,
            # 後方互換性のため既存のキーも残す
            "correlation_win": correlation_win_max,
            "correlation_place": correlation_place_max,
            "model_win": model_win_max,
            "model_place": model_place_max,
            "r2_win": r2_win_max,
            "r2_place": r2_place_max
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

    def _calculate_distance_level_pivot(self, value_column: str = "is_win") -> pd.DataFrame:
        """距離とレースレベルのピボットテーブルを計算"""
        try:
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
            
            # ピボットテーブルの作成（warningを抑制するためにobserved=Trueを指定）
            return df.pivot_table(
                index='距離区分',
                columns='レースレベル区分',
                values=value_column,
                aggfunc='mean',
                observed=True  # FutureWarningを解決
            )
        except Exception as e:
            logger.error(f"ピボットテーブルの作成中にエラーが発生しました: {str(e)}")
            return None 
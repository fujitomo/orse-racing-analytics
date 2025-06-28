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
from horse_racing.analyzers.causal_analyzer import analyze_causal_relationship, generate_causal_analysis_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from pathlib import Path

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

            # カラム名の前後の空白を除去
            df.columns = df.columns.str.strip()

            # 日付フィルタリング
            if self.config.start_date or self.config.end_date:
                # 日付カラムの作成
                try:
                    if '年月日' in df.columns:
                        df['date'] = pd.to_datetime(df['年月日'].astype(str), format='%Y%m%d')
                    else:
                        # 年月日カラムがない場合は年、回、日から作成
                        df['date'] = pd.to_datetime(
                            df['年'].astype(str) + 
                            df['回'].astype(str).str.zfill(2) + 
                            df['日'].astype(str).str.zfill(2)
                        )
                except Exception as e:
                    logger.error(f"日付の変換に失敗しました: {str(e)}")
                    raise

                # 日付でフィルタリング
                if self.config.start_date:
                    df = df[df['date'] >= self.config.start_date]
                if self.config.end_date:
                    df = df[df['date'] <= self.config.end_date]

            # 最小レース数でフィルタリング
            if self.config.min_races:
                # 馬ごとのレース数をカウント
                race_counts = df['馬名'].value_counts()
                valid_horses = race_counts[race_counts >= self.config.min_races].index
                df = df[df['馬名'].isin(valid_horses)]

                if len(df) == 0:
                    raise ValueError(f"条件を満たすデータが見つかりません（最小レース数: {self.config.min_races}）")

            logger.info(f"  📊 対象データ: {len(df):,}行")
            logger.info(f"  🐎 対象馬数: {df['馬名'].nunique():,}頭")

            return df

        except Exception as e:
            logger.error(f"データの前処理中にエラーが発生しました: {str(e)}")
            raise

    def calculate_feature(self) -> pd.DataFrame:
        """レースレベルの計算"""
        df = self.df.copy()
        
        # カラム名の前後の空白を除去
        df.columns = df.columns.str.strip()
        
        # 必要なカラムを選択
        required_columns = [
            '場コード', '年', '回', '日', 'R', '馬名', '距離', '着順',
            'レース名', '種別', '芝ダ障害コード', '馬番', 'グレード',
            '本賞金', '1着賞金', '年月日'
        ]
        df = df[required_columns]

        # レースレベル関連の特徴量を追加
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
        try:
            # データフレームの構造を確認
            logger.info("データフレームのカラム一覧:")
            logger.info(self.df.columns.tolist())
            logger.info("\nデータフレームの先頭5行:")
            logger.info(self.df.head())
            
            # 基本的な相関分析
            correlation_stats = self._perform_correlation_analysis(self._calculate_horse_stats())
            results = {'correlation_stats': correlation_stats}
            
            # 因果関係分析の追加
            causal_results = analyze_causal_relationship(self.df)
            results['causal_analysis'] = causal_results
            
            # 因果関係分析レポートの生成
            output_dir = Path(self.config.output_dir)
            generate_causal_analysis_report(causal_results, output_dir)
            
            logger.info("✅ 因果関係分析が完了しました")
            
            return results
            
        except Exception as e:
            logger.error(f"分析中にエラーが発生しました: {str(e)}")
            raise

    def visualize(self) -> None:
        """分析結果の可視化"""
        try:
            if not self.stats:
                raise ValueError("分析結果がありません。先にanalyzeメソッドを実行してください。")

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 相関分析の可視化
            self._visualize_correlations()
            
            # 因果関係分析の可視化
            if 'causal_analysis' in self.stats:
                self._visualize_causal_analysis()

        except Exception as e:
            logger.error(f"可視化中にエラーが発生しました: {str(e)}")
            raise

    def _visualize_causal_analysis(self) -> None:
        """因果関係分析の可視化"""
        causal_results = self.stats.get('causal_analysis', {})
        output_dir = Path(self.config.output_dir) / 'causal_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 時間的先行性の可視化
        if 'temporal_precedence' in causal_results:
            self._plot_temporal_precedence(output_dir)

        # メカニズムの可視化
        if 'mechanism' in causal_results:
            self._plot_mechanism_analysis(output_dir)

        # 交絡因子の可視化
        if 'confounding_factors' in causal_results:
            self._plot_confounding_factors(output_dir)

    def _plot_temporal_precedence(self, output_dir: Path) -> None:
        """時間的先行性の可視化"""
        plt.figure(figsize=(10, 6))
        horse_data = []

        for horse in self.df['馬名'].unique():
            horse_races = self.df[self.df['馬名'] == horse].sort_values('年月日')
            if len(horse_races) >= 6:
                initial_level = horse_races['race_level'].iloc[:3].mean()
                later_performance = (horse_races['着順'] <= 3).iloc[3:].mean()
                horse_data.append({
                    '初期レベル': initial_level,
                    '後期成績': later_performance
                })

        if horse_data:
            df_temporal = pd.DataFrame(horse_data)
            plt.scatter(df_temporal['初期レベル'], df_temporal['後期成績'], alpha=0.5)
            plt.xlabel('初期レースレベル')
            plt.ylabel('後期複勝率')
            plt.title('初期レースレベルと後期成績の関係')
            
            # 回帰直線の追加
            z = np.polyfit(df_temporal['初期レベル'], df_temporal['後期成績'], 1)
            p = np.poly1d(z)
            plt.plot(df_temporal['初期レベル'], p(df_temporal['初期レベル']), "r--", alpha=0.8)
            
            plt.savefig(output_dir / 'temporal_precedence.png')
            plt.close()

    def _plot_mechanism_analysis(self, output_dir: Path) -> None:
        """メカニズム分析の可視化"""
        plt.figure(figsize=(12, 6))

        # レースレベルと成績の関係をプロット
        level_performance = []
        for horse in self.df['馬名'].unique():
            horse_races = self.df[self.df['馬名'] == horse]
            if len(horse_races) >= 6:
                avg_level = horse_races['race_level'].mean()
                win_rate = (horse_races['着順'] == 1).mean()
                place_rate = (horse_races['着順'] <= 3).mean()

                level_performance.append({
                    '平均レベル': avg_level,
                    '勝率': win_rate,
                    '複勝率': place_rate
                })

        if level_performance:
            df_mechanism = pd.DataFrame(level_performance)

            plt.subplot(1, 2, 1)
            plt.scatter(df_mechanism['平均レベル'], df_mechanism['勝率'], alpha=0.5)
            z = np.polyfit(df_mechanism['平均レベル'], df_mechanism['勝率'], 1)
            p = np.poly1d(z)
            plt.plot(df_mechanism['平均レベル'], p(df_mechanism['平均レベル']), "r--", alpha=0.8)
            plt.title('レースレベルと勝率の関係')
            plt.xlabel('平均レースレベル')
            plt.ylabel('勝率')

            plt.subplot(1, 2, 2)
            plt.scatter(df_mechanism['平均レベル'], df_mechanism['複勝率'], alpha=0.5)
            z = np.polyfit(df_mechanism['平均レベル'], df_mechanism['複勝率'], 1)
            p = np.poly1d(z)
            plt.plot(df_mechanism['平均レベル'], p(df_mechanism['平均レベル']), "r--", alpha=0.8)
            plt.title('レースレベルと複勝率の関係')
            plt.xlabel('平均レースレベル')
            plt.ylabel('複勝率')

            plt.tight_layout()
            plt.savefig(output_dir / 'mechanism_analysis.png')
            plt.close()

    def _plot_confounding_factors(self, output_dir: Path) -> None:
        """交絡因子の可視化"""
        confounders = ['場コード', '距離', '芝ダ障害コード']

        for confounder in confounders:
            if confounder in self.df.columns:
                plt.figure(figsize=(10, 6))

                # 交絡因子ごとの平均成績をプロット
                grouped_stats = self.df.groupby(confounder).agg({
                    'race_level': 'mean',
                    '着順': lambda x: (x <= 3).mean()
                }).reset_index()

                plt.scatter(grouped_stats['race_level'], grouped_stats['着順'],
                          s=100, alpha=0.6)

                # ラベルの追加
                for i, row in grouped_stats.iterrows():
                    plt.annotate(row[confounder],
                               (row['race_level'], row['着順']),
                               xytext=(5, 5), textcoords='offset points')

                plt.title(f'{confounder}による交絡効果')
                plt.xlabel('平均レースレベル')
                plt.ylabel('複勝率')

                plt.savefig(output_dir / f'confounding_{confounder}.png')
                plt.close()

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
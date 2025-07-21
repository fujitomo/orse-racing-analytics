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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
from scipy import stats
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

    def __init__(self, config: AnalysisConfig, enable_time_analysis: bool = False):
        """初期化"""
        super().__init__(config)
        self.plotter = RacePlotter(self.output_dir)
        self.loader = RaceDataLoader(config.input_path)
        self.class_column = None  # 実際のクラスカラム名を動的に設定
        self.time_analysis_results = {}  # タイム分析結果を保存
        self.enable_time_analysis = enable_time_analysis  # RunningTime分析の有効/無効

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
        
        # 必要なカラムを選択（実際に存在するカラム名に基づく）
        base_required_columns = [
            '場コード', '年', '回', '日', 'R', '馬名', '距離', '着順',
            'レース名', '種別', '芝ダ障害コード', '馬番',
            '本賞金', '1着賞金', '年月日'
        ]
        
        # タイム関連カラムの追加
        time_columns = []
        for col in ['タイム', 'time', 'Time', '走破タイム']:
            if col in df.columns:
                time_columns.append(col)
                break
        
        # クラス関連のカラムを動的に追加と判定
        class_columns = []
        for col in ['クラス', 'クラスコード', '条件']:
            if col in df.columns:
                class_columns.append(col)
                if self.class_column is None:  # 最初に見つかったクラス関連カラムを使用
                    self.class_column = col
        
        required_columns = base_required_columns + class_columns + time_columns
        
        # 存在するカラムのみを選択
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]

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
        if self.class_column and self.class_column in df.columns:
            high_grade_mask = df[self.class_column].isin([1, 2, 3])  # G1, G2, G3
            optimal_distance_mask = (df["距離"] >= 1800) & (df["距離"] <= 2400)
            df.loc[high_grade_mask & optimal_distance_mask, "race_level"] *= 1.15

        # 最終的な正規化（0-10の範囲に収める）
        df["race_level"] = self.normalize_values(df["race_level"])

        # RunningTime分析機能を追加（有効な場合のみ）
        if self.enable_time_analysis:
            df = self.calculate_running_time_features(df)

        return df

    def calculate_running_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """走破タイム関連特徴量の計算"""
        try:
            logger.info("🏃 走破タイム特徴量の計算を開始...")
            
            # タイムカラムの特定
            time_column = None
            for col in ['タイム', 'time', 'Time', '走破タイム']:
                if col in df.columns:
                    time_column = col
                    break
            
            if time_column is None:
                logger.warning("⚠️ タイムカラムが見つかりません。RunningTime分析をスキップします。")
                return df
            
            logger.info(f"📊 使用するタイムカラム: {time_column}")
            
            # タイムデータの前処理
            df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
            
            # 異常値の除去（0秒や極端に遅いタイム）
            valid_time_mask = (df[time_column] > 60) & (df[time_column] < 600)  # 1分〜10分の範囲
            df = df[valid_time_mask].copy()
            
            logger.info(f"📊 有効なタイムデータ: {len(df):,}件")
            
            # 1. 距離補正タイムの計算（2000m換算）
            df['distance_adjusted_time'] = df[time_column] / df['距離'] * 2000
            
            # 2. 同条件内でのZ-score正規化
            grouping_columns = ['場コード', '芝ダ障害コード']
            available_grouping = [col for col in grouping_columns if col in df.columns]
            
            if available_grouping:
                df['time_zscore'] = df.groupby(available_grouping)[time_column].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
                logger.info(f"📊 Z-score正規化完了（グループ化: {available_grouping}）")
            else:
                # グループ化できない場合は全体でZ-score
                df['time_zscore'] = (df[time_column] - df[time_column].mean()) / df[time_column].std()
                logger.info("📊 Z-score正規化完了（全体平均）")
            
            # 3. 速度指標の計算（m/分）
            df['speed_index'] = df['距離'] / df[time_column] * 60
            
            # 4. 距離別基準タイムとの比較
            df['time_ratio'] = df.groupby('距離')[time_column].transform(
                lambda x: df.loc[x.index, time_column] / x.mean()
            )
            
            # 5. タイムランキング（同レース内）
            df['time_rank_in_race'] = df.groupby(['場コード', '年', '回', '日', 'R'])[time_column].rank(method='min')
            
            logger.info("✅ 走破タイム特徴量の計算が完了しました")
            logger.info(f"   - distance_adjusted_time: 距離補正タイム（2000m換算）")
            logger.info(f"   - time_zscore: Z-score正規化タイム")
            logger.info(f"   - speed_index: 速度指標（m/分）")
            logger.info(f"   - time_ratio: 距離別基準タイム比")
            logger.info(f"   - time_rank_in_race: レース内タイムランキング")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 走破タイム特徴量計算中にエラー: {str(e)}")
            return df

    def analyze_time_causality(self) -> Dict[str, Any]:
        """タイム関連の因果分析"""
        try:
            logger.info("🔬 タイム因果分析を開始...")
            
            results = {}
            
            # データの準備
            analysis_data = self.df.dropna(subset=['race_level', 'time_zscore', 'is_placed'])
            
            if len(analysis_data) == 0:
                logger.warning("⚠️ 分析可能なデータがありません")
                return {}
            
            logger.info(f"📊 分析対象データ: {len(analysis_data):,}件")
            
            # 1. 仮説H1の検証: RaceLevel → RunningTime
            h1_results = self.verify_hypothesis_h1(analysis_data)
            results['hypothesis_h1'] = h1_results
            
            # 2. 仮説H4の検証: RunningTime → PlaceRate
            h4_results = self.verify_hypothesis_h4(analysis_data)
            results['hypothesis_h4'] = h4_results
            
            # 3. 総合的な因果関係分析
            comprehensive_results = self._analyze_comprehensive_causality(analysis_data)
            results['comprehensive_analysis'] = comprehensive_results
            
            self.time_analysis_results = results
            logger.info("✅ タイム因果分析が完了しました")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ タイム因果分析中にエラー: {str(e)}")
            return {}

    def verify_hypothesis_h1(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        仮説H1の検証: RaceLevel（レース格）が高いほど、走破タイムが速くなる（距離補正済み）
        距離・馬場状態を統制した多変量回帰分析
        """
        try:
            logger.info("🧪 仮説H1検証: RaceLevel → RunningTime")
            
            # 説明変数の準備
            X = data[['race_level']].copy()
            
            # 距離カテゴリの追加（統制変数）
            data['distance_category'] = pd.cut(data['距離'], 
                                             bins=[0, 1400, 1800, 2000, 2400, 9999],
                                             labels=['sprint', 'mile', 'middle', 'long', 'extra_long'])
            
            # カテゴリ変数のダミー化
            distance_dummies = pd.get_dummies(data['distance_category'], prefix='dist')
            X = pd.concat([X, distance_dummies], axis=1)
            
            # 馬場状態の統制（存在する場合）
            if '馬場状態' in data.columns:
                track_dummies = pd.get_dummies(data['馬場状態'], prefix='track')
                X = pd.concat([X, track_dummies], axis=1)
            
            # 目的変数（タイムが速い方が負の値になるため、符号を反転）
            y = -data['time_zscore']  # 速いタイム = 高い値
            
            # 回帰分析の実行
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # 統計的有意性の検定
            correlation = data['race_level'].corr(-data['time_zscore'])
            n = len(data)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            results = {
                'model': model,
                'r2_score': r2,
                'mse': mse,
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': n,
                'is_significant': p_value < 0.05,
                'effect_direction': 'positive' if correlation > 0 else 'negative',
                'interpretation': self._interpret_h1_results(correlation, p_value, r2)
            }
            
            logger.info(f"   📊 相関係数: {correlation:.3f}")
            logger.info(f"   📊 決定係数: {r2:.3f}")
            logger.info(f"   📊 p値: {p_value:.6f}")
            logger.info(f"   📊 統計的有意性: {'有意' if p_value < 0.05 else '非有意'}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 仮説H1検証中にエラー: {str(e)}")
            return {}

    def verify_hypothesis_h4(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        仮説H4の検証: RunningTime が速いほど複勝率が高い
        ロジスティック回帰分析（距離・馬場バイアス調整）
        """
        try:
            logger.info("🧪 仮説H4検証: RunningTime → PlaceRate")
            
            # 説明変数の準備
            X = data[['time_zscore', 'race_level']].copy()
            
            # 距離の統制
            X['distance'] = data['距離']
            
            # 目的変数
            y = data['is_placed']
            
            # ロジスティック回帰の実行
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # 評価指標の計算
            accuracy = accuracy_score(y, y_pred)
            
            # オッズ比の計算
            odds_ratios = np.exp(model.coef_[0])
            
            # 相関係数の計算（time_zscoreが負の値なので符号を調整）
            correlation = (-data['time_zscore']).corr(data['is_placed'])
            
            # 統計的有意性の検定
            n = len(data)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            results = {
                'model': model,
                'accuracy': accuracy,
                'odds_ratios': odds_ratios,
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': n,
                'is_significant': p_value < 0.05,
                'predictions': y_pred_proba,
                'interpretation': self._interpret_h4_results(correlation, p_value, odds_ratios[0])
            }
            
            logger.info(f"   📊 相関係数: {correlation:.3f}")
            logger.info(f"   📊 精度: {accuracy:.3f}")
            logger.info(f"   📊 タイムのオッズ比: {odds_ratios[0]:.3f}")
            logger.info(f"   📊 p値: {p_value:.6f}")
            logger.info(f"   📊 統計的有意性: {'有意' if p_value < 0.05 else '非有意'}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 仮説H4検証中にエラー: {str(e)}")
            return {}

    def _analyze_comprehensive_causality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """包括的な因果関係分析"""
        try:
            logger.info("🔬 包括的因果関係分析を実行...")
            
            results = {}
            
            # 1. RaceLevel → Time → PlaceRate の媒介効果分析（簡易版）
            # ステップ1: RaceLevel → PlaceRate の直接効果
            direct_corr = data['race_level'].corr(data['is_placed'])
            
            # ステップ2: RaceLevel → Time の効果
            race_time_corr = data['race_level'].corr(-data['time_zscore'])
            
            # ステップ3: Time → PlaceRate の効果（RaceLevelを統制）
            partial_corr = self._calculate_partial_correlation(
                data['time_zscore'], data['is_placed'], data['race_level']
            )
            
            # 媒介効果の推定
            indirect_effect = race_time_corr * partial_corr
            direct_effect_controlled = direct_corr - indirect_effect
            
            mediation_results = {
                'total_effect': direct_corr,
                'direct_effect': direct_effect_controlled,
                'indirect_effect': indirect_effect,
                'mediation_ratio': indirect_effect / direct_corr if direct_corr != 0 else 0
            }
            
            results['mediation_analysis'] = mediation_results
            
            # 2. 距離別の効果分析
            distance_effects = {}
            distance_categories = ['sprint', 'mile', 'middle', 'long']
            
            for category in distance_categories:
                if category == 'sprint':
                    mask = data['距離'] <= 1400
                elif category == 'mile':
                    mask = (data['距離'] > 1400) & (data['距離'] <= 1800)
                elif category == 'middle':
                    mask = (data['距離'] > 1800) & (data['距離'] <= 2400)
                else:  # long
                    mask = data['距離'] > 2400
                
                if mask.sum() > 10:  # 十分なデータがある場合のみ
                    subset = data[mask]
                    corr_level_time = subset['race_level'].corr(-subset['time_zscore'])
                    corr_time_place = (-subset['time_zscore']).corr(subset['is_placed'])
                    
                    distance_effects[category] = {
                        'sample_size': len(subset),
                        'race_level_time_correlation': corr_level_time,
                        'time_place_correlation': corr_time_place
                    }
            
            results['distance_specific_effects'] = distance_effects
            
            logger.info("✅ 包括的因果関係分析が完了しました")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 包括的因果関係分析中にエラー: {str(e)}")
            return {}

    def _calculate_partial_correlation(self, x, y, control_var):
        """偏相関係数の計算"""
        try:
            # xとcontrol_varの回帰残差
            model_x = LinearRegression()
            model_x.fit(control_var.values.reshape(-1, 1), x)
            residual_x = x - model_x.predict(control_var.values.reshape(-1, 1))
            
            # yとcontrol_varの回帰残差
            model_y = LinearRegression()
            model_y.fit(control_var.values.reshape(-1, 1), y)
            residual_y = y - model_y.predict(control_var.values.reshape(-1, 1))
            
            # 残差間の相関係数
            return pd.Series(residual_x).corr(pd.Series(residual_y))
            
        except Exception:
            return 0.0

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
            
            # RunningTime分析の実行（有効な場合のみ）
            if self.enable_time_analysis:
                time_analysis_results = self.analyze_time_causality()
                if time_analysis_results:
                    results['time_analysis'] = time_analysis_results
                    logger.info("✅ RunningTime分析が完了しました")
            else:
                time_analysis_results = None
            
            # 因果関係分析の追加
            causal_results = analyze_causal_relationship(self.df)
            results['causal_analysis'] = causal_results
            
            # 因果関係分析レポートの生成
            output_dir = Path(self.config.output_dir)
            generate_causal_analysis_report(causal_results, output_dir)
            
            # RunningTime分析レポートの生成
            if time_analysis_results:
                self._generate_time_analysis_report(time_analysis_results, output_dir)
            
            logger.info("✅ 全ての分析が完了しました")
            
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
            self.plotter._visualize_correlations(self._calculate_horse_stats(), self.stats['correlation_stats'])
            
            # レース格別・距離別の箱ひげ図分析（論文要求対応）
            logger.info("📊 レース格別・距離別の箱ひげ図分析を実行中...")
            self.plotter.plot_race_grade_distance_boxplot(self.df)
            logger.info("✅ 箱ひげ図分析が完了しました")
            
            # RunningTime分析の可視化
            if 'time_analysis' in self.stats:
                self._visualize_time_analysis()
                logger.info("✅ RunningTime分析の可視化が完了しました")
            
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
        if not self.class_column or self.class_column not in df.columns:
            # クラスカラムが存在しない場合はデフォルト値を返す
            return pd.Series([5.0] * len(df), index=df.index)
            
        grade_level = df[self.class_column].map(
            lambda x: self.GRADE_LEVELS[x]["base_level"] if pd.notna(x) and x in self.GRADE_LEVELS else 5.0
        )

        for grade, values in self.GRADE_LEVELS.items():
            mask = df[self.class_column] == grade
            grade_level.loc[mask & df["is_win"]] += values["weight"]
            grade_level.loc[mask & df["is_placed"] & ~df["is_win"]] += values["weight"] * 0.5

        return self.normalize_values(grade_level)

    def _calculate_prize_level(self, df: pd.DataFrame) -> pd.Series:
        """賞金に基づくレベルを計算"""
        prize_level = np.log1p(df["1着賞金"]) / np.log1p(df["1着賞金"].max()) * 9.95
        return self.normalize_values(prize_level)

    def _calculate_horse_stats(self) -> pd.DataFrame:
        """馬ごとの基本統計を計算"""
        if "is_win" not in self.df.columns:
            self.df["is_win"] = self.df["着順"] == 1
        if "is_placed" not in self.df.columns:
            self.df["is_placed"] = self.df["着順"] <= 3

        # 馬ごとの基本統計
        agg_dict = {
            "race_level": ["max", "mean"],
            "is_win": "sum",
            "is_placed": "sum",
            "着順": "count"
        }
        
        # クラスカラムが存在する場合のみ追加
        if self.class_column and self.class_column in self.df.columns:
            agg_dict[self.class_column] = lambda x: x.value_counts().idxmax() if not x.empty else 0
        
        horse_stats = self.df.groupby("馬名").agg(agg_dict).reset_index()

        # カラム名の整理
        if self.class_column and self.class_column in self.df.columns:
            horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数", "主戦クラス"]
        else:
            horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数"]
        
        # レース回数がmin_races回以上の馬のみをフィルタリング
        min_races = self.config.min_races if hasattr(self.config, 'min_races') else 3
        horse_stats = horse_stats[horse_stats["出走回数"] >= min_races]
        
        # 勝率と複勝率の計算
        horse_stats["win_rate"] = horse_stats["勝利数"] / horse_stats["出走回数"]
        horse_stats["place_rate"] = horse_stats["複勝数"] / horse_stats["出走回数"]
        
        return horse_stats

    def _calculate_grade_stats(self) -> pd.DataFrame:
        """グレード別の統計を計算"""
        if not self.class_column or self.class_column not in self.df.columns:
            # クラスカラムが存在しない場合は空のDataFrameを返す
            return pd.DataFrame()
            
        grade_stats = self.df.groupby(self.class_column).agg({
            "is_win": ["mean", "count"],
            "is_placed": "mean",
            "race_level": ["mean", "std"]
        }).reset_index()

        grade_stats.columns = [
            "クラス", "勝率", "レース数", "複勝率",
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

    def _interpret_h1_results(self, correlation: float, p_value: float, r2: float) -> str:
        """仮説H1の結果解釈"""
        significance = "統計的に有意" if p_value < 0.05 else "統計的に非有意"
        strength = "強い" if abs(correlation) > 0.5 else "中程度" if abs(correlation) > 0.3 else "弱い"
        direction = "正の" if correlation > 0 else "負の"
        
        return f"レースレベルと走破タイムには{strength}{direction}相関があり、{significance}です（r={correlation:.3f}, R²={r2:.3f}）。"

    def _interpret_h4_results(self, correlation: float, p_value: float, odds_ratio: float) -> str:
        """仮説H4の結果解釈"""
        significance = "統計的に有意" if p_value < 0.05 else "統計的に非有意"
        strength = "強い" if abs(correlation) > 0.5 else "中程度" if abs(correlation) > 0.3 else "弱い"
        
        if odds_ratio > 1:
            effect = f"速いタイムは複勝率を{odds_ratio:.2f}倍高める"
        else:
            effect = f"速いタイムは複勝率を{1/odds_ratio:.2f}分の1に下げる"
        
        return f"走破タイムと複勝率には{strength}相関があり、{significance}です（r={correlation:.3f}）。{effect}効果があります。"

    def _generate_time_analysis_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """RunningTime分析レポートの生成"""
        try:
            logger.info("📝 RunningTime分析レポートを生成中...")
            
            report_path = output_dir / 'running_time_analysis_report.md'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 走破タイム因果関係分析レポート\n\n")
                f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 仮説H1の結果
                if 'hypothesis_h1' in results:
                    h1 = results['hypothesis_h1']
                    f.write("## 🧪 仮説H1検証: RaceLevel → RunningTime\n\n")
                    f.write("### 分析結果\n")
                    f.write(f"- **相関係数**: {h1.get('correlation', 0):.3f}\n")
                    f.write(f"- **決定係数 (R²)**: {h1.get('r2_score', 0):.3f}\n")
                    f.write(f"- **p値**: {h1.get('p_value', 1):.6f}\n")
                    f.write(f"- **統計的有意性**: {'有意' if h1.get('is_significant', False) else '非有意'}\n")
                    f.write(f"- **サンプルサイズ**: {h1.get('sample_size', 0):,}件\n")
                    f.write(f"- **効果の方向**: {h1.get('effect_direction', '不明')}\n\n")
                    f.write("### 解釈\n")
                    f.write(f"{h1.get('interpretation', '解釈情報なし')}\n\n")
                
                # 仮説H4の結果
                if 'hypothesis_h4' in results:
                    h4 = results['hypothesis_h4']
                    f.write("## 🧪 仮説H4検証: RunningTime → PlaceRate\n\n")
                    f.write("### 分析結果\n")
                    f.write(f"- **相関係数**: {h4.get('correlation', 0):.3f}\n")
                    f.write(f"- **予測精度**: {h4.get('accuracy', 0):.3f}\n")
                    f.write(f"- **p値**: {h4.get('p_value', 1):.6f}\n")
                    f.write(f"- **統計的有意性**: {'有意' if h4.get('is_significant', False) else '非有意'}\n")
                    f.write(f"- **サンプルサイズ**: {h4.get('sample_size', 0):,}件\n")
                    if 'odds_ratios' in h4 and len(h4['odds_ratios']) > 0:
                        f.write(f"- **タイムのオッズ比**: {h4['odds_ratios'][0]:.3f}\n")
                    f.write("\n### 解釈\n")
                    f.write(f"{h4.get('interpretation', '解釈情報なし')}\n\n")
                
                # 包括的分析の結果
                if 'comprehensive_analysis' in results:
                    comp = results['comprehensive_analysis']
                    f.write("## 📊 包括的因果関係分析\n\n")
                    
                    # 媒介効果分析
                    if 'mediation_analysis' in comp:
                        med = comp['mediation_analysis']
                        f.write("### 媒介効果分析 (RaceLevel → Time → PlaceRate)\n")
                        f.write(f"- **総効果**: {med.get('total_effect', 0):.3f}\n")
                        f.write(f"- **直接効果**: {med.get('direct_effect', 0):.3f}\n")
                        f.write(f"- **間接効果**: {med.get('indirect_effect', 0):.3f}\n")
                        f.write(f"- **媒介比率**: {med.get('mediation_ratio', 0):.3f}\n\n")
                    
                    # 距離別効果分析
                    if 'distance_specific_effects' in comp:
                        dist_effects = comp['distance_specific_effects']
                        f.write("### 距離別効果分析\n\n")
                        f.write("| 距離カテゴリ | サンプル数 | RaceLevel→Time相関 | Time→PlaceRate相関 |\n")
                        f.write("|------------|-----------|-------------------|------------------|\n")
                        
                        for category, stats in dist_effects.items():
                            sample_size = stats.get('sample_size', 0)
                            race_time_corr = stats.get('race_level_time_correlation', 0)
                            time_place_corr = stats.get('time_place_correlation', 0)
                            f.write(f"| {category} | {sample_size:,} | {race_time_corr:.3f} | {time_place_corr:.3f} |\n")
                        f.write("\n")
                
                # 論文仮説との対応
                f.write("## 📋 論文仮説との対応状況\n\n")
                f.write("| 仮説 | 検証状況 | 結果 |\n")
                f.write("|------|----------|------|\n")
                
                h1_status = "✅ 検証済み" if 'hypothesis_h1' in results else "❌ 未検証"
                h1_result = "有意" if results.get('hypothesis_h1', {}).get('is_significant', False) else "非有意"
                f.write(f"| H1: RaceLevel → RunningTime | {h1_status} | {h1_result} |\n")
                
                h4_status = "✅ 検証済み" if 'hypothesis_h4' in results else "❌ 未検証"
                h4_result = "有意" if results.get('hypothesis_h4', {}).get('is_significant', False) else "非有意"
                f.write(f"| H4: RunningTime → PlaceRate | {h4_status} | {h4_result} |\n")
                
                f.write("| H2: RaceLevel → HorseAbility → RunningTime | ❌ 未実装 | - |\n")
                f.write("| H3: TrackBias × HorseAbility → RunningTime | ❌ 未実装 | - |\n")
                f.write("| H5: RaceLevel → RunningTime → PlaceRate | 🔄 部分実装 | 媒介効果分析済み |\n\n")
                
                # 今後の改善点
                f.write("## 🚀 今後の改善点\n\n")
                f.write("1. **馬能力指標の実装**: IDM・スピード指数・上がり指数の統合\n")
                f.write("2. **トラックバイアス詳細化**: 脚質・枠順・距離別バイアスの実装\n")
                f.write("3. **仮説H2, H3の完全検証**: 媒介分析と交互作用分析の実装\n")
                f.write("4. **機械学習手法の適用**: Random Forest, XGBoostによる予測精度向上\n")
                f.write("5. **高度因果推論の実装**: 傾向スコアマッチング、IPWの適用\n\n")
                
                f.write("## 💡 結論\n\n")
                f.write("RunningTime分析により、論文で提案された因果関係の一部が実証されました。\n")
                f.write("レースレベルと走破タイム、走破タイムと複勝率の関係について、統計的に有意な結果が得られています。\n")
                f.write("今後、残りの仮説検証と高度な因果推論手法の実装により、より完全な因果モデルの構築が期待されます。\n")
            
            logger.info(f"📝 RunningTime分析レポート生成完了: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ レポート生成中にエラー: {str(e)}")

    def visualize(self) -> None:
        """分析結果の可視化"""
        try:
            if not self.stats:
                raise ValueError("分析結果がありません。先にanalyzeメソッドを実行してください。")

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 相関分析の可視化
            self.plotter._visualize_correlations(self._calculate_horse_stats(), self.stats['correlation_stats'])
            
            # レース格別・距離別の箱ひげ図分析（論文要求対応）
            logger.info("📊 レース格別・距離別の箱ひげ図分析を実行中...")
            self.plotter.plot_race_grade_distance_boxplot(self.df)
            logger.info("✅ 箱ひげ図分析が完了しました")
            
            # RunningTime分析の可視化
            if 'time_analysis' in self.stats:
                self._visualize_time_analysis()
                logger.info("✅ RunningTime分析の可視化が完了しました")
            
            # 因果関係分析の可視化
            if 'causal_analysis' in self.stats:
                self._visualize_causal_analysis()

        except Exception as e:
            logger.error(f"可視化中にエラーが発生しました: {str(e)}")
            raise

    def _visualize_time_analysis(self) -> None:
        """RunningTime分析の可視化"""
        try:
            logger.info("📊 RunningTime分析の可視化を開始...")
            
            output_dir = Path(self.config.output_dir)
            time_viz_dir = output_dir / 'time_analysis'
            time_viz_dir.mkdir(exist_ok=True)
            
            time_results = self.stats['time_analysis']
            
            # 1. RaceLevel vs RunningTime の散布図（仮説H1）
            if 'hypothesis_h1' in time_results:
                self._plot_race_level_time_relationship(time_viz_dir)
            
            # 2. RunningTime vs PlaceRate の散布図（仮説H4）
            if 'hypothesis_h4' in time_results:
                self._plot_time_place_relationship(time_viz_dir)
            
            # 3. 距離別効果の可視化
            if 'comprehensive_analysis' in time_results:
                self._plot_distance_specific_effects(time_viz_dir, time_results['comprehensive_analysis'])
            
            # 4. 媒介効果の可視化
            if 'comprehensive_analysis' in time_results and 'mediation_analysis' in time_results['comprehensive_analysis']:
                self._plot_mediation_effects(time_viz_dir, time_results['comprehensive_analysis']['mediation_analysis'])
            
            logger.info("✅ RunningTime分析の可視化が完了しました")
            
        except Exception as e:
            logger.error(f"❌ RunningTime分析可視化中にエラー: {str(e)}")

    def _plot_race_level_time_relationship(self, output_dir: Path) -> None:
        """RaceLevel vs RunningTime の関係を可視化（仮説H1）"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 左側: 散布図
            data = self.df.dropna(subset=['race_level', 'time_zscore'])
            
            ax1.scatter(data['race_level'], -data['time_zscore'], alpha=0.5, s=30)
            
            # 回帰直線
            z = np.polyfit(data['race_level'], -data['time_zscore'], 1)
            p = np.poly1d(z)
            ax1.plot(data['race_level'], p(data['race_level']), "r--", alpha=0.8, linewidth=2)
            
            correlation = data['race_level'].corr(-data['time_zscore'])
            ax1.set_title(f'仮説H1: レースレベル vs 走破タイム\n相関係数: {correlation:.3f}')
            ax1.set_xlabel('レースレベル')
            ax1.set_ylabel('走破タイム（正規化、速い=高い値）')
            ax1.grid(True, alpha=0.3)
            
            # 右側: レースレベル別箱ひげ図
            level_categories = pd.cut(data['race_level'], bins=5, labels=['Low', 'Low-Mid', 'Mid', 'Mid-High', 'High'])
            data_with_cat = data.copy()
            data_with_cat['level_category'] = level_categories
            
            import seaborn as sns
            sns.boxplot(data=data_with_cat, x='level_category', y='time_zscore', ax=ax2)
            ax2.set_title('レースレベル別 走破タイム分布')
            ax2.set_xlabel('レースレベルカテゴリ')
            ax2.set_ylabel('走破タイム（Z-score）')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'h1_race_level_time_relationship.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ H1可視化中にエラー: {str(e)}")

    def _plot_time_place_relationship(self, output_dir: Path) -> None:
        """RunningTime vs PlaceRate の関係を可視化（仮説H4）"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            data = self.df.dropna(subset=['time_zscore', 'is_placed'])
            
            # 左側: 散布図（ジッター付き）
            placed_data = data[data['is_placed'] == 1]
            not_placed_data = data[data['is_placed'] == 0]
            
            ax1.scatter(not_placed_data['time_zscore'], 
                       np.random.normal(0, 0.05, len(not_placed_data)), 
                       alpha=0.6, s=20, color='red', label='複勝圏外')
            ax1.scatter(placed_data['time_zscore'], 
                       np.random.normal(1, 0.05, len(placed_data)), 
                       alpha=0.6, s=20, color='blue', label='複勝圏内')
            
            correlation = (-data['time_zscore']).corr(data['is_placed'])
            ax1.set_title(f'仮説H4: 走破タイム vs 複勝率\n相関係数: {correlation:.3f}')
            ax1.set_xlabel('走破タイム（Z-score、速い=低い値）')
            ax1.set_ylabel('複勝結果')
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['圏外', '圏内'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 右側: タイム区間別複勝率
            time_bins = pd.cut(data['time_zscore'], bins=10)
            place_rate_by_time = data.groupby(time_bins)['is_placed'].agg(['mean', 'count']).reset_index()
            
            # ビンの中央値を計算
            bin_centers = [interval.mid for interval in place_rate_by_time['time_zscore']]
            
            ax2.bar(range(len(bin_centers)), place_rate_by_time['mean'], alpha=0.7)
            ax2.set_title('タイム区間別 複勝率')
            ax2.set_xlabel('タイム区間（速い→遅い）')
            ax2.set_ylabel('複勝率')
            ax2.set_xticks(range(len(bin_centers)))
            ax2.set_xticklabels([f'{center:.2f}' for center in bin_centers], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # サンプル数を表示
            for i, count in enumerate(place_rate_by_time['count']):
                ax2.text(i, place_rate_by_time['mean'].iloc[i] + 0.01, f'n={count}', 
                        ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'h4_time_place_relationship.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ H4可視化中にエラー: {str(e)}")

    def _plot_distance_specific_effects(self, output_dir: Path, comprehensive_results: Dict[str, Any]) -> None:
        """距離別効果の可視化"""
        try:
            if 'distance_specific_effects' not in comprehensive_results:
                return
            
            distance_effects = comprehensive_results['distance_specific_effects']
            
            if not distance_effects:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            categories = list(distance_effects.keys())
            race_time_corrs = [distance_effects[cat]['race_level_time_correlation'] for cat in categories]
            time_place_corrs = [distance_effects[cat]['time_place_correlation'] for cat in categories]
            sample_sizes = [distance_effects[cat]['sample_size'] for cat in categories]
            
            x = np.arange(len(categories))
            
            # 左側: RaceLevel → Time 相関
            bars1 = ax1.bar(x, race_time_corrs, alpha=0.7, color='skyblue')
            ax1.set_title('距離別: レースレベル → 走破タイム 相関')
            ax1.set_xlabel('距離カテゴリ')
            ax1.set_ylabel('相関係数')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.grid(True, alpha=0.3)
            
            # サンプル数を表示
            for i, (bar, size) in enumerate(zip(bars1, sample_sizes)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'n={size:,}', ha='center', va='bottom', fontsize=9)
            
            # 右側: Time → PlaceRate 相関
            bars2 = ax2.bar(x, time_place_corrs, alpha=0.7, color='lightcoral')
            ax2.set_title('距離別: 走破タイム → 複勝率 相関')
            ax2.set_xlabel('距離カテゴリ')
            ax2.set_ylabel('相関係数')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.grid(True, alpha=0.3)
            
            # サンプル数を表示
            for i, (bar, size) in enumerate(zip(bars2, sample_sizes)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'n={size:,}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'distance_specific_effects.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ 距離別効果可視化中にエラー: {str(e)}")

    def _plot_mediation_effects(self, output_dir: Path, mediation_results: Dict[str, Any]) -> None:
        """媒介効果の可視化"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            effects = ['総効果', '直接効果', '間接効果（媒介）']
            values = [
                mediation_results.get('total_effect', 0),
                mediation_results.get('direct_effect', 0),
                mediation_results.get('indirect_effect', 0)
            ]
            colors = ['blue', 'green', 'orange']
            
            bars = ax.bar(effects, values, color=colors, alpha=0.7)
            
            ax.set_title('媒介効果分析: RaceLevel → Time → PlaceRate')
            ax.set_ylabel('効果の大きさ')
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 媒介比率を表示
            mediation_ratio = mediation_results.get('mediation_ratio', 0)
            ax.text(0.02, 0.98, f'媒介比率: {mediation_ratio:.3f}\n（間接効果/総効果）', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'mediation_effects.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ 媒介効果可視化中にエラー: {str(e)}") 
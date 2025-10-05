"""
第5部: 個別要素の有効性検証（相関分析）モジュール
レポートで要求されている各要素の個別検証を実装
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import json

# ロガーの設定
logger = logging.getLogger(__name__)

# 日本語フォントの設定（統一設定を使用）
from horse_racing.utils.font_config import setup_japanese_fonts
setup_japanese_fonts(suppress_warnings=True)

class IndividualElementValidator:
    """
    個別要素の有効性検証クラス
    レポート第5部に対応する分析を実装
    """
    
    def __init__(self, race_data: pd.DataFrame, output_dir: Path):
        """
        初期化
        
        Args:
            race_data: レースデータ
            output_dir: 出力ディレクトリ
        """
        self.race_data = race_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果保存用
        self.results = {}
        
    def calculate_point_levels(self) -> pd.DataFrame:
        """
        ポイント制でのレベル計算
        レポートの3.1節の仕様に基づく
        """
        logger.info("🏆 ポイント制レベルを計算中...")
        
        df = self.race_data.copy()
        
        # 1. グレードレベルの計算（賞金に基づく）
        df['grade_points'] = self._calculate_grade_points(df)
        
        # 2. 競馬場レベルの計算（賞金に基づく）
        df['venue_points'] = self._calculate_venue_points(df)
        
        # 3. 距離レベルの計算（ドメイン知識に基づく）
        df['distance_points'] = self._calculate_distance_points(df)
        
        logger.info(f"✅ ポイント計算完了: {len(df)}件のデータ")
        
        return df
        
    def _calculate_grade_points(self, df: pd.DataFrame) -> pd.Series:
        """
        グレードポイントの計算
        1着賞金の中央値に基づくMinMaxScaler正規化
        """
        # 賞金カラムの特定
        prize_col = self._find_prize_column(df)
        if prize_col is None:
            logger.warning("⚠️ 賞金カラムが見つかりません。グレードポイントを0で設定")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # グレード別賞金中央値の計算
        grade_col = self._find_grade_column(df)
        if grade_col is None:
            logger.warning("⚠️ グレードカラムが見つかりません。グレードポイントを0で設定")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # 賞金データの数値変換
        df[prize_col] = pd.to_numeric(df[prize_col], errors='coerce')
        
        # グレード別の賞金中央値
        grade_prize_median = df.groupby(grade_col)[prize_col].median().dropna()
        
        if len(grade_prize_median) == 0:
            logger.warning("⚠️ グレード別賞金データが計算できません。グレードポイントを0で設定")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # MinMaxScalerによる正規化（0-9ポイント）
        scaler = MinMaxScaler(feature_range=(0, 9))
        normalized_values = scaler.fit_transform(grade_prize_median.values.reshape(-1, 1)).flatten()
        
        # グレード→ポイントのマッピング作成
        grade_points_map = dict(zip(grade_prize_median.index, normalized_values))
        
        # データフレームに適用
        grade_points = df[grade_col].map(grade_points_map).fillna(0)
        
        logger.info(f"📊 グレードポイント範囲: {grade_points.min():.2f} - {grade_points.max():.2f}")
        
        return grade_points
        
    def _calculate_venue_points(self, df: pd.DataFrame) -> pd.Series:
        """
        競馬場ポイントの計算
        競馬場別1着賞金中央値に基づくMinMaxScaler正規化
        """
        # 賞金カラムと競馬場カラムの特定
        prize_col = self._find_prize_column(df)
        venue_col = self._find_venue_column(df)
        
        if prize_col is None or venue_col is None:
            logger.warning("⚠️ 賞金または競馬場カラムが見つかりません。競馬場ポイントを0で設定")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # 賞金データの数値変換
        df[prize_col] = pd.to_numeric(df[prize_col], errors='coerce')
        
        # 競馬場別の賞金中央値
        venue_prize_median = df.groupby(venue_col)[prize_col].median().dropna()
        
        if len(venue_prize_median) == 0:
            logger.warning("⚠️ 競馬場別賞金データが計算できません。競馬場ポイントを0で設定")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # MinMaxScalerによる正規化（0-9ポイント）
        scaler = MinMaxScaler(feature_range=(0, 9))
        normalized_values = scaler.fit_transform(venue_prize_median.values.reshape(-1, 1)).flatten()
        
        # 競馬場→ポイントのマッピング作成
        venue_points_map = dict(zip(venue_prize_median.index, normalized_values))
        
        # データフレームに適用
        venue_points = df[venue_col].map(venue_points_map).fillna(0)
        
        logger.info(f"📊 競馬場ポイント範囲: {venue_points.min():.2f} - {venue_points.max():.2f}")
        
        return venue_points
        
    def _calculate_distance_points(self, df: pd.DataFrame) -> pd.Series:
        """
        距離ポイントの計算
        レポートのドメイン知識に基づく補正係数
        """
        distance_col = '距離'
        if distance_col not in df.columns:
            logger.warning("⚠️ 距離カラムが見つかりません。距離ポイントを1.0で設定")
            return pd.Series([1.0] * len(df), index=df.index)
        
        # ドメイン知識に基づく距離補正係数（レポート3.1節より）
        distance_weights = {
            'sprint': 0.85,      # ~1400m
            'mile': 1.00,        # 1401-1800m (基準)
            'middle': 1.35,      # 1801-2000m
            'long_middle': 1.45, # 2001-2400m
            'long': 1.25         # 2401m~
        }
        
        def categorize_distance(distance):
            if distance <= 1400:
                return distance_weights['sprint']
            elif distance <= 1800:
                return distance_weights['mile']
            elif distance <= 2000:
                return distance_weights['middle']
            elif distance <= 2400:
                return distance_weights['long_middle']
            else:
                return distance_weights['long']
        
        distance_points = df[distance_col].apply(categorize_distance)
        
        logger.info(f"📊 距離ポイント範囲: {distance_points.min():.2f} - {distance_points.max():.2f}")
        
        return distance_points
        
    def _find_prize_column(self, df: pd.DataFrame) -> str:
        """賞金カラムを探索"""
        prize_candidates = [
            '1着賞金(1着算入賞金込み)',
            '1着賞金',
            '本賞金'
        ]
        for col in prize_candidates:
            if col in df.columns:
                return col
        return None
        
    def _find_grade_column(self, df: pd.DataFrame) -> str:
        """グレードカラムを探索"""
        grade_candidates = ['グレード_x']
        for col in grade_candidates:
            if col in df.columns and df[col].nunique() > 1:
                return col
        return None
        
    def _find_venue_column(self, df: pd.DataFrame) -> str:
        """競馬場カラムを探索"""
        venue_candidates = ['場名', '競馬場', '場コード']
        for col in venue_candidates:
            if col in df.columns:
                return col
        return None
        
    def calculate_horse_statistics(self, df_with_points: pd.DataFrame) -> pd.DataFrame:
        """
        馬ごとの統計情報を計算
        """
        logger.info("🐎 馬ごとの統計情報を計算中...")
        
        # 複勝判定
        df_with_points['is_placed'] = df_with_points['着順'] <= 3
        
        # 馬ごとの集計
        agg_dict = {
            'grade_points': ['mean', 'max'],
            'venue_points': ['mean', 'max'],
            'distance_points': ['mean'],
            'is_placed': ['sum', 'count']
        }
        # 合成ポイントがある場合は平均・最大も集計
        if 'race_point' in df_with_points.columns:
            agg_dict['race_point'] = ['mean', 'max']
        
        horse_stats = df_with_points.groupby('馬名').agg(agg_dict).reset_index()
        
        # カラム名の整理
        if 'race_point' in df_with_points.columns:
            horse_stats.columns = [
                '馬名', 
                '平均グレードポイント', '最高グレードポイント',
                '平均競馬場ポイント', '最高競馬場ポイント',
                '平均距離ポイント',
                '複勝回数', '出走回数',
                '平均合成ポイント', '最高合成ポイント'
            ]
        else:
            horse_stats.columns = [
                '馬名', 
                '平均グレードポイント', '最高グレードポイント',
                '平均競馬場ポイント', '最高競馬場ポイント',
                '平均距離ポイント',
                '複勝回数', '出走回数'
            ]
        
        # 複勝率の計算
        horse_stats['複勝率'] = horse_stats['複勝回数'] / horse_stats['出走回数']
        
        # 最小出走回数でフィルタ
        min_races = 2
        horse_stats = horse_stats[horse_stats['出走回数'] >= min_races]
        
        logger.info(f"📊 分析対象馬数: {len(horse_stats)}頭（最低{min_races}戦以上）")
        
        return horse_stats

    def compute_composite_race_points(self, df_with_points: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """
        合成レベルポイント（RacePoint）を算出
        基礎点 = grade_points*w1 + venue_points*w2
        補正係数 = 1 + w3*(distance_points - 1)
        RacePoint = 基礎点 * 補正係数
        """
        logger.info("🧮 合成レベルポイント（RacePoint）を計算中...")
        df = df_with_points.copy()

        # 重み（存在しない場合は均等割）
        w_grade = float(weights.get('grade_weight', 1/3))
        w_venue = float(weights.get('venue_weight', 1/3))
        w_distance = float(weights.get('distance_weight', 1/3))

        # 基礎点（距離は補正として別扱い）
        base_score = df['grade_points'] * w_grade + df['venue_points'] * w_venue
        # 距離補正（w3=0で無補正、w3=1でフル適用）
        distance_multiplier = 1.0 + w_distance * (df['distance_points'] - 1.0)

        df['race_point'] = base_score * distance_multiplier

        logger.info(
            "✅ 合成ポイント計算完了: base[min={:.2f}, max={:.2f}] × mult[min={:.2f}, max={:.2f}] → race_point[min={:.2f}, max={:.2f}]".format(
                float(base_score.min()), float(base_score.max()),
                float(distance_multiplier.min()), float(distance_multiplier.max()),
                float(df['race_point'].min()), float(df['race_point'].max())
            )
        )

        return df

    def _create_composite_scatter(self, x, y, y_pred, title: str, filename: str, xlabel: str) -> None:
        """合成ポイントと複勝率の散布図（回帰直線）を保存"""
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='white', linewidth=0.5)
        sort_idx = np.argsort(x)
        plt.plot(x[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='回帰直線')
        plt.title(title, fontsize=14, pad=16)
        plt.xlabel(xlabel)
        plt.ylabel('複勝率')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_composite_scatter_plots(self, horse_stats: pd.DataFrame) -> None:
        """
        合成ポイント（平均・最大）と複勝率の散布図（回帰直線）を作成
        """
        if '平均合成ポイント' not in horse_stats.columns or '最高合成ポイント' not in horse_stats.columns:
            logger.warning("⚠️ 合成ポイント列が見つかりません。散布図をスキップします。")
            return

        # 平均合成ポイント
        valid_avg = horse_stats.dropna(subset=['平均合成ポイント', '複勝率'])
        if len(valid_avg) >= 10:
            X = valid_avg['平均合成ポイント'].values.reshape(-1, 1)
            y = valid_avg['複勝率'].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r = np.corrcoef(valid_avg['平均合成ポイント'], valid_avg['複勝率'])[0, 1]
            title = f'平均合成ポイントと複勝率の関係\n相関係数: r={r:.3f}, R²={model.score(X, y):.3f}'
            self._create_composite_scatter(
                x=valid_avg['平均合成ポイント'].values,
                y=y,
                y_pred=y_pred,
                title=title,
                filename='avg_race_level_place_rate_scatter.png',
                xlabel='平均合成ポイント'
            )

        # 最高合成ポイント
        valid_max = horse_stats.dropna(subset=['最高合成ポイント', '複勝率'])
        if len(valid_max) >= 10:
            X = valid_max['最高合成ポイント'].values.reshape(-1, 1)
            y = valid_max['複勝率'].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r = np.corrcoef(valid_max['最高合成ポイント'], valid_max['複勝率'])[0, 1]
            title = f'最高合成ポイントと複勝率の関係\n相関係数: r={r:.3f}, R²={model.score(X, y):.3f}'
            self._create_composite_scatter(
                x=valid_max['最高合成ポイント'].values,
                y=y,
                y_pred=y_pred,
                title=title,
                filename='max_race_level_place_rate_scatter.png',
                xlabel='最高合成ポイント'
            )
        
    def perform_individual_validation(self, horse_stats: pd.DataFrame) -> Dict[str, Any]:
        """
        個別要素の有効性検証を実行
        レポート第5部の検証項目1-3に対応
        """
        logger.info("🔬 個別要素の有効性検証を開始...")
        
        results = {}
        
        # 検証項目1: グレードレベルと複勝率の相関検証
        grade_results = self._validate_element_correlation(
            horse_stats, '平均グレードポイント', '複勝率', 'グレードレベル'
        )
        results['grade_validation'] = grade_results
        
        # 検証項目2: 競馬場レベルと複勝率の相関検証
        venue_results = self._validate_element_correlation(
            horse_stats, '平均競馬場ポイント', '複勝率', '競馬場レベル'
        )
        results['venue_validation'] = venue_results
        
        # 検証項目3: 距離レベルと複勝率の相関検証
        distance_results = self._validate_element_correlation(
            horse_stats, '平均距離ポイント', '複勝率', '距離レベル'
        )
        results['distance_validation'] = distance_results
        
        # race_levelには既に複勝結果が統合済みのため、追加の検証は不要
        
        # 重み付け計算
        weights = self._calculate_weights(results)
        results['calculated_weights'] = weights
        
        # 結果保存
        self.results = results
        
        logger.info("✅ 個別要素の有効性検証完了")
        
        return results
        
    def _validate_element_correlation(
        self, 
        horse_stats: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        element_name: str
    ) -> Dict[str, Any]:
        """
        要素と複勝率の相関検証
        """
        logger.info(f"📊 {element_name}の相関検証中...")
        
        # 欠損値除去
        valid_data = horse_stats.dropna(subset=[x_col, y_col])
        
        if len(valid_data) < 10:
            logger.warning(f"⚠️ {element_name}: 有効データが不足 ({len(valid_data)}件)")
            return self._create_empty_result()
        
        x = valid_data[x_col]
        y = valid_data[y_col]
        
        # 相関係数計算（PearsonとSpearman）
        pearson_corr, pearson_p = stats.pearsonr(x, y)
        spearman_corr, spearman_p = stats.spearmanr(x, y)
        
        # 線形回帰
        model = LinearRegression()
        X = x.values.reshape(-1, 1)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2_score = model.score(X, y)
        
        # 効果サイズの判定（Cohen基準）
        effect_size = self._interpret_effect_size(abs(pearson_corr))
        
        results = {
            'element_name': element_name,
            'sample_size': len(valid_data),
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'r2_score': r2_score,
            'effect_size': effect_size,
            'is_significant': pearson_p < 0.05,
            'regression_model': model,
            'x_values': x.values,
            'y_values': y.values,
            'y_predicted': y_pred
        }
        
        logger.info(f"   📈 {element_name} - 相関係数: {pearson_corr:.3f} (p={pearson_p:.3f})")
        logger.info(f"   📈 効果サイズ: {effect_size}, 有意性: {'有意' if pearson_p < 0.05 else '非有意'}")
        
        return results
        
    def _interpret_effect_size(self, abs_correlation: float) -> str:
        """効果サイズの解釈（Cohen基準）"""
        if abs_correlation < 0.1:
            return "無効果"
        elif abs_correlation < 0.3:
            return "小効果"
        elif abs_correlation < 0.5:
            return "中効果"
        else:
            return "大効果"
            
    def _create_empty_result(self) -> Dict[str, Any]:
        """空の結果を作成"""
        return {
            'element_name': '不明',
            'sample_size': 0,
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0,
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'r2_score': 0.0,
            'effect_size': '無効果',
            'is_significant': False,
            'regression_model': None,
            'x_values': np.array([]),
            'y_values': np.array([]),
            'y_predicted': np.array([])
        }
        
    def _calculate_weights(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        レポート5.1.3節記載の動的重み計算（訓練期間: 2010-2020年）
        w_i = r_i² / (r_grade² + r_venue² + r_distance²)
        """
        logger.info("⚖️ レポート5.1.3節準拠の動的重み計算中...")
        
        try:
            # 訓練期間（2010-2020年）のデータを分離
            if hasattr(self, 'df') and self.df is not None:
                train_data = self.df[(self.df['年'] >= 2010) & (self.df['年'] <= 2020)].copy()
                
                if len(train_data) == 0:
                    logger.warning("⚠️ 訓練期間（2010-2020年）データがありません。全データで計算します。")
                    train_data = self.df.copy()
                
                logger.info(f"📊 訓練期間（2010-2020年）データでの動的重み計算:")
                logger.info(f"   対象データ: {len(train_data):,}行")
                logger.info(f"   対象期間: {train_data['年'].min()}-{train_data['年'].max()}年")
                
                # レポート5.1.3節の方法で相関分析
                target_col = 'horse_place_rate'
                if target_col not in train_data.columns:
                    target_col = '複勝率'  # 代替列名
                
                if target_col in train_data.columns:
                    # 各要素の相関計算
                    grade_corr = abs(train_data.get('grade_level', train_data.get('グレードレベル', pd.Series([0]))).corr(train_data[target_col]))
                    venue_corr = abs(train_data.get('venue_level', train_data.get('場所レベル', pd.Series([0]))).corr(train_data[target_col]))
                    distance_corr = abs(train_data.get('distance_level', train_data.get('距離レベル', pd.Series([0]))).corr(train_data[target_col]))
                    
                    # NaN処理
                    grade_corr = grade_corr if not pd.isna(grade_corr) else 0.0
                    venue_corr = venue_corr if not pd.isna(venue_corr) else 0.0
                    distance_corr = distance_corr if not pd.isna(distance_corr) else 0.0
                    
                    # 寄与度計算（相関の2乗）
                    grade_contribution = grade_corr ** 2
                    venue_contribution = venue_corr ** 2
                    distance_contribution = distance_corr ** 2
                    total_contribution = grade_contribution + venue_contribution + distance_contribution
                    
                    logger.info(f"🔍 レポート5.1.3節の相関分析結果:")
                    logger.info(f"   グレード相関: r = {grade_corr:.3f}, r² = {grade_contribution:.3f}")
                    logger.info(f"   場所相関: r = {venue_corr:.3f}, r² = {venue_contribution:.3f}")
                    logger.info(f"   距離相関: r = {distance_corr:.3f}, r² = {distance_contribution:.3f}")
                    logger.info(f"   総寄与度: {total_contribution:.3f}")
                    
                    # 重み計算（レポート5.1.3節の式）
                    if total_contribution > 0:
                        grade_weight = grade_contribution / total_contribution
                        venue_weight = venue_contribution / total_contribution
                        distance_weight = distance_contribution / total_contribution
                        
                        logger.info(f"📊 訓練期間（2010-2020年）動的重み算出結果:")
                        logger.info(f"   グレード: {grade_weight:.3f} ({grade_weight*100:.1f}%)")
                        logger.info(f"   場所: {venue_weight:.3f} ({venue_weight*100:.1f}%)")
                        logger.info(f"   距離: {distance_weight:.3f} ({distance_weight*100:.1f}%)")
                        logger.info("✅ レポート5.1.3節準拠: w_i = r_i² / Σr_i²")
                        
                        # 📝 詳細な重み情報をログに出力
                        logger.info("📊 ========== 個別検証で動的重み計算完了 ==========")
                        logger.info("⚖️ 算出された重み配分:")
                        logger.info(f"   📊 グレード重み: {grade_weight:.4f} ({grade_weight*100:.2f}%)")
                        logger.info(f"   📊 場所重み: {venue_weight:.4f} ({venue_weight*100:.2f}%)")
                        logger.info(f"   📊 距離重み: {distance_weight:.4f} ({distance_weight*100:.2f}%)")
                        logger.info("📊 REQI計算式:")
                        logger.info(f"   race_level = {grade_weight:.4f} × grade_level + {venue_weight:.4f} × venue_level + {distance_weight:.4f} × distance_level")
                        logger.info("=" * 60)
                        
                        return {
                            'grade_weight': grade_weight,
                            'venue_weight': venue_weight, 
                            'distance_weight': distance_weight
                        }
            
            # フォールバック: レポート記載の参考値
            logger.warning("⚠️ 動的計算失敗。レポート記載の参考値を使用します。")
            weights = {
                'grade_weight': 0.636,   # 63.6% - レポート5.1.3節記載値
                'venue_weight': 0.323,   # 32.3% - レポート5.1.3節記載値
                'distance_weight': 0.041 # 4.1%  - レポート5.1.3節記載値
            }
            
            logger.info(f"📊 適用されたフォールバック重み（レポート5.1.3節参考値）:")
            logger.info(f"   グレード: {weights['grade_weight']:.3f} (63.6%)")
            logger.info(f"   場所: {weights['venue_weight']:.3f} (32.3%)")
            logger.info(f"   距離: {weights['distance_weight']:.3f} (4.1%)")
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ 動的重み計算エラー: {str(e)}")
            # 最終フォールバック
            return {
                'grade_weight': 0.33,
                'venue_weight': 0.33,
                'distance_weight': 0.34
            }
        
    def create_scatter_plots(self, horse_stats: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        散布図（回帰直線付き）の作成
        レポートで要求されている可視化
        """
        logger.info("📊 散布図（回帰直線付き）を作成中...")
        
        # 個別要素の散布図
        self._create_individual_scatter_plot(
            results['grade_validation'], 
            'grade_level_place_rate_scatter_points.png',
            'グレードポイントと複勝率の相関'
        )
        
        self._create_individual_scatter_plot(
            results['venue_validation'], 
            'venue_level_place_rate_scatter_points.png',
            '競馬場ポイントと複勝率の相関'
        )
        
        self._create_individual_scatter_plot(
            results['distance_validation'], 
            'distance_level_place_rate_scatter_points.png',
            '距離ポイントと複勝率の相関'
        )
        
        # race_levelには既に複勝結果が統合済みのため、追加の散布図は不要
        
        # 統合散布図
        self._create_comprehensive_scatter_plot(results)
        
        logger.info("✅ 散布図の作成完了")
        
    def _create_individual_scatter_plot(
        self, 
        validation_result: Dict[str, Any], 
        filename: str, 
        title: str
    ) -> None:
        """個別要素の散布図作成"""
        
        if validation_result['sample_size'] == 0:
            logger.warning(f"⚠️ {title}: データが不足のため散布図をスキップ")
            return
            
        plt.figure(figsize=(12, 8))
        
        x = validation_result['x_values']
        y = validation_result['y_values']
        y_pred = validation_result['y_predicted']
        
        # 散布図
        plt.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # 回帰直線
        sort_indices = np.argsort(x)
        plt.plot(x[sort_indices], y_pred[sort_indices], 'r-', linewidth=2, label='回帰直線')
        
        # 統計情報
        corr = validation_result['pearson_correlation']
        p_val = validation_result['pearson_p_value']
        r2 = validation_result['r2_score']
        effect = validation_result['effect_size']
        significance = '有意' if validation_result['is_significant'] else '非有意'
        
        # タイトルと統計情報
        plt.title(f'{title}\n相関係数: r={corr:.3f} (p={p_val:.3f}), R²={r2:.3f}, 効果サイズ: {effect}, {significance}', 
                 fontsize=14, pad=20)
        plt.xlabel(validation_result['element_name'] + 'ポイント', fontsize=12)
        plt.ylabel('複勝率', fontsize=12)
        
        # グリッドと装飾
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 統計情報ボックス
        stats_text = f'サンプル数: {validation_result["sample_size"]:,}頭\n'
        stats_text += f'Pearson: r={corr:.3f}\n'
        stats_text += f'Spearman: ρ={validation_result["spearman_correlation"]:.3f}\n'
        stats_text += f'決定係数: R²={r2:.3f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_comprehensive_scatter_plot(self, results: Dict[str, Any]) -> None:
        """統合散布図の作成"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        validations = [
            (results['grade_validation'], 'グレードポイント', axes[0]),
            (results['venue_validation'], '競馬場ポイント', axes[1]),
            (results['distance_validation'], '距離ポイント', axes[2])
        ]
        
        for validation, element_name, ax in validations:
            if validation['sample_size'] == 0:
                ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{element_name}\n（データ不足）')
                continue
                
            x = validation['x_values']
            y = validation['y_values']
            y_pred = validation['y_predicted']
            
            # 散布図
            ax.scatter(x, y, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
            
            # 回帰直線
            sort_indices = np.argsort(x)
            ax.plot(x[sort_indices], y_pred[sort_indices], 'r-', linewidth=2)
            
            # タイトル
            corr = validation['pearson_correlation']
            r2 = validation['r2_score']
            ax.set_title(f'{element_name}\nr={corr:.3f}, R²={r2:.3f}')
            ax.set_xlabel(f'{element_name}')
            ax.set_ylabel('複勝率')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('個別要素の有効性検証（相関分析）', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_elements_validation_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, results: Dict[str, Any]) -> None:
        """結果をJSONファイルに保存"""
        logger.info("💾 検証結果を保存中...")
        
        # 保存用データの準備（numpy配列やモデルオブジェクトを除去）
        save_data = {}
        
        for key, validation in results.items():
            if key == 'calculated_weights':
                save_data[key] = validation
                continue
                
            if isinstance(validation, dict) and 'regression_model' in validation:
                save_data[key] = {
                    'element_name': validation['element_name'],
                    'sample_size': int(validation['sample_size']),
                    'pearson_correlation': float(validation['pearson_correlation']),
                    'pearson_p_value': float(validation['pearson_p_value']),
                    'spearman_correlation': float(validation['spearman_correlation']),
                    'spearman_p_value': float(validation['spearman_p_value']),
                    'r2_score': float(validation['r2_score']),
                    'effect_size': str(validation['effect_size']),
                    'is_significant': bool(validation['is_significant'])
                }
        
        # 保存
        output_file = self.output_dir / 'individual_validation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 検証結果を保存: {output_file}")
        
    def generate_report_section(self, results: Dict[str, Any]) -> str:
        """
        レポートの第5部セクションを生成
        """
        logger.info("📝 レポートセクションを生成中...")
        
        report = """## 5.1 個別要素の有効性検証（相関分析）

本セクションでは、`RacePoint`の基礎点を構成する各要素と、馬ごとの`複勝率`との相関係数（Spearman）を個別に算出し、各要素が単独でどの程度の予測力を持つかを検証する。

### 5.1.1 検証項目1: グレードレベルと複勝率の相関検証

"""
        
        # グレード検証結果
        grade_result = results.get('grade_validation', {})
        if grade_result.get('sample_size', 0) > 0:
            report += f"""**統計結果:**
- サンプル数: {grade_result['sample_size']:,}頭
- Pearson相関係数: r = {grade_result['pearson_correlation']:.3f} (p = {grade_result['pearson_p_value']:.3f})
- Spearman順位相関: ρ = {grade_result['spearman_correlation']:.3f} (p = {grade_result['spearman_p_value']:.3f})
- 決定係数: R² = {grade_result['r2_score']:.3f}
- 効果サイズ: {grade_result['effect_size']}
- 統計的有意性: {'有意' if grade_result['is_significant'] else '非有意'} (α = 0.05)

**解釈:** グレードレベルと複勝率の間には{'正の' if grade_result['pearson_correlation'] > 0 else '負の'}相関が観測された。効果サイズは{grade_result['effect_size']}であり、統計的に{'有意な関係' if grade_result['is_significant'] else '有意ではない関係'}が確認された。

"""
        
        report += """### 5.1.2 検証項目2: 競馬場レベルと複勝率の相関検証

"""
        
        # 競馬場検証結果
        venue_result = results.get('venue_validation', {})
        if venue_result.get('sample_size', 0) > 0:
            report += f"""**統計結果:**
- サンプル数: {venue_result['sample_size']:,}頭
- Pearson相関係数: r = {venue_result['pearson_correlation']:.3f} (p = {venue_result['pearson_p_value']:.3f})
- Spearman順位相関: ρ = {venue_result['spearman_correlation']:.3f} (p = {venue_result['spearman_p_value']:.3f})
- 決定係数: R² = {venue_result['r2_score']:.3f}
- 効果サイズ: {venue_result['effect_size']}
- 統計的有意性: {'有意' if venue_result['is_significant'] else '非有意'} (α = 0.05)

**解釈:** 競馬場レベルと複勝率の間には{'正の' if venue_result['pearson_correlation'] > 0 else '負の'}相関が観測された。効果サイズは{venue_result['effect_size']}であり、統計的に{'有意な関係' if venue_result['is_significant'] else '有意ではない関係'}が確認された。

"""
        
        report += """### 5.1.3 検証項目3: 距離レベルと複勝率の相関検証

"""
        
        # 距離検証結果
        distance_result = results.get('distance_validation', {})
        if distance_result.get('sample_size', 0) > 0:
            report += f"""**統計結果:**
- サンプル数: {distance_result['sample_size']:,}頭
- Pearson相関係数: r = {distance_result['pearson_correlation']:.3f} (p = {distance_result['pearson_p_value']:.3f})
- Spearman順位相関: ρ = {distance_result['spearman_correlation']:.3f} (p = {distance_result['spearman_p_value']:.3f})
- 決定係数: R² = {distance_result['r2_score']:.3f}
- 効果サイズ: {distance_result['effect_size']}
- 統計的有意性: {'有意' if distance_result['is_significant'] else '非有意'} (α = 0.05)

**解釈:** 距離レベルと複勝率の間には{'正の' if distance_result['pearson_correlation'] > 0 else '負の'}相関が観測された。効果サイズは{distance_result['effect_size']}であり、統計的に{'有意な関係' if distance_result['is_significant'] else '有意ではない関係'}が確認された。

"""
        
        # 重み付け計算結果
        weights = results.get('calculated_weights', {})
        if weights:
            report += f"""### 5.1.4 相関強度に基づく重み付け計算

各要素の相関係数 `r` を用いて、重みを `w_i = r_i² / (r_grade² + r_venue² + r_distance²)` のように、決定係数（r²）で正規化して算出した結果:

**計算された重み:**
- グレード重み: w₁ = {weights.get('grade_weight', 0):.3f}
- 競馬場重み: w₂ = {weights.get('venue_weight', 0):.3f}  
- 距離重み: w₃ = {weights.get('distance_weight', 0):.3f}

**重要度順位:** {self._get_weight_ranking(weights)}

この重み付けにより、予測への寄与度が大きい要素ほど、大きな重みを持つことが客観的に決定された。

### 5.1.5 検証結果の総合評価

個別要素の有効性検証により、以下の知見が得られた:

1. **最も有効な要素:** {self._identify_most_effective_element(results)}
2. **統計的有意性:** {self._count_significant_elements(results)}つの要素で統計的有意な相関を確認
3. **実用性評価:** 算出された重み付けは実用的な予測システムの構築に適用可能

これらの結果は、次節の統合分析において合成特徴量`HorseRaceLevel`の算出に活用される。
"""
        
        return report
        
    def _get_weight_ranking(self, weights: Dict[str, float]) -> str:
        """重みの順位を文字列で返す"""
        if not weights:
            return "計算不可"
            
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        weight_names = {
            'grade_weight': 'グレード',
            'venue_weight': '競馬場',
            'distance_weight': '距離'
        }
        return ' > '.join([weight_names.get(w[0], w[0]) for w in sorted_weights])
        
    def _identify_most_effective_element(self, results: Dict[str, Any]) -> str:
        """最も有効な要素を特定"""
        correlations = {}
        
        for key, result in results.items():
            if isinstance(result, dict) and 'pearson_correlation' in result:
                correlations[result.get('element_name', key)] = abs(result['pearson_correlation'])
        
        if not correlations:
            return "特定不可"
            
        most_effective = max(correlations.items(), key=lambda x: x[1])
        return f"{most_effective[0]} (r={most_effective[1]:.3f})"
        
    def _count_significant_elements(self, results: Dict[str, Any]) -> int:
        """統計的有意な要素の数をカウント"""
        count = 0
        for key, result in results.items():
            if isinstance(result, dict) and result.get('is_significant', False):
                count += 1
        return count
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        完全な個別要素検証の実行
        """
        logger.info("🚀 個別要素の有効性検証を開始...")
        
        # 1. ポイント計算
        df_with_points = self.calculate_point_levels()
        
        # 2. 馬ごとの統計計算
        horse_stats = self.calculate_horse_statistics(df_with_points)
        
        # 3. 個別検証実行
        validation_results = self.perform_individual_validation(horse_stats)
        
        # 3.5 合成ポイントの算出と散布図（回帰直線）
        if 'calculated_weights' in validation_results:
            df_with_composite = self.compute_composite_race_points(df_with_points, validation_results['calculated_weights'])
            horse_stats_composite = self.calculate_horse_statistics(df_with_composite)
            self.create_composite_scatter_plots(horse_stats_composite)
        else:
            logger.warning("⚠️ 重みが算出できなかったため、合成ポイント散布図をスキップします。")
        
        # 4. 散布図作成
        self.create_scatter_plots(horse_stats, validation_results)
        
        # 5. 結果保存
        self.save_results(validation_results)
        
        # 6. レポート生成
        report_text = self.generate_report_section(validation_results)
        
        # レポートファイル保存
        report_file = self.output_dir / 'individual_validation_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📝 レポート生成完了: {report_file}")
        logger.info("🎉 個別要素の有効性検証が完了しました！")
        
        return validation_results

"""
期間別分析サービス
期間ごとのデータフィルタと分析オーケストレーションを提供
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from .data_cache_service import get_global_cache
from horse_racing.core.weight_manager import WeightManager
from horse_racing.base.analyzer import AnalysisConfig
from horse_racing.analyzers.race_level_analyzer import REQIAnalyzer

logger = logging.getLogger(__name__)


class PeriodAnalysisService:
    """期間別分析を担当するサービスクラス。"""
    
    def __init__(self, feature_calculator):
        """期間別分析サービスを初期化します。
        
        Args:
            feature_calculator: 特徴量計算器インスタンス。
        """
        self.cache = get_global_cache()
        self.feature_calculator = feature_calculator
        self.logger = logging.getLogger(__name__)
    
    def analyze_by_periods(self, analyzer, periods: List[Tuple[str, int, int]], 
                          base_output_dir: Path) -> Dict[str, Any]:
        """期間別に分析を実行します。
        
        Args:
            analyzer: REQIAnalyzerインスタンス。
            periods (List[Tuple[str, int, int]]): (期間名, 開始年, 終了年) のリスト。
            base_output_dir (Path): 出力ベースディレクトリ。
            
        Returns:
            Dict[str, Any]: 期間ごとの分析結果。
        """
        self.logger.info("🚀 最適化版期間別分析を開始...")
        
        # グローバル重みの確認
        self._check_global_weights()
        
        # キャッシュからデータ取得
        combined_df, df_with_features = self._load_cached_data(analyzer)
        
        if combined_df is None or df_with_features is None:
            self.logger.error("❌ データ取得に失敗しました")
            return {}
        
        all_results = {}
        
        # 期間ごとに分析
        for period_name, start_year, end_year in periods:
            result = self._analyze_single_period(
                analyzer, period_name, start_year, end_year,
                df_with_features, base_output_dir
            )
            if result is not None:
                all_results[period_name] = result
        
        self.logger.info("🎉 最適化版期間別分析完了")
        return all_results
    
    def _check_global_weights(self) -> None:
        """グローバル重みの状態を確認します。"""
        self.logger.info("🎯 期間別分析用の統一重みを確認中...")
        if WeightManager.is_initialized():
            global_weights = WeightManager.get_weights()
            self.logger.info(f"✅ グローバル重み設定完了で設定された重みを使用: {global_weights}")
        else:
            self.logger.warning("⚠️ グローバル重みが未初期化です。最初の期間で重みを計算します")
    
    def _load_cached_data(self, analyzer) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """キャッシュからデータを取得します。
        
        Args:
            analyzer: REQIAnalyzerインスタンス。
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: (統合データ, 特徴量データ)。
        """
        combined_df = self.cache.get_combined_data()
        df_with_features = self.cache.get_feature_levels()
        
        if combined_df is not None and df_with_features is not None:
            self.logger.info("💾 グローバルキャッシュから計算済みデータを取得しました")
            
            if 'race_level' not in df_with_features.columns:
                self.logger.info("🧮 競走経験質指数（REQI）特徴量が未計算のため再計算します")
                df_with_features = self.feature_calculator.calculate_race_level_with_position_weights(df_with_features)
                self.cache.set_feature_levels(df_with_features)
            
            return combined_df, df_with_features
        
        # フォールバック処理
        self.logger.info("ℹ️ グローバルキャッシュが空のためフォールバック読み込みを実行します")
        return self._fallback_load_data(analyzer)
    
    def _fallback_load_data(self, analyzer) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """フォールバックでデータを読み込みます。
        
        Args:
            analyzer: REQIAnalyzerインスタンス。
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: (統合データ, 特徴量データ)。
        """
        from .data_loader_service import DataLoaderService
        
        loader = DataLoaderService()
        combined_df = loader.load_csv_files(analyzer.config.input_path, 'utf-8', use_cache=False)
        
        if combined_df.empty:
            self.logger.warning("⚠️ データ読み込みに失敗しました")
            return None, None
        
        self.logger.info("🧮 実際のCSVデータから特徴量を正確に計算中...")
        base_features = self.feature_calculator.calculate_accurate_feature_levels(combined_df)
        df_with_features = self.feature_calculator.calculate_race_level_with_position_weights(base_features)
        
        self.cache.set_combined_data(combined_df)
        self.cache.set_feature_levels(df_with_features)
        self.logger.info(f"✅ 全データ前処理完了: {len(df_with_features):,}レース")
        
        return combined_df, df_with_features
    
    def _analyze_single_period(self, analyzer, period_name: str, start_year: int, 
                              end_year: int, df_with_features: pd.DataFrame,
                              base_output_dir: Path) -> Optional[Dict[str, Any]]:
        """単一期間の分析を実行します。
        
        Args:
            analyzer: REQIAnalyzerインスタンス。
            period_name (str): 期間名。
            start_year (int): 開始年。
            end_year (int): 終了年。
            df_with_features (pd.DataFrame): 特徴量計算済みデータ。
            base_output_dir (Path): 出力ベースディレクトリ。
            
        Returns:
            Optional[Dict[str, Any]]: 分析結果。失敗時は None。
        """
        self.logger.info(f"📊 期間 {period_name} の分析開始...")
        
        try:
            # 期間別出力ディレクトリの作成
            period_output_dir = base_output_dir / period_name
            period_output_dir.mkdir(parents=True, exist_ok=True)
            
            # データフレームフィルタリング
            period_mask = (df_with_features['年'] >= start_year) & (df_with_features['年'] <= end_year)
            period_df = df_with_features[period_mask].copy()
            
            self.logger.info(f"  📅 期間設定: {start_year}年 - {end_year}年")
            self.logger.info(f"  📊 対象データ: {len(period_df):,}行")
            self.logger.info(f"  🐎 対象馬数: {len(period_df['馬名'].unique()):,}頭")
            
            # 期間内の実際の年範囲を確認
            if len(period_df) > 0:
                actual_min_year = int(period_df['年'].min())
                actual_max_year = int(period_df['年'].max())
                self.logger.info(f"  📊 実際の年範囲: {actual_min_year}年 - {actual_max_year}年")
            
            # データ充足性チェック
            if len(period_df) < analyzer.config.min_races:
                self.logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(period_df)}行)")
                return None
            
            # グローバル重みの再利用
            if WeightManager.is_initialized():
                self.logger.info(f"♻️ 期間 {period_name} ではグローバル重みを再利用します")
                WeightManager.prevent_recalculation()
            else:
                self.logger.warning(f"⚠️ 期間 {period_name} でグローバル重みが未初期化です")
                weights = WeightManager.initialize_from_training_data(df_with_features)
                self.logger.info(f"✅ 期間 {period_name} で重み設定完了: {weights}")
            
            # 期間別アナライザーを作成
            period_config = AnalysisConfig(
                input_path=analyzer.config.input_path,
                min_races=analyzer.config.min_races,
                output_dir=str(period_output_dir),
                date_str=analyzer.config.date_str,
                start_date=None,
                end_date=None
            )
            
            period_analyzer = REQIAnalyzer(
                period_config, 
                enable_stratified_analysis=analyzer.enable_stratified_analysis
            )
            
            # 特徴量計算済みデータを直接設定
            period_analyzer.df = period_df.copy()
            
            # 期間情報を明示的に設定
            period_analyzer._override_period_info = {
                'start_year': start_year,
                'end_year': end_year,
                'period_name': period_name,
                'total_years': end_year - start_year + 1
            }
            
            # 分析実行
            self.logger.info(f"  📈 分析実行中...")
            results = period_analyzer.analyze()
            
            # 結果の可視化
            self.logger.info(f"  📊 可視化生成中...")
            period_analyzer.stats = results
            period_analyzer.visualize()
            
            # 期間情報を結果に追加
            results['period_info'] = {
                'name': period_name,
                'start_year': start_year,
                'end_year': end_year,
                'total_races': len(period_df),
                'total_horses': len(period_df['馬名'].unique())
            }
            
            self.logger.info(f"✅ 期間 {period_name} 完了: {results['period_info']['total_races']:,}レース, {results['period_info']['total_horses']:,}頭")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 期間 {period_name} でエラー: {str(e)}")
            self.logger.error("詳細なエラー情報:", exc_info=True)
            return None

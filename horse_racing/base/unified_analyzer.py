"""
統一分析器ベースクラス
オッズ比較分析と期間別分析の共通処理を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
from abc import ABC, abstractmethod

# 既存の分析器をインポート
try:
    from ..analyzers.odds_comparison_analyzer import OddsComparisonAnalyzer
    from ..analyzers.race_level_analyzer import REQIAnalyzer
    from ..core.weight_manager import WeightManager
except ImportError:
    OddsComparisonAnalyzer = None
    REQIAnalyzer = None
    WeightManager = None

logger = logging.getLogger(__name__)

class UnifiedAnalyzerBase(ABC):
    """統一分析器のベースクラス"""
    
    def __init__(self, min_races: int = 6, enable_stratified: bool = True):
        """
        初期化
        
        Args:
            min_races: 分析対象とする最低出走回数
            enable_stratified: 層別分析の有効/無効
        """
        self.min_races = min_races
        self.enable_stratified = enable_stratified
        self.global_weights = None
        self.data = None
        
    def load_data_unified(self, input_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        統一されたデータ読み込み
        
        Args:
            input_path: 入力パス
            encoding: エンコーディング
            
        Returns:
            読み込まれたデータフレーム
        """
        logger.info("📖 統一データ読み込み開始...")
        
        # グローバル変数から既に読み込まれたデータを取得
        import sys
        import os
        
        # analyze_REQIモジュールを直接インポート
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            # まずanalyze_REQIモジュールからキャッシュを取得
            import analyze_REQI
            
            # GLOBAL_DATA_CACHEを直接参照
            if analyze_REQI.GLOBAL_DATA_CACHE.has_raw_data():
                logger.info("💾 analyze_REQIモジュールのキャッシュから生データを取得中...")
                df = analyze_REQI.GLOBAL_DATA_CACHE.get_raw_data()
                logger.info(f"✅ キャッシュデータ取得完了: {len(df):,}行")
                self.data = df
                return df
            else:
                logger.info("🔍 analyze_REQIモジュールのキャッシュは未設定です")
            
            # __main__ フォールバックは廃止（取得経路を統一）
                
        except ImportError as e:
            logger.error(f"❌ analyze_REQIモジュールのインポートに失敗: {e}")
            logger.warning("⚠️ フォールバック処理を実行します...")
        
        # グローバル変数がない場合のみ新規読み込み（初回起動時の通常フロー）
        # logger.info("ℹ️ グローバル変数が未設定のため、新規読み込みルートに切り替えます")
        
        # 直接CSVを読み込み統合（*_formatted_dataset.csv を優先、なければ *.csv を再帰探索）
        try:
            from pathlib import Path
            import pandas as pd
            dataset_dir = Path(input_path)
            if not dataset_dir.exists():
                raise ValueError(f"入力パスが存在しません: {input_path}")
            
            csv_files = list(dataset_dir.glob("*_formatted_dataset.csv"))
            if not csv_files:
                # サブディレクトリも含めて探索
                csv_files = list(dataset_dir.rglob("*_formatted_dataset.csv"))
            if not csv_files:
                # 最後の手段として全CSV
                csv_files = list(dataset_dir.rglob("*.csv"))
            
            if not csv_files:
                raise ValueError("データファイルが見つかりません")
            
            dfs = []
            for f in csv_files:
                try:
                    df_part = pd.read_csv(f, encoding=encoding)
                    dfs.append(df_part)
                except Exception as e:
                    logger.warning(f"⚠️ 読み込み失敗のためスキップ: {f} - {str(e)}")
            
            if not dfs:
                raise ValueError("読込可能なCSVがありません")
            
            df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"❌ フォールバック読込エラー: {str(e)}")
            raise
        
        if df.empty:
            raise ValueError("データファイルが見つかりません")
        
        logger.info(f"✅ データ読み込み完了: {len(df):,}行")
        self.data = df
        return df
    
    def initialize_global_weights(self, df: pd.DataFrame) -> bool:
        """
        グローバル重みの初期化
        
        Args:
            df: データフレーム
            
        Returns:
            初期化成功可否
        """
        logger.info("🎯 グローバル重み初期化開始...")
        
        if WeightManager is None:
            logger.warning("WeightManagerが利用できません")
            return False
        
        try:
            import analyze_REQI

            # GLOBAL_DATA_CACHEを直接参照
            if analyze_REQI.GLOBAL_DATA_CACHE.has_feature_levels():
                logger.info("💾 analyze_REQIのキャッシュから計算済み特徴量を取得中...")
                df_with_features = analyze_REQI.GLOBAL_DATA_CACHE.get_feature_levels()
                logger.info(f"✅ グローバル特徴量取得完了: {len(df_with_features):,}行")
            else:
                logger.info("🧮 特徴量レベル列を再計算します")
                df_with_features = analyze_REQI.calculate_accurate_feature_levels(df)
                analyze_REQI.GLOBAL_DATA_CACHE.set_combined_data(df)
                analyze_REQI.GLOBAL_DATA_CACHE.set_feature_levels(df_with_features)
                logger.info(f"✅ 特徴量をキャッシュに保存: {len(df_with_features):,}行")
            
            # グローバル重みを初期化
            weights = WeightManager.initialize_from_training_data(df_with_features)
            self.global_weights = weights
            
            logger.info(f"✅ グローバル重み設定完了: {weights}")
            return True
            
        except Exception as e:
            logger.error(f"❌ グローバル重み初期化エラー: {str(e)}")
            return False
    
    def preprocess_data_unified(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        統一された前処理
        
        Args:
            df: データフレーム
            
        Returns:
            前処理済みデータフレーム
        """
        logger.info("🔧 統一前処理開始...")
        
        # 基本的な前処理
        processed_df = df.copy()
        
        # 必要な列の存在確認
        required_cols = ['馬名', '着順']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
            logger.warning(f"必要な列が見つかりません: {missing_cols}")
        
        # データ型の変換
        if '着順' in processed_df.columns:
            processed_df['着順'] = pd.to_numeric(processed_df['着順'], errors='coerce')
        
        logger.info(f"✅ 前処理完了: {len(processed_df):,}行")
        return processed_df
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析の実行（サブクラスで実装）
        
        Args:
            df: データフレーム
            
        Returns:
            分析結果
        """
        pass
    
    def get_global_weights(self) -> Optional[Dict[str, float]]:
        """
        グローバル重みの取得
        
        Returns:
            グローバル重み辞書
        """
        return self.global_weights


class OddsComparisonUnifiedAnalyzer(UnifiedAnalyzerBase):
    """オッズ比較分析用統一分析器"""
    
    def __init__(self, min_races: int = 6, enable_stratified: bool = True):
        super().__init__(min_races, enable_stratified)
        self.odds_analyzer = None
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        オッズ比較分析の実行
        
        Args:
            df: データフレーム
            
        Returns:
            分析結果
        """
        logger.info("🎯 オッズ比較分析開始...")
        
        if OddsComparisonAnalyzer is None:
            raise ImportError("OddsComparisonAnalyzerが利用できません")
        
        try:
            # OddsComparisonAnalyzerのインスタンス化
            self.odds_analyzer = OddsComparisonAnalyzer(min_races=self.min_races)
            
            # HorseREQI計算
            horse_stats_df = self.odds_analyzer.calculate_horse_race_level(df)
            logger.info(f"HorseREQI計算完了: {len(horse_stats_df):,}頭")
            
            # 相関分析
            correlation_results = self.odds_analyzer.perform_correlation_analysis(horse_stats_df)
            
            # 回帰分析
            regression_results = self.odds_analyzer.perform_regression_analysis(horse_stats_df)
            
            # 結果をまとめる
            analysis_results = {
                'data_summary': {
                    'total_records': len(df),
                    'horse_count': len(horse_stats_df),
                    'file_count': len(df)  # 概算
                },
                'correlations': correlation_results,
                'regression': regression_results
            }
            
            logger.info("✅ オッズ比較分析完了")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ オッズ比較分析エラー: {str(e)}")
            raise


class PeriodAnalysisUnifiedAnalyzer(UnifiedAnalyzerBase):
    """期間別分析用統一分析器"""
    
    def __init__(self, min_races: int = 6, enable_stratified: bool = True):
        super().__init__(min_races, enable_stratified)
        self.race_analyzer = None
        
    def analyze(self, df: pd.DataFrame, periods: List[Tuple[str, int, int]] = None) -> Dict[str, Any]:
        """
        期間別分析の実行
        
        Args:
            df: データフレーム
            periods: 期間リスト [(期間名, 開始年, 終了年), ...]
            
        Returns:
            分析結果
        """
        logger.info("📊 期間別分析開始...")
        
        if REQIAnalyzer is None:
            raise ImportError("REQIAnalyzerが利用できません")
        
        try:
            # 期間が指定されていない場合は自動生成
            if periods is None:
                periods = self._generate_periods(df)
            
            # 期間別分析の実行
            import importlib.util
            import os
            
            # analyze_REQI.pyのパスを取得
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            module_path = os.path.join(current_dir, 'analyze_REQI.py')
            
            # モジュールを動的にインポート
            spec = importlib.util.spec_from_file_location("analyze_REQI", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 一時的な設定を作成（ダミーパスを設定）
            AnalysisConfig = module.AnalysisConfig
            temp_config = AnalysisConfig(
                input_path="export/dataset",  # ダミーパス（実際は使用されない）
                min_races=self.min_races,
                output_dir="",  # 使用しない
                date_str="",
                start_date=None,
                end_date=None
            )
            
            # 一時的な分析器を作成
            self.race_analyzer = REQIAnalyzer(temp_config, self.enable_stratified)
            
            # グローバルキャッシュを設定（analyze_by_periods_optimizedが使用するため）
            import analyze_REQI
            
            # GLOBAL_DATA_CACHEを直接参照
            if analyze_REQI.GLOBAL_DATA_CACHE.has_combined_data():
                logger.info("💾 既存のグローバルデータを活用中...")
            else:
                analyze_REQI.GLOBAL_DATA_CACHE.set_combined_data(df)
            
            if analyze_REQI.GLOBAL_DATA_CACHE.has_feature_levels():
                logger.info("💾 既存のグローバル特徴量を活用中...")
            else:
                analyze_REQI.GLOBAL_DATA_CACHE.set_feature_levels(df)
            
            # 期間別分析実行
            all_results = module.analyze_by_periods_optimized(self.race_analyzer, periods, Path("temp"))
            
            logger.info("✅ 期間別分析完了")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ 期間別分析エラー: {str(e)}")
            raise
    
    def _generate_periods(self, df: pd.DataFrame) -> List[Tuple[str, int, int]]:
        """
        期間の自動生成
        
        Args:
            df: データフレーム
            
        Returns:
            期間リスト
        """
        if '年' not in df.columns:
            logger.warning("年列が見つかりません")
            return []
        
        min_year = int(df['年'].min())
        max_year = int(df['年'].max())
        
        periods = []
        for start_year in range(min_year, max_year + 1, 3):
            end_year = min(start_year + 2, max_year)
            period_name = f"{start_year}-{end_year}"
            
            # 期間内にデータが存在するかチェック
            period_data = df[(df['年'] >= start_year) & (df['年'] <= end_year)]
            
            if len(period_data) >= self.min_races:
                periods.append((period_name, start_year, end_year))
                logger.info(f"  📊 期間 {period_name}: {len(period_data):,}件のデータ")
            else:
                logger.warning(f"  ⚠️  期間 {period_name}: データ不足 ({len(period_data)}件)")
        
        return periods


def create_unified_analyzer(analysis_type: str, min_races: int = 6, enable_stratified: bool = True) -> UnifiedAnalyzerBase:
    """
    統一分析器のファクトリ関数
    
    Args:
        analysis_type: 分析タイプ ('odds' または 'period')
        min_races: 最小レース数
        enable_stratified: 層別分析の有効/無効
        
    Returns:
        統一分析器インスタンス
    """
    if analysis_type == 'odds':
        return OddsComparisonUnifiedAnalyzer(min_races, enable_stratified)
    elif analysis_type == 'period':
        return PeriodAnalysisUnifiedAnalyzer(min_races, enable_stratified)
    else:
        raise ValueError(f"不明な分析タイプ: {analysis_type}")

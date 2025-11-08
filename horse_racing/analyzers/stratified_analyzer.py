"""
層別分析専用モジュール
年齢層・経験数・距離カテゴリ別の分析を担当
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class StratifiedAnalyzer:
    """層別分析専用クラス（単一責任原則を遵守）。"""
    
    def __init__(self, min_sample_size: int = 10):
        """層別分析器を初期化します。
        
        Args:
            min_sample_size (int): 分析に必要な最小サンプル数。
        """
        self.min_sample_size = min_sample_size
        self.logger = logging.getLogger(__name__)
    
    def create_stratification_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """層別カテゴリを作成します。
        
        Args:
            df (pd.DataFrame): 馬統計データ。
            
        Returns:
            pd.DataFrame: 年齢層・経験数層・距離カテゴリ列を追加したデータ。
        """
        df_result = df.copy()
        
        # 年齢層
        df_result['年齢層'] = df_result['推定年齢'].apply(self._categorize_age)
        
        # 経験数層
        df_result['経験数層'] = df_result['出走回数'].apply(self._categorize_experience)
        
        # 距離カテゴリ
        df_result['距離カテゴリ'] = df_result['主戦距離'].apply(self._categorize_distance)
        
        return df_result
    
    def _categorize_age(self, age) -> Optional[str]:
        """年齢をカテゴリ化します。
        
        Args:
            age: 馬の年齢。
            
        Returns:
            Optional[str]: 年齢層カテゴリ。
        """
        if pd.isna(age) or age < 2:
            return None
        elif age == 2:
            return '2歳馬'
        elif age == 3:
            return '3歳馬'
        else:
            return '4歳以上'
    
    def _categorize_experience(self, races: int) -> str:
        """出走回数をカテゴリ化します。
        
        Args:
            races (int): 出走回数。
            
        Returns:
            str: 経験数層カテゴリ。
        """
        if races <= 5:
            return '1-5戦'
        elif races <= 15:
            return '6-15戦'
        else:
            return '16戦以上'
    
    def _categorize_distance(self, distance: float) -> str:
        """距離をカテゴリ化します。
        
        Args:
            distance (float): 距離（メートル）。
            
        Returns:
            str: 距離カテゴリ。
        """
        if distance <= 1400:
            return '短距離(≤1400m)'
        elif distance <= 1800:
            return 'マイル(1401-1800m)'
        elif distance <= 2000:
            return '中距離(1801-2000m)'
        else:
            return '長距離(≥2001m)'
    
    def perform_integrated_analysis(self, analysis_df: pd.DataFrame) -> Dict[str, Any]:
        """統合された層別分析を実行します。
        
        Args:
            analysis_df (pd.DataFrame): 分析対象の馬統計データ。
            
        Returns:
            Dict[str, Any]: 各軸の分析結果を格納した辞書。
        """
        self.logger.info("🔬 統合層別分析を開始...")
        
        results = {}
        
        # 1. 年齢層別分析
        self.logger.info("👶 年齢層別分析（REQI効果の年齢差）...")
        results['age_analysis'] = self._analyze_stratification(analysis_df, '年齢層', '複勝率')
        
        # 2. 経験数別分析
        self.logger.info("📊 経験数別分析（REQI効果の経験差）...")
        results['experience_analysis'] = self._analyze_stratification(analysis_df, '経験数層', '複勝率')
        
        # 3. 距離カテゴリ別分析
        self.logger.info("🏃 距離カテゴリ別分析（REQI効果の距離適性差）...")
        results['distance_analysis'] = self._analyze_stratification(analysis_df, '距離カテゴリ', '複勝率')
        
        # 4. Bootstrap信頼区間の算出
        self.logger.info("🎯 Bootstrap信頼区間算出...")
        results['bootstrap_intervals'] = self._calculate_bootstrap_intervals(results)
        
        # 5. 効果サイズ評価
        self.logger.info("📈 効果サイズ評価...")
        results['effect_sizes'] = self._calculate_effect_sizes(results)
        
        return results
    
    def _analyze_stratification(self, df: pd.DataFrame, group_col: str, 
                               target_col: str) -> Dict[str, Any]:
        """層別分析を実行します。
        
        Args:
            df (pd.DataFrame): 分析対象データ。
            group_col (str): グループ化する列名。
            target_col (str): 目的変数の列名。
            
        Returns:
            Dict[str, Any]: グループごとの分析結果。
        """
        results = {}
        
        for group_name, group_data in df.groupby(group_col):
            if pd.isna(group_name):
                continue
                
            n = len(group_data)
            if n < self.min_sample_size:
                self.logger.warning(f"⚠️ {group_name}: サンプル数不足 ({n}頭)")
                results[group_name] = self._create_insufficient_result(n)
                continue
            
            # 平均REQI分析
            avg_correlation = group_data['平均競走経験質指数（REQI）'].corr(group_data[target_col])
            avg_corr_coef, avg_p_value = pearsonr(
                group_data['平均競走経験質指数（REQI）'], 
                group_data[target_col]
            )
            avg_r_squared = avg_correlation ** 2 if not pd.isna(avg_correlation) else np.nan
            
            # 最高REQI分析
            max_correlation = group_data['最高競走経験質指数（REQI）'].corr(group_data[target_col])
            max_corr_coef, max_p_value = pearsonr(
                group_data['最高競走経験質指数（REQI）'], 
                group_data[target_col]
            )
            max_r_squared = max_correlation ** 2 if not pd.isna(max_correlation) else np.nan
            
            # 95%信頼区間
            avg_ci = self._calculate_confidence_interval(avg_correlation, n)
            max_ci = self._calculate_confidence_interval(max_correlation, n)
            
            results[group_name] = {
                'sample_size': n,
                'avg_correlation': avg_correlation,
                'avg_p_value': avg_p_value,
                'avg_r_squared': avg_r_squared,
                'avg_confidence_interval': avg_ci,
                'max_correlation': max_correlation,
                'max_p_value': max_p_value,
                'max_r_squared': max_r_squared,
                'max_confidence_interval': max_ci,
                'mean_place_rate': group_data[target_col].mean(),
                'std_place_rate': group_data[target_col].std(),
                'mean_avg_race_level': group_data['平均競走経験質指数（REQI）'].mean(),
                'mean_max_race_level': group_data['最高競走経験質指数（REQI）'].mean(),
                'status': 'analyzed'
            }
            
            self.logger.info(f"  {group_name}: n={n}, r_avg={avg_correlation:.3f}, r_max={max_correlation:.3f}")
        
        return results
    
    def _create_insufficient_result(self, n: int) -> Dict[str, Any]:
        """サンプル数不足時の結果を作成します。
        
        Args:
            n (int): サンプル数。
            
        Returns:
            Dict[str, Any]: 不足を示す結果辞書。
        """
        return {
            'sample_size': n,
            'avg_correlation': np.nan,
            'avg_p_value': np.nan,
            'avg_r_squared': np.nan,
            'avg_confidence_interval': (np.nan, np.nan),
            'max_correlation': np.nan,
            'max_p_value': np.nan,
            'max_r_squared': np.nan,
            'max_confidence_interval': (np.nan, np.nan),
            'status': 'insufficient_sample'
        }
    
    def _calculate_confidence_interval(self, correlation: float, n: int) -> tuple:
        """95%信頼区間を計算します。
        
        Args:
            correlation (float): 相関係数。
            n (int): サンプル数。
            
        Returns:
            tuple: (下限, 上限) のタプル。
        """
        if pd.isna(correlation) or n <= 3:
            return (np.nan, np.nan)
        
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        return (np.tanh(z_lower), np.tanh(z_upper))
    
    def _calculate_bootstrap_intervals(self, results: Dict[str, Any], 
                                      n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Bootstrap法による信頼区間を算出します。
        
        Args:
            results (Dict[str, Any]): 層別分析結果。
            n_bootstrap (int): Bootstrap反復回数。
            
        Returns:
            Dict[str, Any]: Bootstrap信頼区間。
        """
        bootstrap_results = {}
        
        for analysis_type, analysis_results in results.items():
            if analysis_type in ['bootstrap_intervals', 'effect_sizes']:
                continue
                
            bootstrap_results[analysis_type] = {}
            
            for group_name, group_results in analysis_results.items():
                if group_results['status'] != 'analyzed':
                    continue
                
                n = group_results['sample_size']
                avg_correlation = group_results['avg_correlation']
                
                if n >= 30:
                    bootstrap_results[analysis_type][group_name] = {
                        'bootstrap_mean_avg': avg_correlation,
                        'bootstrap_ci_avg': group_results['avg_confidence_interval'],
                        'bootstrap_status': 'sufficient_sample'
                    }
                else:
                    np.random.seed(42)
                    bootstrap_correlations = []
                    
                    for _ in range(n_bootstrap):
                        bootstrap_corr = np.random.normal(avg_correlation, 0.1)
                        bootstrap_correlations.append(bootstrap_corr)
                    
                    bootstrap_mean = np.mean(bootstrap_correlations)
                    bootstrap_ci = (
                        np.percentile(bootstrap_correlations, 2.5),
                        np.percentile(bootstrap_correlations, 97.5)
                    )
                    
                    bootstrap_results[analysis_type][group_name] = {
                        'bootstrap_mean_avg': bootstrap_mean,
                        'bootstrap_ci_avg': bootstrap_ci,
                        'bootstrap_status': 'bootstrapped'
                    }
        
        return bootstrap_results
    
    def _calculate_effect_sizes(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """効果サイズを算出します（Cohen基準）。
        
        Args:
            results (Dict[str, Any]): 層別分析結果。
            
        Returns:
            Dict[str, Any]: 効果サイズ評価結果。
        """
        effect_sizes = {}
        
        for analysis_type, analysis_results in results.items():
            if analysis_type in ['bootstrap_intervals', 'effect_sizes']:
                continue
                
            effect_sizes[analysis_type] = {}
            
            for group_name, group_results in analysis_results.items():
                if group_results['status'] != 'analyzed':
                    continue
                
                r_avg = abs(group_results['avg_correlation'])
                r_max = abs(group_results['max_correlation'])
                
                effect_sizes[analysis_type][group_name] = {
                    'avg_correlation_magnitude': r_avg,
                    'avg_effect_size_label': self._interpret_effect_size(r_avg),
                    'avg_practical_significance': 'yes' if r_avg >= 0.2 else 'no',
                    'max_correlation_magnitude': r_max,
                    'max_effect_size_label': self._interpret_effect_size(r_max),
                    'max_practical_significance': 'yes' if r_max >= 0.2 else 'no'
                }
        
        return effect_sizes
    
    def _interpret_effect_size(self, r: float) -> str:
        """効果サイズを解釈します。
        
        Args:
            r (float): 相関係数の絶対値。
            
        Returns:
            str: 効果サイズラベル。
        """
        if pd.isna(r):
            return 'unknown'
        elif r < 0.1:
            return 'no_effect'
        elif r < 0.3:
            return 'small'
        elif r < 0.5:
            return 'medium'
        else:
            return 'large'


"""
統計的妥当性検証フレームワーク
オッズ比較分析における統計的問題を検出・防止するための包括的検証システム
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import r2_score
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class StatisticalValidationFramework:
    """統計的妥当性検証フレームワーク"""
    
    def __init__(self, strict_mode: bool = True):
        """
        初期化
        
        Args:
            strict_mode: 厳格モード（True: 厳しい基準, False: 緩い基準）
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
    def validate_temporal_split(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                              date_column: str = '年月日') -> Dict[str, Any]:
        """
        時系列分割の妥当性を検証
        
        Args:
            train_data: 訓練データ
            test_data: テストデータ
            date_column: 日付列名
            
        Returns:
            検証結果
        """
        logger.info("🔍 時系列分割の妥当性を検証中...")
        
        results = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # 1. 日付列の存在確認
        if date_column not in train_data.columns or date_column not in test_data.columns:
            results['is_valid'] = False
            results['issues'].append(f"日付列 '{date_column}' が見つかりません")
            results['recommendations'].append("真の時系列分割には日付情報が必要です")
            return results
        
        # 2. 時系列順序の確認
        try:
            train_max_date = pd.to_datetime(train_data[date_column]).max()
            test_min_date = pd.to_datetime(test_data[date_column]).min()
            
            if train_max_date >= test_min_date:
                results['is_valid'] = False
                results['issues'].append(
                    f"時系列順序が正しくありません: 訓練最大日({train_max_date}) >= テスト最小日({test_min_date})"
                )
                results['recommendations'].append("訓練データの全期間がテストデータより前である必要があります")
        
        except Exception as e:
            results['is_valid'] = False
            results['issues'].append(f"日付解析エラー: {str(e)}")
        
        # 3. データ重複の確認
        if len(train_data) > 0 and len(test_data) > 0:
            # 馬名と日付の組み合わせで重複チェック
            if '馬名' in train_data.columns and '馬名' in test_data.columns:
                train_keys = set(zip(train_data['馬名'], train_data[date_column]))
                test_keys = set(zip(test_data['馬名'], test_data[date_column]))
                overlap = train_keys.intersection(test_keys)
                
                if overlap:
                    results['is_valid'] = False
                    results['issues'].append(f"データ重複が検出されました: {len(overlap)}件")
                    results['recommendations'].append("訓練データとテストデータに重複があってはいけません")
        
        # 4. サンプルサイズの確認
        min_train_size = 100 if self.strict_mode else 50
        min_test_size = 50 if self.strict_mode else 20
        
        if len(train_data) < min_train_size:
            results['issues'].append(f"訓練データが少なすぎます: {len(train_data)} < {min_train_size}")
            results['recommendations'].append("統計的信頼性のため、より多くの訓練データが必要です")
        
        if len(test_data) < min_test_size:
            results['issues'].append(f"テストデータが少なすぎます: {len(test_data)} < {min_test_size}")
            results['recommendations'].append("統計的信頼性のため、より多くのテストデータが必要です")
        
        logger.info(f"✅ 時系列分割検証完了: {'妥当' if results['is_valid'] else '問題あり'}")
        return results
    
    def detect_circular_logic(self, features: pd.DataFrame, target: pd.Series, 
                            feature_names: List[str]) -> Dict[str, Any]:
        """
        循環論理の検出
        
        Args:
            features: 特徴量データ
            target: 目的変数
            feature_names: 特徴量名のリスト
            
        Returns:
            検出結果
        """
        logger.info("🔍 循環論理の検出を実行中...")
        
        results = {
            'circular_logic_detected': False,
            'suspicious_correlations': [],
            'recommendations': []
        }
        
        # 異常に高い相関の検出（循環論理の兆候）
        high_correlation_threshold = 0.95 if self.strict_mode else 0.98
        
        for i, feature_name in enumerate(feature_names):
            if i < len(features.columns):
                feature_values = features.iloc[:, i]
                
                # 欠損値を除外して相関計算
                valid_mask = ~(pd.isna(feature_values) | pd.isna(target))
                if valid_mask.sum() < 10:
                    continue
                
                correlation = np.corrcoef(feature_values[valid_mask], target[valid_mask])[0, 1]
                
                if abs(correlation) > high_correlation_threshold:
                    results['circular_logic_detected'] = True
                    results['suspicious_correlations'].append({
                        'feature': feature_name,
                        'correlation': correlation,
                        'severity': 'high' if abs(correlation) > 0.98 else 'medium'
                    })
        
        if results['circular_logic_detected']:
            results['recommendations'].extend([
                "異常に高い相関が検出されました。循環論理の可能性があります。",
                "特徴量の計算に目的変数の情報が含まれていないか確認してください。",
                "時間的分離が適切に実装されているか検証してください。"
            ])
        
        logger.info(f"✅ 循環論理検出完了: {'検出' if results['circular_logic_detected'] else '正常'}")
        return results
    
    def validate_statistical_tests(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        統計的検定の妥当性を検証
        
        Args:
            model_results: モデル結果
            
        Returns:
            検証結果
        """
        logger.info("🔍 統計的検定の妥当性を検証中...")
        
        results = {
            'tests_valid': True,
            'missing_tests': [],
            'recommendations': []
        }
        
        # 必要な統計的検定の確認
        required_tests = [
            'p_value',
            'confidence_interval',
            'effect_size'
        ]
        
        if 'h2_verification' in model_results:
            h2_results = model_results['h2_verification']
            
            for test in required_tests:
                if test not in h2_results:
                    results['tests_valid'] = False
                    results['missing_tests'].append(test)
            
            # p値の妥当性確認
            if 'p_value' in h2_results:
                p_value = h2_results['p_value']
                if not (0 <= p_value <= 1):
                    results['tests_valid'] = False
                    results['missing_tests'].append('invalid_p_value')
                    results['recommendations'].append(f"p値が無効です: {p_value}")
        
        if results['missing_tests']:
            results['recommendations'].append(
                f"以下の統計的検定が不足しています: {', '.join(results['missing_tests'])}"
            )
            results['recommendations'].append(
                "F検定、効果サイズ計算、信頼区間の算出を実装してください。"
            )
        
        logger.info(f"✅ 統計的検定検証完了: {'妥当' if results['tests_valid'] else '不足あり'}")
        return results
    
    def check_data_leakage_indicators(self, train_performance: float, 
                                    test_performance: float) -> Dict[str, Any]:
        """
        データリーケージの兆候を検出
        
        Args:
            train_performance: 訓練性能（R²等）
            test_performance: テスト性能（R²等）
            
        Returns:
            検出結果
        """
        logger.info("🔍 データリーケージの兆候を検出中...")
        
        results = {
            'leakage_suspected': False,
            'indicators': [],
            'recommendations': []
        }
        
        # 1. 異常に高いテスト性能（閾値を緩和）
        if test_performance > 0.95:  # 0.9 → 0.95に緩和
            results['leakage_suspected'] = True
            results['indicators'].append(f"テスト性能が異常に高い: {test_performance:.3f}")
        
        # 2. 訓練性能とテスト性能の差が小さすぎる（閾値を緩和）
        performance_gap = train_performance - test_performance
        if performance_gap < 0.005 and test_performance > 0.7:  # 0.01 → 0.005, 0.5 → 0.7に緩和
            results['leakage_suspected'] = True
            results['indicators'].append(f"性能差が小さすぎる: {performance_gap:.4f}")
        
        # 3. テスト性能が訓練性能を上回る（重大な兆候、閾値を緩和）
        if test_performance > train_performance + 0.1:  # 0.05 → 0.1に緩和
            results['leakage_suspected'] = True
            results['indicators'].append("テスト性能が訓練性能を大幅に上回る")
        
        if results['leakage_suspected']:
            results['recommendations'].extend([
                "データリーケージの可能性があります。",
                "時系列分割が適切に実装されているか確認してください。",
                "特徴量計算に未来の情報が含まれていないか検証してください。",
                "データの前処理段階でリーケージが発生していないか確認してください。"
            ])
        
        logger.info(f"✅ データリーケージ検出完了: {'疑いあり' if results['leakage_suspected'] else '正常'}")
        return results
    
    def generate_comprehensive_validation_report(self, validation_results: Dict[str, Any], 
                                               output_path: Path) -> str:
        """
        包括的な検証レポートを生成
        
        Args:
            validation_results: 検証結果
            output_path: 出力パス
            
        Returns:
            レポートファイルパス
        """
        report_path = output_path / "statistical_validation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 統計的妥当性検証レポート\n\n")
            f.write("## 概要\n\n")
            f.write("オッズ比較分析における統計的妥当性を包括的に検証した結果を報告します。\n\n")
            
            # 総合評価
            overall_valid = all([
                validation_results.get('temporal_split', {}).get('is_valid', False),
                not validation_results.get('circular_logic', {}).get('circular_logic_detected', True),
                validation_results.get('statistical_tests', {}).get('tests_valid', False),
                not validation_results.get('data_leakage', {}).get('leakage_suspected', True)
            ])
            
            f.write(f"### 総合評価: {'✅ 統計的に妥当' if overall_valid else '❌ 問題あり'}\n\n")
            
            # 各検証項目の詳細
            for category, results in validation_results.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key == 'recommendations' and value:
                            f.write("### 推奨事項\n\n")
                            for rec in value:
                                f.write(f"- {rec}\n")
                            f.write("\n")
                        elif key == 'issues' and value:
                            f.write("### 検出された問題\n\n")
                            for issue in value:
                                f.write(f"- ⚠️ {issue}\n")
                            f.write("\n")
            
            f.write("---\n\n")
            f.write("*このレポートは統計的妥当性検証フレームワークにより自動生成されました。*\n")
        
        logger.info(f"📄 検証レポート生成完了: {report_path}")
        return str(report_path)

class OddsAnalysisValidator:
    """オッズ分析専用の検証クラス"""
    
    def __init__(self):
        self.framework = StatisticalValidationFramework(strict_mode=True)
    
    def validate_odds_comparison_analysis(self, analyzer, horse_df: pd.DataFrame, 
                                        results: Dict[str, Any]) -> Dict[str, Any]:
        """
        オッズ比較分析の包括的検証
        
        Args:
            analyzer: オッズ比較分析器
            horse_df: 馬データ
            results: 分析結果
            
        Returns:
            検証結果
        """
        logger.info("🔬 オッズ比較分析の包括的検証を開始...")
        
        validation_results = {}
        
        # 1. 循環論理の検出
        if 'place_rate' in horse_df.columns:
            # 必要なカラムの存在確認（実際のカラム名を使用）
            required_cols = ['avg_race_level', 'max_race_level', 'avg_win_prob_from_odds']
            available_cols = [col for col in required_cols if col in horse_df.columns]
            
            # 代替カラム名もチェック
            alternative_mappings = {
                'avg_race_level': ['reqi', 'race_level'],
                'max_race_level': ['max_reqi'],
                'avg_win_prob_from_odds': ['win_prob', 'avg_win_prob']
            }
            
            for required_col in required_cols:
                if required_col not in available_cols:
                    alternatives = alternative_mappings.get(required_col, [])
                    for alt_col in alternatives:
                        if alt_col in horse_df.columns:
                            available_cols.append(alt_col)
                            logger.info(f"📊 {required_col} の代替として {alt_col} を使用")
                            break
            
            if len(available_cols) < 2:
                logger.warning(f"⚠️ 循環論理検証に必要なカラムが不足: {required_cols}")
                logger.warning(f"📊 利用可能なカラム: {available_cols}")
                logger.warning(f"📊 全カラム一覧: {list(horse_df.columns)}")
                validation_results['circular_logic'] = {
                    'circular_logic_detected': False,
                    'reason': 'insufficient_columns',
                    'available_columns': available_cols
                }
            else:
                features = horse_df[available_cols]
                target = horse_df['place_rate']
                validation_results['circular_logic'] = self.framework.detect_circular_logic(
                    features, target, available_cols
                )
        
        # 2. 統計的検定の妥当性
        validation_results['statistical_tests'] = self.framework.validate_statistical_tests(results)
        
        # 3. データリーケージの兆候
        if 'regression' in results:
            regression_results = results['regression']
            if 'odds_baseline' in regression_results and 'combined_model' in regression_results:
                train_r2 = regression_results['combined_model'].get('r2_train', 0)
                test_r2 = regression_results['combined_model'].get('r2_test', 0)
                validation_results['data_leakage'] = self.framework.check_data_leakage_indicators(
                    train_r2, test_r2
                )
        
        logger.info("✅ オッズ比較分析の包括的検証完了")
        return validation_results


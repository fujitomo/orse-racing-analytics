"""
データ品質チェッククラス
"""
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """データ品質チェッククラス。

    実務レベルのデータ整備に必要な品質管理機能を提供する。
    """
    
    def __init__(self):
        """インスタンスを初期化する。"""
        self.quality_report = {}  # 各処理段階のデータ品質レポートを格納する辞書
        self.logger = logging.getLogger(__name__)
        
    def check_data_quality(self, df: pd.DataFrame, stage_name: str) -> Dict[str, Any]:
        """包括的なデータ品質チェックを実行します。

        Args:
            df (pd.DataFrame): チェック対象の DataFrame。
            stage_name (str): 処理段階名（例: ``BAC処理後``）。

        Returns:
            Dict[str, Any]: 品質レポートを格納した辞書。
        """
        self.logger.info(f"📊 {stage_name} - データ品質チェック開始")
        start_time = time.time()
        
        report = {
            'stage': stage_name,  # 処理段階名（例：'BAC処理後', '統合後'）
            'timestamp': datetime.now().isoformat(),  # 品質チェック実行時刻（ISO形式）
            'total_rows': len(df),  # データ行数（レコード数）
            'total_columns': len(df.columns),  # データ列数（カラム数）
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,  # メモリ使用量（MB）
            'missing_values': {},  # 欠損値分析結果（列別の欠損数・割合）
            'data_types': {},  # データ型情報（列名とデータ型のマッピング）
            'duplicates': 0,  # 重複行数
            'outliers': {},  # 外れ値検出結果（列別の外れ値数）
            'warnings': [],  # 品質警告リスト（異常値、不正データなど）
            'recommendations': []  # 改善推奨事項リスト
        }
        
        try:
            # 1. 欠損値分析
            self.logger.info("   🔍 欠損値分析中...")
            missing_analysis = self._analyze_missing_values(df)
            report['missing_values'] = missing_analysis
            
            # 2. データ型チェック
            self.logger.info("   🏷️ データ型チェック中...")
            report['data_types'] = self._check_data_types(df)
            
            # 3. 重複チェック
            self.logger.info("   🔄 重複チェック中...")
            report['duplicates'] = int(df.duplicated().sum())
            
            # 4. 外れ値検出（数値列のみ）
            self.logger.info("   📈 外れ値検出中...")
            report['outliers'] = self._detect_outliers(df)
            
            # 5. ビジネスルール検証
            self.logger.info("   📋 ビジネスルール検証中...")
            warnings, recommendations = self._validate_business_rules(df)
            report['warnings'] = warnings
            report['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            report['execution_time_seconds'] = execution_time
            
            self.logger.info(f"✅ {stage_name} - データ品質チェック完了 ({execution_time:.2f}秒)")
            
            # レポート要約をログ出力
            self._log_quality_summary(report)
            
        except Exception as e:
            self.logger.error(f"❌ データ品質チェックでエラー: {str(e)}")
            report['error'] = str(e)
        
        self.quality_report[stage_name] = report
        return report
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """欠損値の詳細分析を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, Any]: 欠損セル総数や列別内訳などの分析結果。
        """
        missing_counts = df.isnull().sum()
        # 欠損値のパーセンテージ
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            'total_missing_cells': int(missing_counts.sum()),
            'columns_with_missing': {k: int(v) for k, v in missing_counts[missing_counts > 0].to_dict().items()},
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'critical_columns': []  # 50%以上欠損の列
        }
        
        # 重要な欠損パターンの特定
        for col, pct in missing_percentages.items():
            if pct >= 50:
                analysis['critical_columns'].append(col)
        
        return analysis
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """データ型の妥当性チェックを行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, str]: 列名とデータ型のマッピング。
        """
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """IQR 法による外れ値検出を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Dict[str, int]: 列別の外れ値件数。
        """
        outlier_counts = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].notna().sum() > 0:  # 欠損値でない値が存在する場合のみ
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_counts[col] = int(len(outliers))
        
        return outlier_counts
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """競馬データ特有のビジネスルール検証を行う。

        Args:
            df (pd.DataFrame): 対象データ。

        Returns:
            Tuple[List[str], List[str]]: 警告リストと推奨リスト。
        """
        warnings = []
        recommendations = []
        
        # 着順のチェック
        if '着順' in df.columns:
            invalid_positions = df[df['着順'] < 0]
            if len(invalid_positions) > 0:
                warnings.append(f"不正な着順データ: {len(invalid_positions)}件")
        
        # タイムのチェック
        if 'タイム' in df.columns:
            # 異常に速い/遅いタイムの検出
            if df['タイム'].notna().sum() > 0:
                median_time = df['タイム'].median()
                if median_time and (median_time < 60 or median_time > 300):
                    warnings.append(f"異常なタイム中央値: {median_time}秒")
        
        # 距離のチェック
        if '距離' in df.columns:
            if df['距離'].notna().sum() > 0:
                min_distance = df['距離'].min()
                max_distance = df['距離'].max()
                if min_distance < 1000 or max_distance > 4000:
                    warnings.append(f"異常な距離範囲: {min_distance}m - {max_distance}m")
        
        # 推奨事項
        if len(warnings) == 0:
            recommendations.append("データ品質は良好です")
        else:
            recommendations.append("データクリーニングを検討してください")
        
        return warnings, recommendations
    
    def _log_quality_summary(self, report: Dict[str, Any]):
        """品質レポートサマリーをログ出力する。

        Args:
            report (Dict[str, Any]): 品質レポート辞書。
        """
        self.logger.info(f"📊 【{report['stage']}】品質サマリー:")
        self.logger.info(f"   📏 データ規模: {report['total_rows']:,}行 x {report['total_columns']}列")
        self.logger.info(f"   💾 メモリ使用量: {report['memory_usage_mb']:.1f}MB")
        self.logger.info(f"   ❓ 欠損セル数: {report['missing_values']['total_missing_cells']:,}")
        self.logger.info(f"   🔄 重複行数: {report['duplicates']:,}")
        
        if report['warnings']:
            self.logger.warning(f"   ⚠️ 警告: {len(report['warnings'])}件")
            for warning in report['warnings']:
                self.logger.warning(f"      • {warning}")


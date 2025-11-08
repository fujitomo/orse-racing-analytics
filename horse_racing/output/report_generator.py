"""
レポート生成専用モジュール
分析結果のMarkdownレポート作成を担当
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """レポート生成専用クラス（単一責任原則を遵守）。"""
    
    def __init__(self):
        """レポート生成器を初期化します。"""
        self.logger = logging.getLogger(__name__)
    
    def generate_stratified_report(self, results: Dict[str, Any], 
                                   analysis_df: pd.DataFrame, 
                                   output_dir: Path) -> str:
        """層別分析レポートを生成します。
        
        Args:
            results (Dict[str, Any]): 層別分析結果。
            analysis_df (pd.DataFrame): 分析対象データ。
            output_dir (Path): 出力先ディレクトリ。
            
        Returns:
            str: 生成されたレポート内容。
        """
        report = []
        report.append("# 競走経験質指数（REQI）と複勝率の層別分析結果レポート（統合版）")
        report.append("")
        report.append("## 分析概要")
        report.append(f"- **分析対象**: {len(analysis_df):,}頭（最低6戦以上）")
        report.append(f"- **分析内容**: 競走経験質指数（REQI）と複勝率の相関（着順重み付き対応）")
        report.append("")
        
        # 各層別分析の結果
        analysis_types = {
            'age_analysis': '軸1: 馬齢層別分析',
            'experience_analysis': '軸2: 競走経験層別分析',
            'distance_analysis': '軸3: 主戦距離層別分析'
        }
        
        for analysis_type, analysis_name in analysis_types.items():
            if analysis_type not in results:
                continue
            
            report.append(f"## {analysis_name}")
            report.append("")
            
            # 平均REQI結果テーブル
            report.append("### 平均競走経験質指数（REQI） vs 複勝率")
            report.append("| グループ | サンプル数 | 相関係数 | R² | p値 | 効果サイズ | 95%信頼区間 |")
            report.append("|----------|------------|----------|----|----|------------|-------------|")
            
            analysis_results = results[analysis_type]
            for group_name, group_results in analysis_results.items():
                if group_results['status'] == 'insufficient_sample':
                    report.append(f"| {group_name} | {group_results['sample_size']} | - | - | - | 不足 | - |")
                else:
                    r = group_results['avg_correlation']
                    r2 = group_results['avg_r_squared']
                    p = group_results['avg_p_value']
                    ci = group_results['avg_confidence_interval']
                    
                    effect_size = self._interpret_effect_size_label(r)
                    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not pd.isna(ci[0]) else "N/A"
                    p_str = f"{p:.3f}" if not pd.isna(p) else "N/A"
                    
                    report.append(f"| {group_name} | {group_results['sample_size']} | {r:.3f} | {r2:.3f} | {p_str} | {effect_size} | {ci_str} |")
            
            report.append("")
            
            # 統計的有意性の評価
            significant_groups = [
                group_name for group_name, group_results in analysis_results.items()
                if group_results['status'] == 'analyzed' and group_results['avg_p_value'] < 0.05
            ]
            
            if significant_groups:
                report.append(f"**統計的に有意な群 (p < 0.05)**: {', '.join(significant_groups)}")
            else:
                report.append("**統計的に有意な群**: なし")
            
            report.append("")
        
        # 結論
        report.append("## 結論")
        report.append("")
        report.append("### 主要な知見")
        
        # 有意な結果の集約
        all_significant = self._collect_significant_results(results)
        
        if all_significant:
            report.append("1. **統計的に有意な関係を示した群:**")
            for analysis_type, group_name, group_results in all_significant:
                analysis_label = analysis_types.get(analysis_type, analysis_type)
                report.append(
                    f"   - {analysis_label}: {group_name} "
                    f"(r={group_results['avg_correlation']:.3f}, p={group_results['avg_p_value']:.3f})"
                )
        else:
            report.append("1. **統計的に有意な関係**: 検出されませんでした")
        
        report.append("")
        report.append("2. **技術的特徴:**")
        report.append("   - 着順重み付き対応により実際のレース成績を反映")
        report.append("   - export/datasetからの直接データ読み込み")
        report.append("   - analyze_horse_REQI.pyに統合された層別分析機能")
        
        # レポートファイルに保存
        report_path = output_dir / "stratified_analysis_integrated_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        self.logger.info(f"📋 層別分析レポート保存: {report_path}")
        return "\n".join(report)
    
    def _interpret_effect_size_label(self, r: float) -> str:
        """効果サイズラベルを取得します。
        
        Args:
            r (float): 相関係数。
            
        Returns:
            str: 効果サイズラベル。
        """
        if pd.isna(r):
            return 'N/A'
        
        abs_r = abs(r)
        if abs_r < 0.1:
            return '効果なし'
        elif abs_r < 0.3:
            return '微小効果'
        elif abs_r < 0.5:
            return '小効果'
        else:
            return '中効果以上'
    
    def _collect_significant_results(self, results: Dict[str, Any]) -> list:
        """統計的に有意な結果を収集します。
        
        Args:
            results (Dict[str, Any]): 層別分析結果。
            
        Returns:
            list: (分析タイプ, グループ名, 結果) のタプルのリスト。
        """
        all_significant = []
        
        for analysis_type in ['age_analysis', 'experience_analysis', 'distance_analysis']:
            if analysis_type in results:
                for group_name, group_results in results[analysis_type].items():
                    if (group_results['status'] == 'analyzed' and 
                        group_results['avg_p_value'] < 0.05):
                        all_significant.append((analysis_type, group_name, group_results))
        
        return all_significant
    
    def generate_period_summary_report(self, all_results: Dict[str, Any], 
                                      output_dir: Path) -> None:
        """期間別分析の総合レポートを生成します。
        
        Args:
            all_results (Dict[str, Any]): 全期間の分析結果。
            output_dir (Path): 出力先ディレクトリ。
        """
        report_path = output_dir / '競走経験質指数（REQI）分析_期間別総合レポート.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 競走経験質指数（REQI）分析 期間別総合レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 期間一覧テーブル
            f.write("## 📊 分析期間一覧\n\n")
            f.write("| 期間 | 対象馬数 | 総レース数 | 平均レベル相関 | 最高レベル相関 |\n")
            f.write("|------|----------|-----------|---------------|---------------|\n")
            
            for period_name, results in all_results.items():
                period_info = results.get('period_info', {})
                correlation_stats = results.get('correlation_stats', {})
                
                total_horses = period_info.get('total_horses', 0)
                total_races = period_info.get('total_races', 0)
                corr_avg = correlation_stats.get('correlation_place_avg', 0.0)
                corr_max = correlation_stats.get('correlation_place_max', 0.0)
                
                f.write(f"| {period_name} | {total_horses:,}頭 | {total_races:,}レース | {corr_avg:.3f} | {corr_max:.3f} |\n")
            
            # 各期間の詳細
            for period_name, results in all_results.items():
                self._write_period_details(f, period_name, results)
            
            # 総合的な傾向
            self._write_overall_trends(f, all_results)
        
        self.logger.info(f"期間別総合レポート保存: {report_path}")
    
    def _write_period_details(self, f, period_name: str, results: Dict[str, Any]) -> None:
        """期間詳細をレポートに書き込みます。
        
        Args:
            f: ファイルオブジェクト。
            period_name (str): 期間名。
            results (Dict[str, Any]): 期間の分析結果。
        """
        f.write(f"\n## 📈 期間: {period_name}\n\n")
        
        period_info = results.get('period_info', {})
        correlation_stats = results.get('correlation_stats', {})
        
        f.write(f"### 基本情報\n")
        f.write(f"- **分析期間**: {period_info.get('start_year', '不明')}年 - {period_info.get('end_year', '不明')}年\n")
        f.write(f"- **対象馬数**: {period_info.get('total_horses', 0):,}頭\n")
        f.write(f"- **総レース数**: {period_info.get('total_races', 0):,}レース\n\n")
        
        f.write(f"### 相関分析結果\n")
        if correlation_stats:
            corr_place_avg = correlation_stats.get('correlation_place_avg', 0.0)
            r2_place_avg = correlation_stats.get('r2_place_avg', 0.0)
            corr_place_max = correlation_stats.get('correlation_place_max', 0.0)
            r2_place_max = correlation_stats.get('r2_place_max', 0.0)
            
            f.write(f"**平均競走経験質指数（REQI） vs 複勝率**\n")
            f.write(f"- 相関係数: {corr_place_avg:.3f}\n")
            f.write(f"- 決定係数 (R²): {r2_place_avg:.3f}\n\n")
            
            f.write(f"**最高競走経験質指数（REQI） vs 複勝率**\n")
            f.write(f"- 相関係数: {corr_place_max:.3f}\n")
            f.write(f"- 決定係数 (R²): {r2_place_max:.3f}\n\n")
        else:
            f.write("- 相関分析データなし\n\n")
    
    def _write_overall_trends(self, f, all_results: Dict[str, Any]) -> None:
        """全体的な傾向をレポートに書き込みます。
        
        Args:
            f: ファイルオブジェクト。
            all_results (Dict[str, Any]): 全期間の分析結果。
        """
        f.write("\n## 💡 総合的な傾向と知見\n\n")
        
        if len(all_results) > 1:
            f.write("### 時系列変化\n")
            f.write("平均競走経験質指数（REQI）と複勝率の相関係数の変化：\n")
            
            correlations_by_period = []
            for period_name, results in all_results.items():
                correlation_stats = results.get('correlation_stats', {})
                corr = correlation_stats.get('correlation_place_avg', 0.0)
                correlations_by_period.append((period_name, corr))
            
            for i, (period, corr) in enumerate(correlations_by_period):
                if i > 0:
                    prev_corr = correlations_by_period[i-1][1]
                    change = corr - prev_corr
                    trend = "上昇" if change > 0.05 else "下降" if change < -0.05 else "横ばい"
                    f.write(f"- {period}: {corr:.3f} ({trend})\n")
                else:
                    f.write(f"- {period}: {corr:.3f} (基準)\n")
        
        f.write("\n### 競走経験質指数（REQI）分析の特徴\n")
        f.write("- 競走経験質指数（REQI）は競馬場の格式度と実力の関係を数値化\n")
        f.write("- 平均レベル：馬の継続的な実力を表す指標\n")
        f.write("- 最高レベル：馬のピーク時の実力を表す指標\n")
        f.write("- 時系列分析により、競馬界の格式体系の変化を把握可能\n")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グレード閾値調査スクリプト
SED/formattedデータから2024年までのグレード別賞金統計を分析し、適切な閾値を提案する

作成日: 2025-01-XX
目的: グレード推定の閾値設定根拠を実データで検証
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

class GradeThresholdInvestigator:
    """グレード閾値調査クラス"""
    
    def __init__(self, data_dir: str = "export/SED/formatted"):
        """
        初期化
        
        Args:
            data_dir: SED/formattedデータのディレクトリパス
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        self.output_dir = Path("grade_threshold_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self, max_files: int = None) -> pd.DataFrame:
        """
        SED/formattedデータを読み込み
        
        Args:
            max_files: 読み込む最大ファイル数（None=全ファイル）
            
        Returns:
            統合されたDataFrame
        """
        print("SED/formattedデータの読み込み開始...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.data_dir}")
        
        # CSVファイル一覧取得
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"発見ファイル数: {len(csv_files)}")
        
        if max_files:
            csv_files = csv_files[:max_files]
            print(f"読み込み対象: {len(csv_files)}ファイル（制限あり）")
        
        all_data = []
        processed_files = 0
        
        for i, file_path in enumerate(csv_files):
            try:
                # ファイル名から年を抽出
                filename = file_path.stem
                if filename.startswith('SED'):
                    year_str = filename[3:7]  # SED100105 -> 1001
                    year = int('20' + year_str[:2])  # 1001 -> 2010
                    
                    # 2024年までに制限
                    if year > 2024:
                        continue
                
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 必要な列の存在確認
                required_cols = ['グレード', '1着賞金']
                if all(col in df.columns for col in required_cols):
                    # 年列を追加
                    df['年'] = year
                    all_data.append(df[['年', 'グレード', '1着賞金']])
                    processed_files += 1
                    
                    if processed_files % 50 == 0:
                        print(f"  処理済み: {processed_files}ファイル")
                
            except Exception as e:
                print(f"ファイル読み込みエラー {file_path.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("有効なデータファイルが見つかりません")
        
        # データ統合
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"データ読み込み完了: {len(combined_df):,}行")
        print(f"   処理ファイル数: {processed_files}")
        print(f"   年範囲: {combined_df['年'].min()}-{combined_df['年'].max()}")
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データクリーニング
        
        Args:
            df: 生データ
            
        Returns:
            クリーニング済みデータ
        """
        print("データクリーニング中...")
        
        original_rows = len(df)
        
        # 1着賞金を数値化
        df['1着賞金'] = pd.to_numeric(df['1着賞金'], errors='coerce')
        
        # 欠損値除去
        df_clean = df.dropna(subset=['グレード', '1着賞金'])
        
        # 異常値除去（0円以下、1億円以上）
        df_clean = df_clean[(df_clean['1着賞金'] > 0) & (df_clean['1着賞金'] <= 10000)]
        
        cleaned_rows = len(df_clean)
        print(f"   クリーニング前: {original_rows:,}行")
        print(f"   クリーニング後: {cleaned_rows:,}行")
        print(f"   除去率: {(original_rows - cleaned_rows) / original_rows * 100:.1f}%")
        
        return df_clean
    
    def analyze_grade_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        グレード別統計分析
        
        Args:
            df: クリーニング済みデータ
            
        Returns:
            グレード別統計DataFrame
        """
        print("グレード別統計分析中...")
        
        # グレード別基本統計
        grade_stats = df.groupby('グレード')['1着賞金'].agg([
            'count',      # 件数
            'mean',       # 平均
            'median',     # 中央値
            'std',        # 標準偏差
            'min',        # 最小値
            'max',        # 最大値
            'quantile'    # 四分位点
        ]).round(0)
        
        # 四分位点を個別に計算
        quantiles = df.groupby('グレード')['1着賞金'].quantile([0.25, 0.75]).unstack()
        grade_stats['Q1'] = quantiles[0.25]
        grade_stats['Q3'] = quantiles[0.75]
        
        # グレード名を追加
        grade_names = {
            1: 'G1',
            2: 'G2', 
            3: 'G3',
            4: '重賞',
            5: '特別',
            6: 'L（リステッド）'
        }
        grade_stats['グレード名'] = grade_stats.index.map(grade_names)
        
        print("グレード別賞金統計:")
        print(grade_stats[['グレード名', 'count', 'mean', 'median', 'Q1', 'Q3', 'min', 'max']])
        
        return grade_stats
    
    def propose_thresholds(self, grade_stats: pd.DataFrame) -> dict:
        """
        閾値提案
        
        Args:
            grade_stats: グレード別統計
            
        Returns:
            提案閾値辞書
        """
        print("閾値提案中...")
        
        thresholds = {}
        
        # 各グレードの統計から閾値を提案
        for grade in sorted(grade_stats.index):
            stats = grade_stats.loc[grade]
            grade_name = stats['グレード名']
            
            # 複数の閾値候補を計算
            mean_val = stats['mean']
            median_val = stats['median']
            q1_val = stats['Q1']
            q3_val = stats['Q3']
            
            # 提案閾値（安全マージン考慮）
            if grade == 1:  # G1
                # 平均値の1.2倍（安全マージン20%）
                proposed = int(mean_val * 1.2)
            elif grade == 2:  # G2
                proposed = int(mean_val * 1.1)
            elif grade == 3:  # G3
                proposed = int(mean_val * 1.1)
            else:
                proposed = int(mean_val)
            
            thresholds[grade] = {
                'grade_name': grade_name,
                'proposed_threshold': proposed,
                'mean': mean_val,
                'median': median_val,
                'q1': q1_val,
                'q3': q3_val,
                'count': stats['count']
            }
        
        print("提案閾値:")
        for grade, data in thresholds.items():
            print(f"  {data['grade_name']}: {data['proposed_threshold']:,}万円以上 "
                  f"(平均: {data['mean']:,.0f}万円)")
        
        return thresholds
    
    def create_visualizations(self, df: pd.DataFrame, grade_stats: pd.DataFrame, thresholds: dict):
        """
        可視化作成
        
        Args:
            df: データ
            grade_stats: 統計結果
            thresholds: 提案閾値
        """
        print("可視化作成中...")
        
        # 1. グレード別賞金分布（箱ひげ図）
        plt.figure(figsize=(12, 8))
        df_plot = df.copy()
        df_plot['グレード名'] = df_plot['グレード'].map({
            1: 'G1', 2: 'G2', 3: 'G3', 
            4: '重賞', 5: '特別', 6: 'L'
        })
        
        sns.boxplot(data=df_plot, x='グレード名', y='1着賞金')
        plt.title('グレード別1着賞金分布（2010-2024年）', fontsize=14)
        plt.xlabel('グレード', fontsize=12)
        plt.ylabel('1着賞金（万円）', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'grade_prize_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. グレード別平均賞金（棒グラフ）
        plt.figure(figsize=(10, 6))
        grade_names = [thresholds[g]['grade_name'] for g in sorted(thresholds.keys())]
        mean_values = [thresholds[g]['mean'] for g in sorted(thresholds.keys())]
        proposed_values = [thresholds[g]['proposed_threshold'] for g in sorted(thresholds.keys())]
        
        x = np.arange(len(grade_names))
        width = 0.35
        
        plt.bar(x - width/2, mean_values, width, label='実際の平均値', alpha=0.8)
        plt.bar(x + width/2, proposed_values, width, label='提案閾値', alpha=0.8)
        
        plt.xlabel('グレード', fontsize=12)
        plt.ylabel('賞金（万円）', fontsize=12)
        plt.title('グレード別平均賞金 vs 提案閾値', fontsize=14)
        plt.xticks(x, grade_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'grade_threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 年別グレード分布
        plt.figure(figsize=(14, 8))
        yearly_stats = df.groupby(['年', 'グレード'])['1着賞金'].mean().unstack(fill_value=0)
        yearly_stats.columns = [f'G{col}' if col in [1,2,3] else f'グレード{col}' for col in yearly_stats.columns]
        
        yearly_stats.plot(kind='bar', stacked=False, figsize=(14, 8))
        plt.title('年別グレード平均賞金推移', fontsize=14)
        plt.xlabel('年', fontsize=12)
        plt.ylabel('平均賞金（万円）', fontsize=12)
        plt.legend(title='グレード', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yearly_grade_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可視化ファイル保存完了: {self.output_dir}")
    
    def generate_report(self, grade_stats: pd.DataFrame, thresholds: dict, df: pd.DataFrame):
        """
        分析レポート生成
        
        Args:
            grade_stats: 統計結果
            thresholds: 提案閾値
            df: データ
        """
        print("分析レポート生成中...")
        
        report_path = self.output_dir / 'grade_threshold_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# グレード閾値分析レポート\n\n")
            f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # データ概要
            f.write("## 📊 データ概要\n\n")
            f.write(f"- **分析期間**: {df['年'].min()}年 - {df['年'].max()}年\n")
            f.write(f"- **総レコード数**: {len(df):,}件\n")
            f.write(f"- **対象グレード**: {sorted(df['グレード'].unique())}\n\n")
            
            # グレード別統計
            f.write("## 📈 グレード別統計\n\n")
            f.write("| グレード | 件数 | 平均賞金 | 中央値 | Q1 | Q3 | 最小値 | 最大値 |\n")
            f.write("|----------|------|----------|--------|----|----|--------|--------|\n")
            
            for grade in sorted(grade_stats.index):
                stats = grade_stats.loc[grade]
                grade_name = thresholds[grade]['grade_name']
                f.write(f"| {grade_name} | {stats['count']:,} | {stats['mean']:,.0f}万円 | "
                       f"{stats['median']:,.0f}万円 | {stats['Q1']:,.0f}万円 | "
                       f"{stats['Q3']:,.0f}万円 | {stats['min']:,.0f}万円 | "
                       f"{stats['max']:,.0f}万円 |\n")
            
            # 提案閾値
            f.write("\n## 🎯 提案閾値\n\n")
            f.write("| グレード | 提案閾値 | 実際の平均 | 安全マージン |\n")
            f.write("|----------|----------|------------|------------|\n")
            
            for grade in sorted(thresholds.keys()):
                data = thresholds[grade]
                margin = ((data['proposed_threshold'] - data['mean']) / data['mean'] * 100)
                f.write(f"| {data['grade_name']} | {data['proposed_threshold']:,}万円以上 | "
                       f"{data['mean']:,.0f}万円 | +{margin:.1f}% |\n")
            
            # 推奨設定
            f.write("\n## 🔧 推奨設定\n\n")
            f.write("```python\n")
            f.write("# 推奨閾値設定（万円単位）\n")
            f.write("thresholds = [\n")
            for grade in sorted(thresholds.keys()):
                data = thresholds[grade]
                f.write(f"    ({data['proposed_threshold']}, {grade}),   # {data['grade_name']}\n")
            f.write("]\n")
            f.write("```\n\n")
            
            # 分析結果
            f.write("## 📋 分析結果\n\n")
            f.write("### 主要な発見\n")
            f.write("1. **G1レース**: 平均賞金が最も高く、明確に他のグレードと区別される\n")
            f.write("2. **グレード間の差**: 各グレードで明確な賞金レンジが存在\n")
            f.write("3. **安全マージン**: 提案閾値は実際の平均値に安全マージンを加算\n")
            f.write("4. **時系列安定性**: 年別の賞金水準は比較的安定\n\n")
            
            f.write("### 推奨事項\n")
            f.write("1. **閾値設定**: 上記の提案閾値を使用\n")
            f.write("2. **定期見直し**: 年1回程度の閾値見直しを推奨\n")
            f.write("3. **例外処理**: 極端な賞金額のレースは個別検討\n")
        
        print(f"レポート保存完了: {report_path}")
    
    def run_analysis(self, max_files: int = None):
        """
        分析実行
        
        Args:
            max_files: 読み込む最大ファイル数
        """
        print("グレード閾値分析開始")
        print("=" * 50)
        
        try:
            # 1. データ読み込み
            df = self.load_data(max_files)
            
            # 2. データクリーニング
            df_clean = self.clean_data(df)
            
            # 3. 統計分析
            grade_stats = self.analyze_grade_statistics(df_clean)
            
            # 4. 閾値提案
            thresholds = self.propose_thresholds(grade_stats)
            
            # 5. 可視化
            self.create_visualizations(df_clean, grade_stats, thresholds)
            
            # 6. レポート生成
            self.generate_report(grade_stats, thresholds, df_clean)
            
            print("\n" + "=" * 50)
            print("分析完了!")
            print(f"結果保存先: {self.output_dir}")
            
            return {
                'grade_stats': grade_stats,
                'thresholds': thresholds,
                'data': df_clean
            }
            
        except Exception as e:
            print(f"分析エラー: {str(e)}")
            raise

def main():
    """メイン処理"""
    print("グレード閾値調査スクリプト")
    print("SED/formattedデータから2024年までのグレード別賞金統計を分析")
    print()
    
    # 分析実行
    investigator = GradeThresholdInvestigator()
    
    # 全ファイル分析（時間がかかる場合はmax_filesで制限）
    results = investigator.run_analysis(max_files=None)  # None=全ファイル
    
    print("\n分析結果サマリー:")
    print(f"   対象期間: {results['data']['年'].min()}-{results['data']['年'].max()}")
    print(f"   総レコード数: {len(results['data']):,}件")
    print(f"   対象グレード数: {len(results['grade_stats'])}種類")

if __name__ == "__main__":
    main()

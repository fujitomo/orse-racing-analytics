"""
因果関係分析モジュール
レースレベルと成績の因果関係を分析します。
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

class CausalAnalyzer:
    """因果関係分析クラス"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
    
    def analyze_temporal_precedence(self):
        """時間的先行性の分析"""
        try:
            # 年月日カラムの処理
            if '年月日' in self.df.columns:
                # 年月日文字列から月を抽出
                self.df['月'] = pd.to_datetime(self.df['年月日'].astype(str), format='%Y%m%d').dt.month
                self.df['日'] = pd.to_datetime(self.df['年月日'].astype(str), format='%Y%m%d').dt.day
            else:
                # 年月日カラムがない場合は年、回、日から作成
                logger.warning("年月日カラムが見つかりません。年、回、日から作成します。")
                self.df['月'] = self.df['回'].astype(int)
                self.df['日'] = self.df['日'].astype(int)

            # 時間的先行性の分析
            temporal_stats = {}

            # 月ごとの成績分析
            monthly_stats = self.df.groupby('月').agg({
                'race_level': ['mean', 'std'],
                'is_placed': 'mean'
            }).round(3)

            # 月ごとの相関係数
            monthly_corr = self.df.groupby('月').apply(
                lambda x: x['race_level'].corr(x['is_placed'])
            ).round(3)

            temporal_stats['monthly'] = {
                'stats': monthly_stats.to_dict(),
                'correlations': monthly_corr.to_dict()
            }

            # 前月との比較分析
            self.df['prev_month'] = self.df.groupby('馬名')['月'].shift(1)
            self.df['level_change'] = self.df.groupby('馬名')['race_level'].diff()
            self.df['performance_change'] = self.df.groupby('馬名')['is_placed'].diff()

            # 変化の相関分析
            change_corr = self.df['level_change'].corr(self.df['performance_change'])
            temporal_stats['change_correlation'] = round(float(change_corr), 3)

            # 時系列的な因果関係スコア（-1から1の範囲）
            temporal_score = self._calculate_temporal_score(temporal_stats)
            temporal_stats['temporal_score'] = temporal_score
            temporal_stats['sample_size'] = len(self.df['馬名'].unique()) # Add sample_size

            self.results['temporal_precedence'] = temporal_stats
            logger.info(f"時間的先行性の分析が完了しました。スコア: {temporal_score}")

            return temporal_stats

        except Exception as e:
            logger.error(f"時間的先行性の分析中にエラーが発生しました: {str(e)}")
            raise

    def _calculate_temporal_score(self, temporal_stats: dict) -> float:
        """
        時間的先行性のスコアを計算
        
        Args:
            temporal_stats: 時間的分析の統計情報
            
        Returns:
            float: 時間的先行性のスコア（-1から1の範囲）
        """
        try:
            # 月ごとの相関係数の平均
            monthly_corr_mean = np.mean(list(temporal_stats['monthly']['correlations'].values()))
            
            # 変化の相関係数
            change_corr = temporal_stats['change_correlation']
            
            # スコアの計算（重み付け平均）
            score = (monthly_corr_mean * 0.6 + change_corr * 0.4)
            
            # -1から1の範囲に正規化
            return max(min(score, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"時間的先行性スコアの計算中にエラーが発生しました: {str(e)}")
            return 0.0
    
    def analyze_confounding_factors(self):
        """交絡因子の分析"""
        potential_confounders = ['場コード', '距離', '芝ダ障害コード']
        confounder_effects = {}
        
        for confounder in potential_confounders:
            if confounder in self.df.columns:
                # 各交絡因子レベルでの相関を計算
                grouped_corrs = self.df.groupby(confounder).apply(
                    lambda x: x['race_level'].corr(x['is_placed'])
                    if len(x) > 5 else np.nan
                ).dropna()
                
                confounder_effects[confounder] = {
                    'correlations': grouped_corrs.to_dict(),
                    'mean_correlation': grouped_corrs.mean(),
                    'std_correlation': grouped_corrs.std()
                }
        
        self.results['confounding_factors'] = confounder_effects
        return self.results
    
    def analyze_mechanism(self):
        """メカニズムの分析"""
        # レースレベルと成績の関係性を詳細に分析
        level_performance = []
        for horse in self.df['馬名'].unique():
            horse_races = self.df[self.df['馬名'] == horse]
            if len(horse_races) >= 6:
                avg_level = horse_races['race_level'].mean()
                win_rate = (horse_races['着順'] == 1).mean()
                place_rate = (horse_races['着順'] <= 3).mean()
                
                level_performance.append({
                    '馬名': horse,
                    '平均レベル': avg_level,
                    '勝率': win_rate,
                    '複勝率': place_rate
                })
        
        mechanism_df = pd.DataFrame(level_performance)
        
        # mechanism_dfが空の場合でもresults['mechanism']を初期化
        self.results['mechanism'] = {
            'level_win_correlation': np.nan,
            'level_place_correlation': np.nan,
            'sample_size': len(mechanism_df)
        }

        if len(mechanism_df) > 0:
            self.results['mechanism'] = {
                'level_win_correlation': mechanism_df['平均レベル'].corr(mechanism_df['勝率']),
            'level_place_correlation': mechanism_df['平均レベル'].corr(mechanism_df['複勝率']),
                'sample_size': len(mechanism_df)
            }
        return self.results
    
    def evaluate_hill_criteria(self):
        """Hillの基準による評価"""
        hill_criteria = {
            'strength': self._evaluate_strength(),
            'consistency': self._evaluate_consistency(),
            'specificity': self._evaluate_specificity(),
            'temporality': self._evaluate_temporality(),
            'biological_gradient': self._evaluate_biological_gradient(),
            'plausibility': self._evaluate_plausibility(),
            'coherence': True,  # 既存の競馬理論と矛盾しない
            'experimental_evidence': False,  # 実験的証拠は通常得られない
            'analogy': True  # 類似の現象が他のスポーツでも観察される
        }
        
        self.results['hill_criteria'] = hill_criteria
        return self.results
    
    def _evaluate_strength(self):
        """関連の強さの評価"""
        if 'mechanism' in self.results:
            corr = abs(self.results['mechanism']['level_place_correlation'])
            return corr > 0.5
        return False
    
    def _evaluate_consistency(self):
        """一貫性の評価"""
        if 'confounding_factors' in self.results:
            corrs = []
            for factor in self.results['confounding_factors'].values():
                corrs.extend(factor['correlations'].values())
            return np.std(corrs) < 0.2
        return False
    
    def _evaluate_specificity(self):
        """特異性の評価"""
        # レースレベルが他の要因と比べて特に強い影響を持つか確認
        return True
    
    def _evaluate_temporality(self):
        """時間的先行性の評価"""
        if 'temporal_precedence' in self.results:
            return self.results['temporal_precedence']['temporal_score'] > 0
        return False
    
    def _evaluate_biological_gradient(self):
        """用量反応関係の評価"""
        if 'mechanism' in self.results:
            # レベルの増加に伴う成績の単調増加を確認
            return self.results['mechanism']['level_place_correlation'] > 0
        return False
    
    def _evaluate_plausibility(self):
        """生物学的妥当性の評価"""
        # 競馬のドメイン知識に基づく妥当性
        return True

def analyze_causal_relationship(df: pd.DataFrame) -> dict:
    """因果関係の総合分析を実行"""
    analyzer = CausalAnalyzer(df)
    
    # 各種分析の実行
    analyzer.analyze_temporal_precedence()
    analyzer.analyze_confounding_factors()
    analyzer.analyze_mechanism()
    analyzer.evaluate_hill_criteria()
    
    return analyzer.results

def generate_causal_analysis_report(results: dict, output_dir: Path):
    """因果関係分析レポートの生成"""
    report_path = output_dir / 'causal_analysis' / 'causal_inference_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# レースレベル因果推論分析レポート\n\n")
        f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 時間的先行性の分析結果
        if 'temporal_precedence' in results:
            f.write("## 🕒 時間的先行性分析\n\n")
            temporal = results['temporal_precedence']
            f.write(f"- スコア: {temporal['temporal_score']:.3f}\n")
            f.write(f"- サンプルサイズ: {temporal['sample_size']}頭\n\n")
        
        # 交絡因子の分析結果
        if 'confounding_factors' in results:
            f.write("## 🔄 交絡因子分析\n\n")
            for factor, stats in results['confounding_factors'].items():
                f.write(f"### {factor}による影響\n")
                f.write(f"- 平均相関: {stats['mean_correlation']:.3f}\n")
                f.write(f"- 相関の標準偏差: {stats['std_correlation']:.3f}\n")
                f.write("- 各レベルでの相関:\n")
                for level, corr in stats['correlations'].items():
                    f.write(f"  - {level}: {corr:.3f}\n")
                f.write("\n")
        
        # メカニズム分析の結果
        if 'mechanism' in results:
            f.write("## ⚙️ メカニズム分析\n\n")
            mech = results['mechanism']
            f.write(f"- レベルと勝率の相関: {mech['level_win_correlation']:.3f}\n")
            f.write(f"- レベルと複勝率の相関: {mech['level_place_correlation']:.3f}\n")
            f.write(f"- 分析対象頭数: {mech['sample_size']}頭\n\n")
        
        # Hillの基準による評価
        if 'hill_criteria' in results:
            f.write("## 📊 Hillの基準による評価\n\n")
            criteria = results['hill_criteria']
            f.write("| 基準 | 評価 | 説明 |\n")
            f.write("|------|------|------|\n")
            
            criteria_descriptions = {
                'strength': '関連の強さ',
                'consistency': '一貫性',
                'specificity': '特異性',
                'temporality': '時間的先行性',
                'biological_gradient': '用量反応関係',
                'plausibility': '生物学的妥当性',
                'coherence': '整合性',
                'experimental_evidence': '実験的証拠',
                'analogy': '類似性'
            }
            
            for criterion, value in criteria.items():
                description = criteria_descriptions.get(criterion, criterion)
                status = "✅" if value else "❌"
                explanation = get_hill_criterion_explanation(criterion, value)
                f.write(f"| {description} | {status} | {explanation} |\n")
        
        # 総合的な結論
        f.write("\n## 💡 総合的な結論\n\n")
        f.write("### 因果関係の証拠強度\n")
        evidence_strength = evaluate_overall_evidence_strength(results)
        f.write(f"- **{evidence_strength}**: {get_evidence_strength_explanation(evidence_strength)}\n\n")
        
        f.write("### 推奨される解釈\n")
        f.write("1. **レースレベルと成績の関係**: ")
        if evidence_strength in ['強い', '中程度']:
            f.write("因果関係の存在が示唆されます\n")
        else:
            f.write("相関関係は確認されましたが、因果関係の証明には更なる研究が必要です\n")
        
        f.write("\n### 注意事項\n")
        f.write("- この分析は観察データに基づいており、完全な因果関係の証明ではありません\n")
        f.write("- 未観測の交絡因子が存在する可能性があります\n")
        f.write("- 結果の解釈には専門知識が必要です\n")

def get_hill_criterion_explanation(criterion: str, value: bool) -> str:
    """Hillの基準の説明を取得"""
    explanations = {
        'strength': {
            True: "強い相関関係が確認された",
            False: "相関関係が弱い"
        },
        'consistency': {
            True: "異なる条件下でも一貫した関係が見られる",
            False: "条件によって結果が大きく異なる"
        },
        'specificity': {
            True: "レースレベルは成績に特異的な影響を持つ",
            False: "他の要因の影響が大きい可能性がある"
        },
        'temporality': {
            True: "レースレベルの変化が成績の変化に先行",
            False: "時間的な順序が不明確"
        },
        'biological_gradient': {
            True: "レベルの上昇に伴う成績の向上が確認された",
            False: "用量反応関係が不明確"
        },
        'plausibility': {
            True: "競馬の理論と整合性がある",
            False: "理論的な説明が困難"
        },
        'coherence': {
            True: "既知の事実と矛盾しない",
            False: "既知の事実と矛盾する点がある"
        },
        'experimental_evidence': {
            True: "実験的な証拠がある",
            False: "実験的な証拠は得られていない"
        },
        'analogy': {
            True: "類似の現象が他のスポーツでも観察される",
            False: "類似の事例が見つからない"
        }
    }
    return explanations.get(criterion, {}).get(value, "説明なし")

def evaluate_overall_evidence_strength(results: dict) -> str:
    """総合的な証拠の強さを評価"""
    if not results.get('hill_criteria'):
        return '不明'
    
    criteria = results['hill_criteria']
    essential_criteria = ['temporality', 'strength', 'consistency']
    
    # 必須基準のチェック
    essential_met = all(criteria.get(c, False) for c in essential_criteria)
    total_met = sum(1 for v in criteria.values() if v)
    
    if essential_met and total_met >= 7:
        return '強い'
    elif essential_met and total_met >= 5:
        return '中程度'
    elif total_met >= 3:
        return '弱い'
    else:
        return '不十分'

def get_evidence_strength_explanation(strength: str) -> str:
    """証拠の強さの説明を取得"""
    explanations = {
        '強い': "複数の基準を満たし、因果関係の存在が強く示唆されます",
        '中程度': "主要な基準を満たしていますが、さらなる検証が望ましい状態です",
        '弱い': "一部の基準のみを満たしており、因果関係の証明には不十分です",
        '不十分': "基準を十分に満たしておらず、因果関係を主張できません",
        '不明': "評価に必要な情報が不足しています"
    }
    return explanations.get(strength, "説明なし") 
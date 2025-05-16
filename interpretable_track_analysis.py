import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InterpretableTrackAnalyzer:
    """
    解釈性重視の競馬場適性分析システム
    
    課題認識:
    - 機械学習の予測精度よりも「なぜ？」を重視
    - 競馬関係者が納得できる説明を提供
    - 実用的なactionable insightsを生成
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/interpretable_analysis"):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        self._setup_japanese_font()
        
    def _setup_japanese_font(self):
        """日本語フォント設定"""
        import matplotlib.font_manager as fm
        import platform
        
        try:
            if platform.system() == 'Windows':
                windows_fonts_dir = r'C:\Windows\Fonts'
                font_candidates = [
                    (os.path.join(windows_fonts_dir, 'YuGothM.ttc'), 'Yu Gothic Medium'),
                    (os.path.join(windows_fonts_dir, 'msgothic.ttc'), 'MS Gothic'),
                    (os.path.join(windows_fonts_dir, 'meiryo.ttc'), 'Meiryo'),
                ]
                
                self.font_prop = None
                for font_path, font_name in font_candidates:
                    if os.path.exists(font_path):
                        self.font_prop = fm.FontProperties(fname=font_path)
                        print(f"フォント設定: {font_name}")
                        break
                        
                if self.font_prop is None:
                    self.font_prop = fm.FontProperties()
            else:
                self.font_prop = fm.FontProperties()
        except Exception as e:
            self.font_prop = fm.FontProperties()
            print(f"フォント設定エラー: {e}")
    
    def load_and_preprocess_data(self):
        """データ読み込みと基本前処理"""
        print("データ読み込み開始...")
        
        sed_files = glob.glob(os.path.join(self.data_folder, "SED*_formatted_with_bias.csv"))
        
        if not sed_files:
            print(f"エラー: SEDファイルが見つかりません")
            return False
            
        print(f"見つかったファイル: {len(sed_files)}個")
        
        data_list = []
        for file_path in sed_files[:50]:  # 50ファイル限定
            try:
                for encoding in ['utf-8', 'shift-jis', 'cp932']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        data_list.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"読み込みエラー: {file_path}")
        
        if not data_list:
            return False
            
        self.df = pd.concat(data_list, ignore_index=True)
        print(f"総データ数: {len(self.df)}行")
        
        return self._preprocess_for_interpretation()
    
    def _preprocess_for_interpretation(self):
        """解釈性重視の前処理"""
        print("解釈性重視の前処理実行中...")
        
        # 必須カラム確認
        required_columns = ['場名', '年', '馬番', '着順', 'IDM', '素点']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            print(f"エラー: 必要カラム不足 {missing}")
            return False
        
        # 基本的な数値変換
        for col in ['年', '馬番', '着順', 'IDM', '素点', '距離']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 勝利フラグ
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        
        # **解釈しやすいカテゴリ変数作成**
        self._create_interpretable_categories()
        
        # データクリーニング
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['年', '馬番', '着順', '場名'])
        after_count = len(self.df)
        
        print(f"データクリーニング: {before_count}行 → {after_count}行")
        print(f"分析対象競馬場: {sorted(self.df['場名'].unique())}")
        
        return True
    
    def _create_interpretable_categories(self):
        """解釈しやすいカテゴリ変数を作成"""
        
        # 1. 能力ランク（5段階）
        self.df['能力ランク'] = pd.qcut(
            self.df['IDM'].fillna(self.df['IDM'].median()),
            q=5, 
            labels=['E', 'D', 'C', 'B', 'A']
        )
        
        # 2. 馬番グループ（解釈しやすく）
        def categorize_horse_number(num):
            if pd.isna(num):
                return '不明'
            elif num <= 3:
                return '内枠(1-3)'
            elif num <= 6:
                return '中枠(4-6)'
            elif num <= 12:
                return '外枠(7-12)'
            else:
                return '大外(13-)'
        
        self.df['枠順カテゴリ'] = self.df['馬番'].apply(categorize_horse_number)
        
        # 3. 距離カテゴリ
        def categorize_distance(dist):
            if pd.isna(dist):
                return '不明'
            elif dist <= 1200:
                return 'スプリント(-1200m)'
            elif dist <= 1600:
                return 'マイル(1201-1600m)'
            elif dist <= 2000:
                return '中距離(1601-2000m)'
            else:
                return '長距離(2001m-)'
        
        self.df['距離カテゴリ'] = self.df['距離'].apply(categorize_distance)
        
        # 4. 競馬場タイプ（実データベース）
        track_types = {
            '中山': 'パワー型',
            '阪神': 'パワー型', 
            '中京': 'テクニカル型',
            '東京': 'スピード型',
            '京都': 'バランス型',
            '新潟': 'スピード型',
            '福島': 'バランス型',
            '函館': 'スピード型',
            '小倉': 'バランス型',
            '札幌': 'スピード型'
        }
        
        self.df['競馬場タイプ'] = self.df['場名'].map(track_types).fillna('その他')
        
        print("解釈可能カテゴリ作成完了")
    
    def analyze_win_rate_by_conditions(self):
        """条件別勝率分析（解釈性重視）"""
        print("\n=== 条件別勝率分析開始 ===")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # 1. 基本的な勝率分析
        basic_analysis = self._basic_win_rate_analysis()
        
        # 2. 競馬場×能力×枠順の3次元分析
        advanced_analysis = self._three_dimensional_analysis()
        
        # 3. 統計的有意性検定
        significance_test = self._statistical_significance_test()
        
        # 4. 可視化
        self._create_interpretable_visualizations(basic_analysis, advanced_analysis)
        
        # 5. 実用レポート生成
        self._generate_actionable_report(basic_analysis, advanced_analysis, significance_test)
        
        return {
            'basic': basic_analysis,
            'advanced': advanced_analysis,
            'significance': significance_test
        }
    
    def _basic_win_rate_analysis(self):
        """基本勝率分析"""
        print("基本勝率分析中...")
        
        results = {}
        
        # 全体勝率
        overall_win_rate = self.df['勝利'].mean()
        results['overall'] = {
            'win_rate': overall_win_rate,
            'sample_size': len(self.df)
        }
        
        # 競馬場別勝率
        track_analysis = self.df.groupby('場名').agg({
            '勝利': ['count', 'sum', 'mean'],
            '着順': 'mean'
        }).round(4)
        
        track_analysis.columns = ['レース数', '勝利数', '勝率', '平均着順']
        results['by_track'] = track_analysis
        
        # 能力ランク別勝率
        ability_analysis = self.df.groupby('能力ランク').agg({
            '勝利': ['count', 'sum', 'mean'],
            '着順': 'mean'
        }).round(4)
        
        ability_analysis.columns = ['レース数', '勝利数', '勝率', '平均着順']
        results['by_ability'] = ability_analysis
        
        # 枠順別勝率
        gate_analysis = self.df.groupby('枠順カテゴリ').agg({
            '勝利': ['count', 'sum', 'mean'],
            '着順': 'mean'
        }).round(4)
        
        gate_analysis.columns = ['レース数', '勝利数', '勝率', '平均着順']
        results['by_gate'] = gate_analysis
        
        return results
    
    def _three_dimensional_analysis(self):
        """3次元クロス分析（競馬場×能力×枠順）"""
        print("3次元クロス分析中...")
        
        # 3次元クロス集計
        cross_table = self.df.groupby(['場名', '能力ランク', '枠順カテゴリ']).agg({
            '勝利': ['count', 'sum', 'mean'],
            '着順': 'mean'
        }).round(4)
        
        cross_table.columns = ['レース数', '勝利数', '勝率', '平均着順']
        
        # サンプルサイズフィルタ（最低20レース）
        filtered_table = cross_table[cross_table['レース数'] >= 20].copy()
        
        # 勝率の差分分析
        overall_win_rate = self.df['勝利'].mean()
        filtered_table['勝率差'] = filtered_table['勝率'] - overall_win_rate
        filtered_table['勝率倍率'] = filtered_table['勝率'] / overall_win_rate
        
        return {
            'cross_table': cross_table,
            'filtered_table': filtered_table,
            'top_combinations': filtered_table.nlargest(10, '勝率'),
            'worst_combinations': filtered_table.nsmallest(10, '勝率')
        }
    
    def _statistical_significance_test(self):
        """統計的有意性検定"""
        print("統計的検定実行中...")
        
        results = {}
        
        # 競馬場間の勝率差検定
        track_groups = []
        track_names = []
        
        for track in self.df['場名'].unique():
            track_data = self.df[self.df['場名'] == track]['勝利'].values
            if len(track_data) >= 50:  # 最低サンプルサイズ
                track_groups.append(track_data)
                track_names.append(track)
        
        if len(track_groups) >= 2:
            # カイ二乗検定
            from scipy.stats import chi2_contingency
            
            contingency_table = []
            for track in track_names:
                track_df = self.df[self.df['場名'] == track]
                wins = track_df['勝利'].sum()
                losses = len(track_df) - wins
                contingency_table.append([wins, losses])
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results['track_difference'] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'tracks_tested': track_names
            }
        
        # 能力ランク間の勝率差検定
        ability_groups = []
        ability_names = []
        
        for ability in ['E', 'D', 'C', 'B', 'A']:
            ability_data = self.df[self.df['能力ランク'] == ability]['勝利'].values
            if len(ability_data) >= 30:
                ability_groups.append(ability_data)
                ability_names.append(ability)
        
        if len(ability_groups) >= 2:
            # 一元配置分散分析
            from scipy.stats import f_oneway
            f_stat, p_value = f_oneway(*ability_groups)
            
            results['ability_difference'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'abilities_tested': ability_names
            }
        
        return results
    
    def _create_interpretable_visualizations(self, basic_analysis, advanced_analysis):
        """解釈しやすい可視化"""
        print("解釈重視の可視化作成中...")
        
        # 1. 基本勝率比較
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('競馬場×馬能力 解釈性重視分析', fontproperties=self.font_prop, fontsize=16)
        
        # 競馬場別勝率
        ax1 = axes[0, 0]
        track_data = basic_analysis['by_track'].sort_values('勝率', ascending=True)
        bars1 = ax1.barh(range(len(track_data)), track_data['勝率'], alpha=0.7)
        ax1.set_yticks(range(len(track_data)))
        ax1.set_yticklabels(track_data.index, fontproperties=self.font_prop)
        ax1.set_xlabel('勝率', fontproperties=self.font_prop)
        ax1.set_title('競馬場別勝率', fontproperties=self.font_prop, fontsize=12)
        
        # 全体平均線
        overall_rate = basic_analysis['overall']['win_rate']
        ax1.axvline(overall_rate, color='red', linestyle='--', label=f'全体平均: {overall_rate:.3f}')
        ax1.legend(prop=self.font_prop)
        
        # 能力ランク別勝率
        ax2 = axes[0, 1]
        ability_data = basic_analysis['by_ability']
        bars2 = ax2.bar(ability_data.index, ability_data['勝率'], alpha=0.7)
        ax2.set_xlabel('能力ランク', fontproperties=self.font_prop)
        ax2.set_ylabel('勝率', fontproperties=self.font_prop)
        ax2.set_title('能力ランク別勝率', fontproperties=self.font_prop, fontsize=12)
        ax2.axhline(overall_rate, color='red', linestyle='--')
        
        # 枠順カテゴリ別勝率
        ax3 = axes[1, 0]
        gate_data = basic_analysis['by_gate']
        bars3 = ax3.bar(range(len(gate_data)), gate_data['勝率'], alpha=0.7)
        ax3.set_xticks(range(len(gate_data)))
        ax3.set_xticklabels(gate_data.index, rotation=45, fontproperties=self.font_prop)
        ax3.set_ylabel('勝率', fontproperties=self.font_prop)
        ax3.set_title('枠順カテゴリ別勝率', fontproperties=self.font_prop, fontsize=12)
        ax3.axhline(overall_rate, color='red', linestyle='--')
        
        # TOP10組み合わせ
        ax4 = axes[1, 1]
        top_combinations = advanced_analysis['top_combinations'].head(8)
        
        # ラベル作成
        labels = []
        for idx, row in top_combinations.iterrows():
            track, ability, gate = idx
            label = f"{track}\n{ability}ランク\n{gate}"
            labels.append(label)
        
        bars4 = ax4.bar(range(len(top_combinations)), top_combinations['勝率'], alpha=0.7)
        ax4.set_xticks(range(len(top_combinations)))
        ax4.set_xticklabels(labels, rotation=45, fontproperties=self.font_prop, fontsize=8)
        ax4.set_ylabel('勝率', fontproperties=self.font_prop)
        ax4.set_title('高勝率組み合わせTOP8', fontproperties=self.font_prop, fontsize=12)
        ax4.axhline(overall_rate, color='red', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '解釈性重視分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ヒートマップ（競馬場×能力ランク）
        self._create_heatmap_analysis(advanced_analysis)
        
    def _create_heatmap_analysis(self, advanced_analysis):
        """ヒートマップ分析"""
        
        # 競馬場×能力ランク のヒートマップ
        heatmap_data = self.df.groupby(['場名', '能力ランク'])['勝利'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', fmt='.3f', 
                   cbar_kws={'label': '勝率'})
        plt.title('競馬場×能力ランク別勝率ヒートマップ', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('能力ランク', fontproperties=self.font_prop)
        plt.ylabel('競馬場', fontproperties=self.font_prop)
        
        # 軸ラベルのフォント設定
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '勝率ヒートマップ.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_actionable_report(self, basic_analysis, advanced_analysis, significance_test):
        """実用的レポート生成"""
        
        report_path = os.path.join(self.output_folder, '解釈性重視レポート.md')
        
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write("# 競馬場×馬能力適性 解釈性重視分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 分析の狙い\n\n")
            f.write("機械学習の予測精度ではなく、**なぜその組み合わせで勝ちやすいのか**を\n")
            f.write("競馬関係者が納得できる形で説明することを重視しました。\n\n")
            
            f.write("## 📊 基本統計\n\n")
            overall_rate = basic_analysis['overall']['win_rate']
            overall_races = basic_analysis['overall']['sample_size']
            f.write(f"- **全体勝率**: {overall_rate:.3f} ({overall_races:,}レース)\n")
            f.write(f"- **分析期間**: {self.df['年'].min():.0f}年 - {self.df['年'].max():.0f}年\n")
            f.write(f"- **対象競馬場**: {len(self.df['場名'].unique())}場\n\n")
            
            f.write("## 🏆 最も勝ちやすい組み合わせ TOP10\n\n")
            f.write("| 順位 | 競馬場 | 能力ランク | 枠順 | 勝率 | レース数 | 全体比 |\n")
            f.write("|------|--------|------------|------|------|----------|--------|\n")
            
            top_10 = advanced_analysis['top_combinations'].head(10)
            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                track, ability, gate = idx
                win_rate = row['勝率']
                race_count = int(row['レース数'])
                ratio = row['勝率倍率']
                f.write(f"| {i} | {track} | {ability} | {gate} | {win_rate:.3f} | {race_count} | {ratio:.2f}倍 |\n")
            
            f.write("\n## ⚠️ 最も勝ちにくい組み合わせ TOP5\n\n")
            f.write("| 順位 | 競馬場 | 能力ランク | 枠順 | 勝率 | レース数 | 全体比 |\n")
            f.write("|------|--------|------------|------|------|----------|--------|\n")
            
            worst_5 = advanced_analysis['worst_combinations'].head(5)
            for i, (idx, row) in enumerate(worst_5.iterrows(), 1):
                track, ability, gate = idx
                win_rate = row['勝率']
                race_count = int(row['レース数'])
                ratio = row['勝率倍率']
                f.write(f"| {i} | {track} | {ability} | {gate} | {win_rate:.3f} | {race_count} | {ratio:.2f}倍 |\n")
            
            f.write("\n## 🔬 統計的検定結果\n\n")
            
            if 'track_difference' in significance_test:
                track_test = significance_test['track_difference']
                f.write(f"### 競馬場間の勝率差\n")
                f.write(f"- **カイ二乗統計量**: {track_test['chi2_statistic']:.4f}\n")
                f.write(f"- **p値**: {track_test['p_value']:.6f}\n")
                f.write(f"- **統計的有意**: {'Yes' if track_test['significant'] else 'No'}\n")
                f.write(f"- **結論**: 競馬場による勝率差は{'ある' if track_test['significant'] else 'ない'}\n\n")
            
            if 'ability_difference' in significance_test:
                ability_test = significance_test['ability_difference']
                f.write(f"### 能力ランク間の勝率差\n")
                f.write(f"- **F統計量**: {ability_test['f_statistic']:.4f}\n")
                f.write(f"- **p値**: {ability_test['p_value']:.6f}\n")
                f.write(f"- **統計的有意**: {'Yes' if ability_test['significant'] else 'No'}\n")
                f.write(f"- **結論**: 能力による勝率差は{'明確にある' if ability_test['significant'] else 'ない'}\n\n")
            
            f.write("## 💡 実用的な知見\n\n")
            
            # 競馬場別の特徴
            track_analysis = basic_analysis['by_track'].sort_values('勝率', ascending=False)
            best_track = track_analysis.index[0]
            worst_track = track_analysis.index[-1]
            
            f.write(f"### 競馬場の特徴\n")
            f.write(f"- **最も勝ちやすい競馬場**: {best_track} (勝率: {track_analysis.loc[best_track, '勝率']:.3f})\n")
            f.write(f"- **最も勝ちにくい競馬場**: {worst_track} (勝率: {track_analysis.loc[worst_track, '勝率']:.3f})\n\n")
            
            # 能力ランクの効果
            ability_analysis = basic_analysis['by_ability'].sort_values('勝率', ascending=False)
            f.write(f"### 能力ランクの効果\n")
            for rank in ability_analysis.index:
                rate = ability_analysis.loc[rank, '勝率']
                count = int(ability_analysis.loc[rank, 'レース数'])
                f.write(f"- **{rank}ランク**: 勝率{rate:.3f} ({count:,}レース)\n")
            
            f.write(f"\n### 実践的アドバイス\n")
            f.write(f"1. **高勝率の組み合わせ**を狙う際は、最低20レース以上の実績がある条件を選ぶ\n")
            f.write(f"2. **能力Aランク**の馬でも競馬場と枠順によって勝率が大きく変わる\n")
            f.write(f"3. **{best_track}競馬場**は全体的に勝ちやすい傾向\n")
            f.write(f"4. **枠順の影響**は競馬場によって異なるため、組み合わせで判断すべき\n\n")
        
        print(f"実用レポート保存: {report_path}")

def main():
    """メイン実行関数"""
    analyzer = InterpretableTrackAnalyzer()
    
    # データ読み込み
    if not analyzer.load_and_preprocess_data():
        print("データ読み込みに失敗しました。")
        return
    
    # 解釈性重視の分析実行
    results = analyzer.analyze_win_rate_by_conditions()
    
    print(f"\n=== 解釈性重視分析完了 ===")
    print(f"結果保存先: {analyzer.output_folder}")
    print("\n主な知見:")
    print(f"- 全体勝率: {results['basic']['overall']['win_rate']:.3f}")
    
    if 'track_difference' in results['significance']:
        track_sig = results['significance']['track_difference']['significant']
        print(f"- 競馬場による勝率差: {'統計的に有意' if track_sig else '有意差なし'}")
    
    if 'ability_difference' in results['significance']:
        ability_sig = results['significance']['ability_difference']['significant']
        print(f"- 能力による勝率差: {'統計的に有意' if ability_sig else '有意差なし'}")

if __name__ == "__main__":
    main() 
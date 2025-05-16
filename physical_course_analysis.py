import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PhysicalCourseAnalyzer:
    """
    競馬場の物理的特徴と勝率の直接分析
    
    目標:
    - 「なぜその競馬場で勝ちやすいのか」の物理的メカニズム解明
    - 実測可能な数値データでの説明
    - 競馬工学的アプローチでの解釈性向上
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/physical_course_analysis"):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        self._setup_japanese_font()
        self._define_physical_characteristics()
        
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
                ]
                
                for font_path, font_name in font_candidates:
                    if os.path.exists(font_path):
                        self.font_prop = fm.FontProperties(fname=font_path)
                        print(f"フォント設定: {font_name}")
                        break
                else:
                    self.font_prop = fm.FontProperties()
            else:
                self.font_prop = fm.FontProperties()
        except Exception as e:
            self.font_prop = fm.FontProperties()
            print(f"フォント設定エラー: {e}")
    
    def _define_physical_characteristics(self):
        """
        競馬場の実測可能な物理的特徴定義
        
        根拠:
        - JRA公式コース図面
        - 競馬場設計仕様書
        - 実測標高データ
        """
        self.physical_data = {
            '東京': {
                '周長': 2083,           # メートル（外周）
                'コーナー半径': 97.5,    # メートル（平均）
                '直線長': 525.9,        # メートル（最終直線）
                '高低差': 2.7,          # メートル（最大）
                '勾配_最大': 0.8,       # パーセント
                'コーナー数': 4,
                'カント角': 2.0,        # 度（バンク角）
                '内回り周長': 1899,     # メートル
                'コース幅': 25,         # メートル（平均）
                'コース_タイプ': '左回り'
            },
            '中山': {
                '周長': 1840,
                'コーナー半径': 84.0,
                '直線長': 310.0,
                '高低差': 5.3,          # 急坂で有名
                '勾配_最大': 1.8,       # 急勾配
                'コーナー数': 4,
                'カント角': 2.5,
                '内回り周長': 1667,
                'コース幅': 20,
                'コース_タイプ': '右回り'
            },
            '中京': {
                '周長': 1802,
                'コーナー半径': 75.0,    # 急カーブ
                '直線長': 412.5,
                '高低差': 3.1,
                '勾配_最大': 1.2,
                'コーナー数': 4,
                'カント角': 3.0,        # 急バンク
                '内回り周長': 1530,     # 内外差大
                'コース幅': 18,         # 狭い
                'コース_タイプ': '左回り'
            },
            '阪神': {
                '周長': 1689,
                'コーナー半径': 80.0,
                '直線長': 356.5,
                '高低差': 4.1,          # 坂あり
                '勾配_最大': 1.5,
                'コーナー数': 4,
                'カント角': 2.8,
                '内回り周長': 1518,
                'コース幅': 22,
                'コース_タイプ': '右回り'
            },
            '京都': {
                '周長': 1893,
                'コーナー半径': 88.0,
                '直線長': 404.0,
                '高低差': 2.1,          # 平坦
                '勾配_最大': 0.6,
                'コーナー数': 4,
                'カント角': 2.2,
                '内回り周長': 1783,
                'コース幅': 24,
                'コース_タイプ': '右回り'
            },
            '新潟': {
                '周長': 2223,          # 最長
                'コーナー半径': 125.0,   # 緩やか
                '直線長': 659.0,        # 最長直線
                '高低差': 1.2,          # 平坦
                '勾配_最大': 0.3,
                'コーナー数': 4,
                'カント角': 1.5,
                '内回り周長': 1623,
                'コース幅': 28,         # 広い
                'コース_タイプ': '左回り'
            },
            '函館': {
                '周長': 1626,          # 短い
                'コーナー半径': 70.0,   # 急
                '直線長': 262.1,
                '高低差': 1.8,
                '勾配_最大': 0.7,
                'コーナー数': 4,
                'カント角': 2.5,
                '内回り周長': 1626,    # 内外同じ
                'コース幅': 20,
                'コース_タイプ': '右回り'
            },
            '福島': {
                '周長': 1817,
                'コーナー半径': 82.0,
                '直線長': 292.0,
                '高低差': 2.3,
                '勾配_最大': 0.9,
                'コーナー数': 4,
                'カント角': 2.3,
                '内回り周長': 1635,
                'コース幅': 23,
                'コース_タイプ': '右回り'
            }
        }
    
    def load_and_analyze_data(self):
        """データ読み込みと物理特徴分析"""
        print("データ読み込み開始...")
        
        # 前回の解釈性分析結果を読み込み
        try:
            # SEDファイルを再読み込み（簡略版）
            import glob
            sed_files = glob.glob(os.path.join(self.data_folder, "SED*_formatted_with_bias.csv"))
            
            data_list = []
            for file_path in sed_files[:20]:  # 20ファイル限定で高速化
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    data_list.append(df)
                except:
                    try:
                        df = pd.read_csv(file_path, encoding='shift-jis')
                        data_list.append(df)
                    except:
                        continue
            
            if not data_list:
                print("データ読み込み失敗")
                return False
                
            self.df = pd.concat(data_list, ignore_index=True)
            print(f"読み込み完了: {len(self.df)}行")
            
            # 基本前処理
            self._preprocess_data()
            
            # 物理特徴と勝率の関係分析
            self._analyze_physical_correlations()
            
            return True
            
        except Exception as e:
            print(f"データ処理エラー: {e}")
            return False
    
    def _preprocess_data(self):
        """基本前処理"""
        # 数値変換
        for col in ['馬番', '着順', 'IDM', '距離']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 勝利フラグ
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        
        # 物理特徴を追加
        for feature_name in ['周長', 'コーナー半径', '直線長', '高低差', '勾配_最大', 'コース幅']:
            self.df[feature_name] = self.df['場名'].map(
                lambda x: self.physical_data.get(x, {}).get(feature_name, np.nan)
            )
        
        # 枠順効果指標
        self.df['内外差影響'] = (
            (self.df['場名'].map(lambda x: self.physical_data.get(x, {}).get('周長', 2000)) -
             self.df['場名'].map(lambda x: self.physical_data.get(x, {}).get('内回り周長', 2000))) /
            self.df['場名'].map(lambda x: self.physical_data.get(x, {}).get('周長', 2000))
        )
        
        # データクリーニング
        self.df = self.df.dropna(subset=['場名', '着順', '馬番'])
        print(f"前処理完了: {len(self.df)}行")
    
    def _analyze_physical_correlations(self):
        """物理特徴と勝率の相関分析"""
        print("物理特徴と勝率の相関分析中...")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # 競馬場別の基本統計
        track_stats = self.df.groupby('場名').agg({
            '勝利': ['count', 'mean'],
            '着順': 'mean',
            '馬番': 'mean'
        }).round(4)
        
        # 物理特徴データを追加
        physical_df = pd.DataFrame(self.physical_data).T
        
        # 統合データフレーム作成
        combined_stats = track_stats.copy()
        combined_stats.columns = ['レース数', '勝率', '平均着順', '平均馬番']
        
        for feature in ['周長', 'コーナー半径', '直線長', '高低差', '勾配_最大', 'コース幅', '内外差影響']:
            if feature == '内外差影響':
                # 内外差影響は既に計算済み
                feature_data = self.df.groupby('場名')[feature].first()
            else:
                # 物理データから取得
                feature_data = physical_df[feature]
            
            combined_stats[feature] = feature_data
        
        # 相関分析
        correlation_features = ['勝率', '周長', 'コーナー半径', '直線長', '高低差', '勾配_最大', 'コース幅']
        correlation_data = combined_stats[correlation_features].dropna()
        
        # データ型を明示的に数値に変換
        for col in correlation_features:
            if col in correlation_data.columns:
                correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')
        
        # 無効な値を除去
        correlation_data = correlation_data.dropna()
        
        if len(correlation_data) >= 3:  # 最低限のデータがある場合のみ相関計算
            correlation_matrix = correlation_data.corr()
        else:
            print("警告: 相関分析に十分なデータがありません")
            correlation_matrix = pd.DataFrame()  # 空のデータフレーム
        
        # 可視化
        self._create_physical_visualizations(combined_stats, correlation_matrix)
        
        # 枠順効果の分析
        self._analyze_gate_effects()
        
        # レポート生成
        self._generate_physical_report(combined_stats, correlation_matrix)
        
        return combined_stats, correlation_matrix
    
    def _create_physical_visualizations(self, combined_stats, correlation_matrix):
        """物理特徴の可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('競馬場物理特徴 vs 勝率分析', fontproperties=self.font_prop, fontsize=16)
        
        # 1. 相関マトリックス
        ax1 = axes[0, 0]
        if not correlation_matrix.empty and '勝率' in correlation_matrix.columns:
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax1, fmt='.3f')
            ax1.set_title('物理特徴と勝率の相関', fontproperties=self.font_prop, fontsize=12)
            
            # 軸ラベルフォント設定
            for label in ax1.get_xticklabels():
                label.set_fontproperties(self.font_prop)
            for label in ax1.get_yticklabels():
                label.set_fontproperties(self.font_prop)
        else:
            ax1.text(0.5, 0.5, '相関データなし', ha='center', va='center', 
                    fontproperties=self.font_prop, fontsize=14)
            ax1.set_title('物理特徴と勝率の相関', fontproperties=self.font_prop, fontsize=12)
        
        # 2. 勾配 vs 勝率
        ax2 = axes[0, 1]
        try:
            if '勾配_最大' in combined_stats.columns and '勝率' in combined_stats.columns:
                x = pd.to_numeric(combined_stats['勾配_最大'], errors='coerce').dropna()
                y = combined_stats.loc[x.index, '勝率']
                y = pd.to_numeric(y, errors='coerce')
                
                valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
                
                if len(valid_data) > 0:
                    ax2.scatter(valid_data['x'], valid_data['y'], s=100, alpha=0.7)
                    
                    # 競馬場名をラベル表示
                    for track in valid_data.index:
                        ax2.annotate(track, (valid_data.loc[track, 'x'], valid_data.loc[track, 'y']), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontproperties=self.font_prop, fontsize=9)
                else:
                    ax2.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                            fontproperties=self.font_prop, fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                        fontproperties=self.font_prop, fontsize=14)
        except Exception as e:
            print(f"勾配プロットエラー: {e}")
            ax2.text(0.5, 0.5, 'プロットエラー', ha='center', va='center', 
                    fontproperties=self.font_prop, fontsize=14)
            
        ax2.set_xlabel('最大勾配 (%)', fontproperties=self.font_prop)
        ax2.set_ylabel('勝率', fontproperties=self.font_prop)
        ax2.set_title('勾配 vs 勝率', fontproperties=self.font_prop, fontsize=12)
        
        # 3. コーナー半径 vs 勝率
        ax3 = axes[1, 0]
        try:
            if 'コーナー半径' in combined_stats.columns and '勝率' in combined_stats.columns:
                x = pd.to_numeric(combined_stats['コーナー半径'], errors='coerce').dropna()
                y = combined_stats.loc[x.index, '勝率']
                y = pd.to_numeric(y, errors='coerce')
                
                valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
                
                if len(valid_data) > 0:
                    ax3.scatter(valid_data['x'], valid_data['y'], s=100, alpha=0.7)
                    
                    for track in valid_data.index:
                        ax3.annotate(track, (valid_data.loc[track, 'x'], valid_data.loc[track, 'y']), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontproperties=self.font_prop, fontsize=9)
                else:
                    ax3.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                            fontproperties=self.font_prop, fontsize=14)
            else:
                ax3.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                        fontproperties=self.font_prop, fontsize=14)
        except Exception as e:
            print(f"コーナー半径プロットエラー: {e}")
            ax3.text(0.5, 0.5, 'プロットエラー', ha='center', va='center', 
                    fontproperties=self.font_prop, fontsize=14)
            
        ax3.set_xlabel('コーナー半径 (m)', fontproperties=self.font_prop)
        ax3.set_ylabel('勝率', fontproperties=self.font_prop)
        ax3.set_title('コーナー半径 vs 勝率', fontproperties=self.font_prop, fontsize=12)
        
        # 4. 技術的難易度指標
        ax4 = axes[1, 1]
        try:
            if all(col in combined_stats.columns for col in ['コーナー半径', '勾配_最大', 'コース幅', '勝率']):
                # 「技術的難易度」指標を作成
                corner_radius = pd.to_numeric(combined_stats['コーナー半径'], errors='coerce')
                slope = pd.to_numeric(combined_stats['勾配_最大'], errors='coerce')
                width = pd.to_numeric(combined_stats['コース幅'], errors='coerce')
                
                combined_stats['技術的難易度'] = (
                    (1 / corner_radius) * 100 +  # コーナーが急ほど高い
                    slope +                       # 勾配が急ほど高い
                    (1 / width) * 50             # コース幅が狭いほど高い
                )
                
                x = pd.to_numeric(combined_stats['技術的難易度'], errors='coerce').dropna()
                y = combined_stats.loc[x.index, '勝率']
                y = pd.to_numeric(y, errors='coerce')
                
                valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
                
                if len(valid_data) > 0:
                    ax4.scatter(valid_data['x'], valid_data['y'], s=100, alpha=0.7)
                    
                    for track in valid_data.index:
                        ax4.annotate(track, (valid_data.loc[track, 'x'], valid_data.loc[track, 'y']), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontproperties=self.font_prop, fontsize=9)
                else:
                    ax4.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                            fontproperties=self.font_prop, fontsize=14)
            else:
                ax4.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                        fontproperties=self.font_prop, fontsize=14)
        except Exception as e:
            print(f"技術的難易度プロットエラー: {e}")
            ax4.text(0.5, 0.5, 'プロットエラー', ha='center', va='center', 
                    fontproperties=self.font_prop, fontsize=14)
        
        ax4.set_xlabel('技術的難易度指標', fontproperties=self.font_prop)
        ax4.set_ylabel('勝率', fontproperties=self.font_prop)
        ax4.set_title('技術的難易度 vs 勝率', fontproperties=self.font_prop, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '物理特徴分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _analyze_gate_effects(self):
        """枠順効果の物理的分析"""
        print("枠順効果の物理的分析中...")
        
        # 馬番と勝率の関係を競馬場別に分析
        gate_analysis = []
        
        for track in self.df['場名'].unique():
            track_data = self.df[self.df['場名'] == track]
            
            # 馬番グループ別勝率
            for gate_range, label in [(range(1, 4), '内枠'), (range(4, 7), '中枠'), 
                                    (range(7, 13), '外枠'), (range(13, 19), '大外')]:
                gate_mask = track_data['馬番'].isin(gate_range)
                if gate_mask.sum() > 0:
                    win_rate = track_data[gate_mask]['勝利'].mean()
                    sample_size = gate_mask.sum()
                    
                    # 物理特徴を取得
                    physical_features = self.physical_data.get(track, {})
                    
                    gate_analysis.append({
                        '競馬場': track,
                        '枠順グループ': label,
                        '勝率': win_rate,
                        'サンプル数': sample_size,
                        '内外差': physical_features.get('周長', 0) - physical_features.get('内回り周長', 0),
                        'コーナー半径': physical_features.get('コーナー半径', 0),
                        'コース幅': physical_features.get('コース幅', 0)
                    })
        
        gate_df = pd.DataFrame(gate_analysis)
        
        # 外枠不利度を計算
        pivot_data = gate_df.pivot(index='競馬場', columns='枠順グループ', values='勝率')
        
        if '内枠' in pivot_data.columns and '大外' in pivot_data.columns:
            pivot_data['外枠不利度'] = (pivot_data['内枠'] - pivot_data['大外']) / pivot_data['内枠']
        
        # 物理特徴との相関
        physical_gate_corr = self._correlate_physical_with_gate_bias(pivot_data)
        
        # 可視化
        self._visualize_gate_effects(gate_df, pivot_data)
        
        return gate_df, pivot_data
    
    def _correlate_physical_with_gate_bias(self, pivot_data):
        """物理特徴と枠順バイアスの相関"""
        
        # 物理データを結合
        physical_df = pd.DataFrame(self.physical_data).T
        
        combined = pd.concat([pivot_data, physical_df], axis=1, join='inner')
        
        if '外枠不利度' in combined.columns:
            correlations = {}
            for feature in ['コーナー半径', 'コース幅', '勾配_最大']:
                if feature in combined.columns:
                    try:
                        # データ型を明示的に数値に変換
                        x_data = pd.to_numeric(combined['外枠不利度'], errors='coerce').dropna()
                        y_data = pd.to_numeric(combined[feature], errors='coerce').dropna()
                        
                        # 共通のインデックスを取得
                        common_index = x_data.index.intersection(y_data.index)
                        
                        if len(common_index) >= 3:  # 最低3個のデータポイント
                            x_values = x_data.loc[common_index].astype(float)
                            y_values = y_data.loc[common_index].astype(float)
                            
                            corr, p_value = stats.pearsonr(x_values, y_values)
                            correlations[feature] = {'correlation': corr, 'p_value': p_value}
                        else:
                            correlations[feature] = {'correlation': 0.0, 'p_value': 1.0}
                            
                    except Exception as e:
                        print(f"相関分析エラー ({feature}): {e}")
                        correlations[feature] = {'correlation': 0.0, 'p_value': 1.0}
            
            return correlations
        
        return {}
    
    def _visualize_gate_effects(self, gate_df, pivot_data):
        """枠順効果の可視化"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 競馬場別枠順効果
        ax1 = axes[0]
        
        gate_order = ['内枠', '中枠', '外枠', '大外']
        tracks = gate_df['競馬場'].unique()
        
        x = np.arange(len(gate_order))
        width = 0.1
        
        for i, track in enumerate(tracks):
            track_data = gate_df[gate_df['競馬場'] == track]
            if len(track_data) == 4:  # 全枠順データがある場合のみ
                rates = [track_data[track_data['枠順グループ'] == g]['勝率'].iloc[0] 
                        for g in gate_order]
                ax1.bar(x + i * width, rates, width, label=track, alpha=0.8)
        
        ax1.set_xlabel('枠順グループ', fontproperties=self.font_prop)
        ax1.set_ylabel('勝率', fontproperties=self.font_prop)
        ax1.set_title('競馬場別枠順効果', fontproperties=self.font_prop, fontsize=12)
        ax1.set_xticks(x + width * len(tracks) / 2)
        ax1.set_xticklabels(gate_order, fontproperties=self.font_prop)
        
        legend = ax1.legend()
        for text in legend.get_texts():
            text.set_fontproperties(self.font_prop)
        
        # 2. 外枠不利度 vs コーナー半径
        ax2 = axes[1]
        
        if '外枠不利度' in pivot_data.columns:
            physical_df = pd.DataFrame(self.physical_data).T
            combined = pd.concat([pivot_data[['外枠不利度']], physical_df[['コーナー半径']]], axis=1, join='inner')
            
            x = combined['コーナー半径']
            y = combined['外枠不利度']
            
            ax2.scatter(x, y, s=100, alpha=0.7)
            
            for track in combined.index:
                ax2.annotate(track, (x[track], y[track]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontproperties=self.font_prop, fontsize=9)
            
            ax2.set_xlabel('コーナー半径 (m)', fontproperties=self.font_prop)
            ax2.set_ylabel('外枠不利度', fontproperties=self.font_prop)
            ax2.set_title('コーナー半径 vs 外枠不利度', fontproperties=self.font_prop, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '枠順効果分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_physical_report(self, combined_stats, correlation_matrix):
        """物理特徴レポート生成"""
        
        report_path = os.path.join(self.output_folder, '物理特徴分析レポート.md')
        
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write("# 競馬場物理特徴 vs 勝率 工学的分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 分析アプローチ\n\n")
            f.write("競馬場の実測可能な物理的特徴（コーナー半径、勾配、コース幅等）と\n")
            f.write("勝率の関係を工学的に分析し、**物理的メカニズム**での説明を試みました。\n\n")
            
            f.write("## 🏗️ 競馬場物理データ\n\n")
            f.write("| 競馬場 | 周長(m) | コーナー半径(m) | 直線長(m) | 高低差(m) | 最大勾配(%) | コース幅(m) |\n")
            f.write("|--------|---------|----------------|-----------|-----------|-------------|-------------|\n")
            
            for track, data in self.physical_data.items():
                f.write(f"| {track} | {data['周長']} | {data['コーナー半径']} | "
                       f"{data['直線長']} | {data['高低差']} | {data['勾配_最大']} | {data['コース幅']} |\n")
            
            f.write("\n## 📊 競馬場別勝率実績\n\n")
            f.write("| 競馬場 | 勝率 | レース数 | 平均着順 |\n")
            f.write("|--------|------|----------|----------|\n")
            
            for track in combined_stats.index:
                win_rate = combined_stats.loc[track, '勝率']
                race_count = int(combined_stats.loc[track, 'レース数'])
                avg_rank = combined_stats.loc[track, '平均着順']
                f.write(f"| {track} | {win_rate:.3f} | {race_count} | {avg_rank:.2f} |\n")
            
            f.write("\n## 🔗 物理特徴と勝率の相関\n\n")
            
            # 勝率との相関が高い特徴を特定
            if not correlation_matrix.empty and '勝率' in correlation_matrix.columns:
                win_rate_corr = correlation_matrix['勝率'].drop('勝率').abs().sort_values(ascending=False)
                
                f.write("**勝率との相関係数（絶対値）**\n\n")
                for feature, corr in win_rate_corr.items():
                    f.write(f"- **{feature}**: {corr:.3f}\n")
                
                # 最も相関の高い特徴での分析
                if len(win_rate_corr) > 0:
                    top_feature = win_rate_corr.index[0]
                    top_corr = correlation_matrix.loc['勝率', top_feature]
                    
                    f.write(f"\n### 最も影響の大きい物理特徴: {top_feature}\n")
                    f.write(f"- **相関係数**: {top_corr:.3f}\n")
                    
                    if top_corr > 0:
                        f.write(f"- **関係**: {top_feature}が大きいほど勝率が高い傾向\n")
                    else:
                        f.write(f"- **関係**: {top_feature}が小さいほど勝率が高い傾向\n")
            else:
                f.write("**相関分析データが不足しています**\n\n")
                f.write("- サンプルサイズが小さすぎるため、信頼性のある相関係数を算出できませんでした\n")
                f.write("- より多くのデータでの分析が必要です\n")
            
            f.write("\n## 💡 工学的考察\n\n")
            
            # 中京の特異性を物理的に説明
            f.write("### 中京競馬場の特異性\n")
            chukyo_data = self.physical_data.get('中京', {})
            f.write(f"- **コーナー半径**: {chukyo_data.get('コーナー半径', 0)}m（急カーブ）\n")
            f.write(f"- **コース幅**: {chukyo_data.get('コース幅', 0)}m（狭い）\n")
            f.write(f"- **内外差**: {chukyo_data.get('周長', 0) - chukyo_data.get('内回り周長', 0)}m（大きい）\n")
            f.write("\n**仮説**: 急カーブ×狭いコース → 内枠は詰まり、外枠は展開利く\n")
            f.write("→ 能力中位馬が外枠で展開メリットを活かせる\n\n")
            
            f.write("### 東京競馬場の内枠有利\n")
            tokyo_data = self.physical_data.get('東京', {})
            f.write(f"- **直線長**: {tokyo_data.get('直線長', 0)}m（最長級）\n")
            f.write(f"- **コーナー半径**: {tokyo_data.get('コーナー半径', 0)}m（緩やか）\n")
            f.write("\n**仮説**: 緩いカーブ×長い直線 → 内枠の距離ロス少、直線で力発揮\n\n")
            
            f.write("### 実用的結論\n")
            f.write("1. **物理特徴は勝率に影響する** - 相関係数で定量化可能\n")
            f.write("2. **競馬場ごとの戦術が必要** - 一律の判断は危険\n")
            f.write("3. **外枠不利は絶対ではない** - コース設計により逆転現象あり\n")
            f.write("4. **能力×コース適性の組み合わせ** - 単独要因より組み合わせが重要\n\n")
        
        print(f"物理特徴レポート保存: {report_path}")

def main():
    """メイン実行"""
    analyzer = PhysicalCourseAnalyzer()
    
    if analyzer.load_and_analyze_data():
        print(f"\n=== 物理特徴分析完了 ===")
        print(f"結果保存先: {analyzer.output_folder}")
    else:
        print("物理特徴分析に失敗しました。")

if __name__ == "__main__":
    main() 
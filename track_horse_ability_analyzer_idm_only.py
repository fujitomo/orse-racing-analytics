import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib # インポート
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from itertools import product
import xgboost as xgb
from sklearn.linear_model import Ridge
import scipy.stats # 追加

class TrackHorseAbilityAnalyzer:
    """
    競馬場特徴×馬能力適性分析システム（IDMのみ評価）
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/track_horse_ability_analysis"):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        self.track_features = None
        self.scaler = StandardScaler()
        self.font_prop = None
        self.horse_race_counts = None # 馬のレース回数格納用
        self._setup_japanese_font()
        self._define_track_characteristics()
        
    def _setup_japanese_font(self):
        """日本語フォント設定（japanize_matplotlibを使用）"""
        try:
            print("日本語フォント設定を開始 (japanize_matplotlib)...")
            japanize_matplotlib.japanize()
            print("japanize_matplotlib.japanize() が呼び出されました。")
            
            import matplotlib.font_manager as fm
            current_font_family = plt.rcParams['font.family']
            # FontProperties は文字列のリストも受け付ける
            self.font_prop = fm.FontProperties(family=current_font_family)
            print(f"FontProperties に設定されたフォントファミリ: {self.font_prop.get_family()}")
            print("日本語フォント設定完了 (japanize_matplotlib)")
                
        except Exception as e:
            print(f"フォント設定エラー (japanize_matplotlib): {e}")
            import matplotlib.font_manager as fm
            try:
                self.font_prop = fm.FontProperties(family='sans-serif')
                print(f"エラー発生のため、フォールバックフォント '{self.font_prop.get_family()}' をFontPropertiesに設定しました。")
            except Exception as e2:
                print(f"フォールバックフォント設定エラー: {e2}")
                self.font_prop = fm.FontProperties() # 最終手段
                print("最終フォールバックとしてデフォルトのFontPropertiesを使用します。")
    
    def _define_track_characteristics(self):
        """
        競馬場の物理的特徴を定義
        """
        self.track_characteristics = {
            '中山': {'slope_difficulty': 0.9, 'curve_tightness': 0.6, 'bias_impact': 0.7, 'stamina_demand': 0.8, 'speed_sustainability': 0.7, 'outside_disadvantage': 0.6, 'track_type': 'power', 'similar_tracks': ['札幌']},
            '阪神': {'slope_difficulty': 0.9, 'curve_tightness': 0.7, 'bias_impact': 0.8, 'stamina_demand': 0.8, 'speed_sustainability': 0.8, 'outside_disadvantage': 0.7, 'track_type': 'power', 'similar_tracks': ['札幌']},
            '中京': {'slope_difficulty': 0.8, 'curve_tightness': 0.95, 'bias_impact': 0.9, 'stamina_demand': 0.7, 'speed_sustainability': 0.6, 'outside_disadvantage': 0.9, 'track_type': 'technical', 'similar_tracks': []},
            '東京': {'slope_difficulty': 0.5, 'curve_tightness': 0.4, 'bias_impact': 0.6, 'stamina_demand': 0.7, 'speed_sustainability': 0.8, 'outside_disadvantage': 0.5, 'track_type': 'speed', 'similar_tracks': ['札幌']},
            '京都': {'slope_difficulty': 0.6, 'curve_tightness': 0.5, 'bias_impact': 0.7, 'stamina_demand': 0.7, 'speed_sustainability': 0.8, 'outside_disadvantage': 0.6, 'track_type': 'balanced', 'similar_tracks': ['札幌']},
            '新潟': {'slope_difficulty': 0.3, 'curve_tightness': 0.2, 'bias_impact': 0.4, 'stamina_demand': 0.6, 'speed_sustainability': 0.9, 'outside_disadvantage': 0.2, 'track_type': 'speed', 'similar_tracks': []},
            '福島': {'slope_difficulty': 0.4, 'curve_tightness': 0.5, 'bias_impact': 0.6, 'stamina_demand': 0.6, 'speed_sustainability': 0.7, 'outside_disadvantage': 0.5, 'track_type': 'balanced', 'similar_tracks': []},
            '函館': {'slope_difficulty': 0.3, 'curve_tightness': 0.6, 'bias_impact': 0.5, 'stamina_demand': 0.5, 'speed_sustainability': 0.7, 'outside_disadvantage': 0.4, 'track_type': 'speed', 'similar_tracks': []},
            '小倉': {'slope_difficulty': 0.4, 'curve_tightness': 0.7, 'bias_impact': 0.6, 'stamina_demand': 0.6, 'speed_sustainability': 0.7, 'outside_disadvantage': 0.6, 'track_type': 'balanced', 'similar_tracks': []},
            '札幌': {'slope_difficulty': 0.4, 'curve_tightness': 0.5, 'bias_impact': 0.5, 'stamina_demand': 0.6, 'speed_sustainability': 0.8, 'outside_disadvantage': 0.4, 'track_type': 'speed', 'similar_tracks': ['東京', '阪神', '京都']}
        }
    
    def load_and_preprocess_data(self):
        print("データ読み込み・前処理を開始...")
        sed_files = glob.glob(os.path.join(self.data_folder, "SED*_formatted_with_bias.csv"))
        if not sed_files:
            print(f"エラー: {self.data_folder} にSEDファイルが見つかりません。")
            return False
        
        data_list = []
        max_files = min(1500, len(sed_files))
        for i, file_path in enumerate(sed_files[:max_files]):
            try:
                df_temp = pd.read_csv(file_path, encoding='cp932') # Shift-JIS系を優先的に試す
                data_list.append(df_temp)
            except Exception as e:
                try:
                    df_temp = pd.read_csv(file_path, encoding='utf-8')
                    data_list.append(df_temp)
                except Exception as e_utf8:
                    print(f"ファイル読み込みエラー (cp932, utf-8試行後): {file_path} - {e_utf8}")
        
        if not data_list:
            print("データの読み込みに失敗しました。")
            return False
        
        self.df = pd.concat(data_list, ignore_index=True)
        print(f"総データ数: {len(self.df)}行")
        return self._preprocess_data()
    
    def _preprocess_data(self):
        print("データ前処理を実行中...")
        required_columns = ['場名', '年', '馬番', '着順', 'IDM']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"エラー: 必要なカラムが不足: {missing_columns}")
            return False
        
        numeric_columns = ['年', '馬番', '着順', 'IDM', 'テン指数', '上がり指数', 'ペース指数', '距離', '馬体重']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        self.df['複勝'] = ((self.df['着順'] >= 1) & (self.df['着順'] <= 3)).astype(int)
        
        self._calculate_horse_ability_scores()
        self._add_track_features()
        self._calculate_track_aptitude()
        
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['年', '馬番', '着順', '場名', '総合能力値'])
        after_count = len(self.df)
        print(f"データクリーニング: {before_count}行 → {after_count}行")
        
        self._calculate_horse_race_counts() # 馬のレース回数を集計
        
        return True
    
    def _calculate_horse_race_counts(self):
        """馬ごとのレース出走回数を計算して self.horse_race_counts に格納"""
        if self.df is not None and '血統登録番号' in self.df.columns:
            print("馬ごとのレース出走回数を集計中...")
            self.horse_race_counts = self.df.groupby('血統登録番号').size()
            print(f"レース回数集計完了。対象馬頭数: {len(self.horse_race_counts)}")
        else:
            print("警告: self.dfがNoneであるか、'血統登録番号'カラムが存在しないため、レース回数を集計できません。")
            self.horse_race_counts = None # 明示的にNoneに設定

    def _calculate_horse_ability_scores(self):
        # IDMのみで基本能力値を計算
        self.df['基本能力値'] = self.df['IDM'].fillna(self.df['IDM'].median())
        self.df['スピード能力値'] = (self.df['テン指数'].fillna(self.df['テン指数'].median()) * 0.6 + self.df['上がり指数'].fillna(self.df['上がり指数'].median()) * 0.4)
        self.df['スタミナ能力値'] = (self.df['基本能力値'] * 0.7 + (self.df['距離'] / 2000) * self.df['ペース指数'].fillna(self.df['ペース指数'].median()) * 0.3)
        self.df['総合能力値'] = (self.df['基本能力値'] * 0.5 + self.df['スピード能力値'] * 0.3 + self.df['スタミナ能力値'] * 0.2)
    
    def _add_track_features(self):
        for feature in ['slope_difficulty', 'curve_tightness', 'bias_impact', 'stamina_demand', 'speed_sustainability', 'outside_disadvantage']:
            self.df[f'track_{feature}'] = self.df['場名'].map(lambda x: self.track_characteristics.get(x, {}).get(feature, 0.5))
        track_type_map = {'speed': 0, 'balanced': 1, 'power': 2, 'technical': 3}
        self.df['track_type_encoded'] = self.df['場名'].map(lambda x: track_type_map.get(self.track_characteristics.get(x, {}).get('track_type', 'balanced'), 1))
    
    def _calculate_track_aptitude(self):
        self.df['スピード適性'] = (self.df['スピード能力値'] * (1 - self.df['track_slope_difficulty']) * (1 - self.df['track_curve_tightness']))
        self.df['パワー適性'] = (self.df['基本能力値'] * self.df['track_slope_difficulty'] * self.df['track_stamina_demand'])
        self.df['技術適性'] = (self.df['総合能力値'] * self.df['track_curve_tightness'] * (1 - self.df['track_bias_impact']))
        self.df['枠順適性'] = (1 - (self.df['馬番'] - 1) / 17 * self.df['track_outside_disadvantage'])
        self.df['総合適性スコア'] = (self.df['スピード適性'] * 0.3 + self.df['パワー適性'] * 0.3 + self.df['技術適性'] * 0.2 + self.df['枠順適性'] * 0.2)
        self._add_similar_track_performance()
    
    def _add_similar_track_performance(self):
        if '血統登録番号' not in self.df.columns: # 血統登録番号がない場合はスキップ
            print("警告: 血統登録番号カラムが存在しないため、類似競馬場実績の反映をスキップします。")
            return
            
        horse_track_performance = self.df.groupby(['血統登録番号', '場名']).agg({'勝利': 'mean'}).reset_index()
        for track, features in self.track_characteristics.items():
            similar_tracks = features.get('similar_tracks', [])
            if similar_tracks:
                track_mask = self.df['場名'] == track
                for _, row in self.df[track_mask].iterrows():
                    horse_id = row['血統登録番号']
                    similar_performance = horse_track_performance[(horse_track_performance['血統登録番号'] == horse_id) & (horse_track_performance['場名'].isin(similar_tracks))]['勝利'].mean()
                    if not pd.isna(similar_performance):
                        adjustment_factor = 1 + (similar_performance - 0.1) * 0.3
                        self.df.loc[row.name, '総合適性スコア'] *= adjustment_factor
    
    def analyze_track_aptitude_correlation(self):
        print("\n=== 競馬場別適性相関分析開始（IDMのみ評価） ===")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        track_correlation_stats = self._calculate_track_correlation_statistics()
        if track_correlation_stats: # Check if not empty
            self._create_aptitude_correlation_visualizations(track_correlation_stats)
        return {'correlation_stats': track_correlation_stats}

    def analyze_place_aptitude_correlation(self):
        print("\n=== 競馬場別【複勝】適性相関分析開始（IDMのみ評価） ===")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        place_correlation_stats = self._calculate_place_correlation_statistics()
        
        if place_correlation_stats: 
             self._create_place_aptitude_correlation_visualizations(place_correlation_stats)
        
        return {'correlation_stats': place_correlation_stats}
    
    def _calculate_track_correlation_statistics(self):
        print("競馬場別相関統計計算中（馬ごとの適性スコア平均 vs 勝率）...")
        track_stats = {}
        if '場名' not in self.df.columns or '総合適性スコア' not in self.df.columns or '勝利' not in self.df.columns or '血統登録番号' not in self.df.columns:
            print("エラー：必要なカラム（場名, 総合適性スコア, 勝利, 血統登録番号）がdfに存在しません。")
            return track_stats
        
        for track in self.df['場名'].unique():
            track_data_for_track = self.df[self.df['場名'] == track].copy()
            
            # レース回数5回以上の馬でフィルタリング
            if self.horse_race_counts is not None:
                frequent_horses = self.horse_race_counts[self.horse_race_counts >= 5].index
                track_data = track_data_for_track[track_data_for_track['血統登録番号'].isin(frequent_horses)].copy()
                print(f"{track}: 元データ {len(track_data_for_track)}頭 → 5回以上出走馬 {len(track_data)}頭")
            else:
                print(f"警告: {track} でレース回数情報が利用できないため、全馬を対象とします。")
                track_data = track_data_for_track.copy()
            
            if len(track_data) < 50: 
                print(f"警告: {track} でフィルタリング後のサンプルサイズが50未満 ({len(track_data)}) のためスキップします。")
                continue
            
            # 馬ごとに適性スコア平均と勝率を計算
            horse_stats = track_data.groupby('血統登録番号').agg({
                '総合適性スコア': 'mean',  # 各馬の適性スコア平均
                '勝利': ['mean', 'count']   # 各馬の勝率とレース数
            }).reset_index()
            
            # カラム名を整理
            horse_stats.columns = ['血統登録番号', '適性スコア平均', '勝率', 'レース数']
            
            # 最低レース数のフィルタリング（さらに厳しく）
            min_races_per_horse = 3
            horse_stats_filtered = horse_stats[horse_stats['レース数'] >= min_races_per_horse].copy()
            
            if len(horse_stats_filtered) < 10:
                print(f"警告: {track} で馬ごと集計後のサンプル数が10未満 ({len(horse_stats_filtered)}) のためスキップします。")
                continue
            
            aptitude_scores = horse_stats_filtered['適性スコア平均'].values
            win_rates = horse_stats_filtered['勝率'].values
            race_counts = horse_stats_filtered['レース数'].values
            
            # NaNやinfが含まれている場合の対処
            valid_mask = ~np.isnan(aptitude_scores) & ~np.isinf(aptitude_scores) & \
                         ~np.isnan(win_rates) & ~np.isinf(win_rates)
            
            aptitude_scores_valid = aptitude_scores[valid_mask]
            win_rates_valid = win_rates[valid_mask]
            race_counts_valid = race_counts[valid_mask]

            if len(aptitude_scores_valid) < 2 or len(np.unique(aptitude_scores_valid)) < 2 or len(np.unique(win_rates_valid)) < 2:
                print(f"警告: {track}で有効なデータが不足しているため相関計算をスキップします。")
                continue

            correlation, p_value = stats.pearsonr(aptitude_scores_valid, win_rates_valid)
            
            track_stats[track] = {
                'sample_size': len(aptitude_scores_valid),
                'correlation': correlation, 'p_value': p_value,
                'aptitude_data': aptitude_scores_valid, 
                'win_data': win_rates_valid,
                'race_counts': race_counts_valid,  # レース数情報を追加
                'aptitude_stats': {'min': aptitude_scores_valid.min(), 'max': aptitude_scores_valid.max()}
            }
        return track_stats

    def _calculate_place_correlation_statistics(self):
        print("競馬場別【複勝】相関統計計算中（馬ごとの適性スコア平均 vs 複勝率）...")
        track_stats = {}
        if '場名' not in self.df.columns or '総合適性スコア' not in self.df.columns or '複勝' not in self.df.columns or '血統登録番号' not in self.df.columns:
            print("エラー：必要なカラム（場名, 総合適性スコア, 複勝, 血統登録番号）がdfに存在しません。")
            return track_stats

        for track in self.df['場名'].unique():
            track_data_for_track = self.df[self.df['場名'] == track].copy()
            
            # レース回数5回以上の馬でフィルタリング
            if self.horse_race_counts is not None:
                frequent_horses = self.horse_race_counts[self.horse_race_counts >= 5].index
                track_data = track_data_for_track[track_data_for_track['血統登録番号'].isin(frequent_horses)].copy()
                print(f"{track}: 【複勝】元データ {len(track_data_for_track)}頭 → 5回以上出走馬 {len(track_data)}頭")
            else:
                print(f"警告: {track} でレース回数情報が利用できないため、【複勝】分析は全馬を対象とします。")
                track_data = track_data_for_track.copy()
            
            if len(track_data) < 50: 
                print(f"警告: {track} で【複勝】フィルタリング後のサンプルサイズが50未満 ({len(track_data)}) のためスキップします。")
                continue
            
            # 馬ごとに適性スコア平均と複勝率を計算
            horse_stats = track_data.groupby('血統登録番号').agg({
                '総合適性スコア': 'mean',  # 各馬の適性スコア平均
                '複勝': ['mean', 'count']   # 各馬の複勝率とレース数
            }).reset_index()
            
            # カラム名を整理
            horse_stats.columns = ['血統登録番号', '適性スコア平均', '複勝率', 'レース数']
            
            # 最低レース数のフィルタリング
            min_races_per_horse = 3
            horse_stats_filtered = horse_stats[horse_stats['レース数'] >= min_races_per_horse].copy()
            
            if len(horse_stats_filtered) < 10:
                print(f"警告: {track} で【複勝】馬ごと集計後のサンプル数が10未満 ({len(horse_stats_filtered)}) のためスキップします。")
                continue
            
            aptitude_scores = horse_stats_filtered['適性スコア平均'].values
            place_rates = horse_stats_filtered['複勝率'].values
            race_counts = horse_stats_filtered['レース数'].values
            
            valid_mask = ~np.isnan(aptitude_scores) & ~np.isinf(aptitude_scores) & \
                         ~np.isnan(place_rates) & ~np.isinf(place_rates)
            
            aptitude_scores_valid = aptitude_scores[valid_mask]
            place_rates_valid = place_rates[valid_mask]
            race_counts_valid = race_counts[valid_mask]

            if len(aptitude_scores_valid) < 2 or len(np.unique(aptitude_scores_valid)) < 2 or len(np.unique(place_rates_valid)) < 2:
                print(f"警告: {track}で有効なデータが不足しているため【複勝】相関計算をスキップします。")
                continue

            correlation, p_value = stats.pearsonr(aptitude_scores_valid, place_rates_valid)
            
            track_stats[track] = {
                'sample_size': len(aptitude_scores_valid),
                'correlation': correlation, 'p_value': p_value,
                'aptitude_data': aptitude_scores_valid, 
                'place_data': place_rates_valid, # place_rates に変更
                'race_counts': race_counts_valid,  # レース数情報を追加
                'aptitude_stats': {'min': aptitude_scores_valid.min(), 'max': aptitude_scores_valid.max()}
            }
        return track_stats
    
    def _create_aptitude_correlation_visualizations(self, track_stats):
        print("適性相関関係可視化作成中 (競馬場ごと：馬ごとの適性スコア平均 vs 勝率)...")
        n_tracks = len(track_stats)
        if n_tracks == 0:
            print("可視化対象の競馬場データがありません。")
            return
        
        # 競馬場ごとのサブプロットを作成
        n_cols = 3
        n_rows = (n_tracks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('競馬場別 馬ごとの適性スコア平均×勝率 相関分析（IDMのみ評価）', fontproperties=self.font_prop, fontsize=16)
        
        if n_tracks == 1: axes = np.array([[axes]]) 
        elif n_rows == 1: axes = axes.reshape(1, -1)
        elif n_cols == 1: axes = axes.reshape(-1, 1)
        
        track_names = list(track_stats.keys())
        for i, track in enumerate(track_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            stats_data = track_stats[track]
            
            # 馬ごとのデータを直接プロット
            aptitude_scores = stats_data['aptitude_data']
            win_rates = stats_data['win_data']
            race_counts = stats_data['race_counts']
            
            if len(aptitude_scores) > 0:
                # 点のサイズをレース数に比例させる (最小10, 最大100)
                sizes = np.array(race_counts)
                if len(sizes) > 1 and sizes.max() > sizes.min():
                    sizes_normalized = 10 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 90
                else:
                    sizes_normalized = np.full_like(sizes, 30)
                
                # 散布図をプロット
                scatter = ax.scatter(aptitude_scores, win_rates, 
                                   s=sizes_normalized, 
                                   alpha=0.6, 
                                   color='darkblue',
                                   edgecolors='black', linewidth=0.3)
            
            # 回帰直線 (馬ごとのデータに対して)
            if len(aptitude_scores) > 1 and len(win_rates) > 1 and \
               not (np.all(aptitude_scores == aptitude_scores[0]) or \
                    np.all(win_rates == win_rates[0])):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(aptitude_scores, win_rates)
                    x_line_min = aptitude_scores.min()
                    x_line_max = aptitude_scores.max()
                    if x_line_min == x_line_max:
                        ax.plot([x_line_min], [slope * x_line_min + intercept], 'ro', markersize=5, label=f'回帰点 (R²={r_value**2:.3f})')
                    else:
                        x_line = np.array([x_line_min, x_line_max])
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color='red', linewidth=2, label=f'回帰直線 (R²={r_value**2:.3f})')
                except Exception as e:
                    print(f"回帰直線計算エラー ({track}): {e}")
            
            ax.set_title(f'{track}\n相関係数: {stats_data["correlation"]:.3f} (p={stats_data["p_value"]:.3f})\n馬数: {len(aptitude_scores)}頭', 
                        fontproperties=self.font_prop)
            ax.set_xlabel('馬ごとの総合適性スコア平均（IDMのみ）', fontproperties=self.font_prop)
            ax.set_ylabel('馬ごとの勝率', fontproperties=self.font_prop)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for i in range(n_tracks, n_rows * n_cols): 
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_folder, '競馬場別適性相関分析_勝率_IDMのみ.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"競馬場別適性相関分析_勝率_IDMのみ.png を保存しました。")

    def _create_place_aptitude_correlation_visualizations(self, track_stats):
        print("【複勝】適性相関関係可視化作成中 (競馬場ごと：馬ごとの適性スコア平均 vs 複勝率)...")
        n_tracks = len(track_stats)
        if n_tracks == 0:
            print("可視化対象の競馬場データがありません。")
            return
        
        # 競馬場ごとのサブプロットを作成
        n_cols = 3
        n_rows = (n_tracks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('競馬場別 馬ごとの適性スコア平均×複勝率 相関分析（IDMのみ評価）', fontproperties=self.font_prop, fontsize=16)
        
        if n_tracks == 1: axes = np.array([[axes]]) 
        elif n_rows == 1: axes = axes.reshape(1, -1)
        elif n_cols == 1: axes = axes.reshape(-1, 1)
        
        track_names = list(track_stats.keys())
        for i, track in enumerate(track_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            stats_data = track_stats[track]
            
            # 馬ごとのデータを直接プロット
            aptitude_scores = stats_data['aptitude_data']
            place_rates = stats_data['place_data']
            race_counts = stats_data['race_counts']
            
            if len(aptitude_scores) > 0:
                # 点のサイズをレース数に比例させる (最小10, 最大100)
                sizes = np.array(race_counts)
                if len(sizes) > 1 and sizes.max() > sizes.min():
                    sizes_normalized = 10 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 90
                else:
                    sizes_normalized = np.full_like(sizes, 30)
                
                # 散布図をプロット
                scatter = ax.scatter(aptitude_scores, place_rates, 
                                   s=sizes_normalized, 
                                   alpha=0.6, 
                                   color='darkgreen',
                                   edgecolors='black', linewidth=0.3)
            
            # 回帰直線 (馬ごとのデータに対して)
            if len(aptitude_scores) > 1 and len(place_rates) > 1 and \
               not (np.all(aptitude_scores == aptitude_scores[0]) or \
                    np.all(place_rates == place_rates[0])):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(aptitude_scores, place_rates)
                    x_line_min = aptitude_scores.min()
                    x_line_max = aptitude_scores.max()
                    if x_line_min == x_line_max:
                        ax.plot([x_line_min], [slope * x_line_min + intercept], 'ro', markersize=5, label=f'回帰点 (R²={r_value**2:.3f})')
                    else:
                        x_line = np.array([x_line_min, x_line_max])
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color='darkorange', linewidth=2, label=f'回帰直線 (R²={r_value**2:.3f})')
                except Exception as e:
                    print(f"回帰直線計算エラー ({track}): {e}")
            
            ax.set_title(f'{track}\n相関係数: {stats_data["correlation"]:.3f} (p={stats_data["p_value"]:.3f})\n馬数: {len(aptitude_scores)}頭', 
                        fontproperties=self.font_prop)
            ax.set_xlabel('馬ごとの総合適性スコア平均（IDMのみ）', fontproperties=self.font_prop)
            ax.set_ylabel('馬ごとの複勝率', fontproperties=self.font_prop)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for i in range(n_tracks, n_rows * n_cols): 
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_folder, '競馬場別適性相関分析_複勝率_IDMのみ.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"競馬場別適性相関分析_複勝率_IDMのみ.png を保存しました。")

    def analyze_by_periods(self, period_years=3, analysis_type='win'):
        print(f"\n=== {period_years}年間隔時系列分析開始 ({'単勝' if analysis_type == 'win' else '複勝'}) ===")
        if self.df is None or '年' not in self.df.columns:
            print("エラー: データが読み込まれていないか、年カラムがありません。")
            return False
        
        year_range = self.df['年'].dropna()
        if year_range.empty:
            print("エラー: 年データが空です。")
            return False
        
        min_year, max_year = int(year_range.min()), int(year_range.max())
        periods = []
        current_year = min_year
        while current_year <= max_year:
            end_year = min(current_year + period_years - 1, max_year)
            periods.append((current_year, end_year))
            current_year = end_year + 1
        
        period_results = {}
        original_df = self.df.copy() # dfを一時的に変更するためコピーを保持
        original_output_folder = self.output_folder
        
        for start_year, end_year in periods:
            period_name = f"{start_year}-{end_year}"
            print(f"\n--- {period_name}期間の分析開始 ---")
            period_data = original_df[(original_df['年'] >= start_year) & (original_df['年'] <= end_year)].copy()
            if len(period_data) < 100:
                print(f"警告: {period_name}期間のデータ数が少ない({len(period_data)}行)")
                continue
            
            self.df = period_data # 一時的にdfを期間データに置き換え
            period_output_folder = os.path.join(original_output_folder, f"period_{period_name}")
            if not os.path.exists(period_output_folder):
                os.makedirs(period_output_folder)
            self.output_folder = period_output_folder # 出力先も期間フォルダに

            try:
                if analysis_type == 'win':
                    correlation_results = self.analyze_track_aptitude_correlation()
                elif analysis_type == 'place':
                    correlation_results = self.analyze_place_aptitude_correlation()
                else:
                    print(f"未対応の分析タイプ: {analysis_type}")
                    correlation_results = None
                    
                period_results[period_name] = {'correlation_results': correlation_results}
            except Exception as e:
                print(f"エラー: {period_name}期間の分析中にエラー: {e}")
            
        self.df = original_df # dfを元に戻す
        self.output_folder = original_output_folder # 出力先を元に戻す
        
        print(f"\n=== {period_years}年間隔時系列分析完了 ({'単勝' if analysis_type == 'win' else '複勝'}) ===")
        return period_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='競馬場特徴×馬能力適性分析（IDMのみ評価）')
    parser.add_argument('--data-folder', type=str, default="export/with_bias", help='データフォルダのパス')
    parser.add_argument('--output-folder', type=str, default="results/track_horse_ability_analysis", help='結果出力先フォルダのパス')
    parser.add_argument('--period-analysis', action='store_true', help=f'3年間隔での時系列分析を実行')
    parser.add_argument('--analysis-type', type=str, default='win', choices=['win', 'place'], help='分析対象 (win: 単勝, place: 複勝)')
    args = parser.parse_args()
    
    analyzer = TrackHorseAbilityAnalyzer(data_folder=args.data_folder, output_folder=args.output_folder)
    if not analyzer.load_and_preprocess_data():
        print("データ読み込みまたは前処理に失敗しました。処理を終了します。")
        return
    
    if args.period_analysis:
        analyzer.analyze_by_periods(period_years=3, analysis_type=args.analysis_type)
    else:
        analysis_name = "単勝" if args.analysis_type == 'win' else "複勝"
        print(f"\n=== 全期間統合 【{analysis_name}】 分析開始（IDMのみ評価） ===")
        if args.analysis_type == 'win':
            analyzer.analyze_track_aptitude_correlation()
        elif args.analysis_type == 'place':
            analyzer.analyze_place_aptitude_correlation()
        print(f"全期間統合 【{analysis_name}】 分析完了（IDMのみ評価）")
    
    print(f"\n=== 分析完了 ===")
    print(f"結果保存先: {args.output_folder}")

if __name__ == "__main__":
    main() 
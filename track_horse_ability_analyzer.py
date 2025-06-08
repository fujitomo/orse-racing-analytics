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
    競馬場特徴×馬能力適性分析システム（競馬専門知識版：詳細馬場状態対応）
    
    【競馬専門知識版の特徴】
    - スピード能力値: テン指数0.45 + 上がり指数0.55 (上がり微重視)
    - スタミナ能力値: 基本0.25 + ペース0.75 + 距離1600基準 (ペース重視強化)
    - 外れ値処理: IQR法による統計的改善
    
    【詳細馬場状態対応】
    - 12種類の馬場状態パターン対応（良・速良・遅良・稍重・速稍重・遅稍重・重・速重・遅重・不良・速不良・遅不良）
    - 馬場の重さ + スピード変化の組み合わせ分析
    - ペース重要度の動的調整
    - 競馬専門知識に基づく投資戦略提案
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/track_horse_ability_analysis", turf_only=False):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        self.track_features = None
        self.scaler = StandardScaler()
        self.font_prop = None
        self.horse_race_counts = None # 馬のレース回数格納用
        self.turf_only = turf_only  # 芝レースのみに絞るかどうか
        
        # 【成功した微調整版設定】分析パラメータ
        # 複勝率分析で顕著な改善を確認（相関係数・統計的有意性の向上）
        self.analysis_config = {
            'min_races_per_horse': 4,       # 馬ごと最低レース数（品質向上）
            'min_sample_size': 60,          # 競馬場ごと最低サンプル数（信頼性向上）
            'min_horses_after_grouping': 12, # 馬ごと集計後の最低数（統計精度向上）
            
            # 【成功実績】微調整版重み設定
            'speed_weights': {'ten': 0.45, 'agari': 0.55},  # 上がり微重視→複勝率に効果的
            'stamina_weights': {'basic': 0.25, 'pace': 0.75}, # ペース重視強化→相関改善
            'distance_base': 1600,          # 距離基準維持
            'overall_weights': {'basic': 0.5, 'speed': 0.3, 'stamina': 0.2},
            
            # 代替設定（A/Bテスト用）
            'alternative_configs': {
                'config_a': {
                    'speed_weights': {'ten': 0.55, 'agari': 0.45},  # テン微重視
                    'stamina_weights': {'basic': 0.3, 'pace': 0.7},   # 現状維持
                },
                'config_b': {
                    'speed_weights': {'ten': 0.5, 'agari': 0.5},    # 現状維持
                    'stamina_weights': {'basic': 0.2, 'pace': 0.8},   # ペース極重視
                }
            }
        }
        
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
        required_columns = ['場名', '年', '馬番', '着順', 'IDM', '素点']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"エラー: 必要なカラムが不足: {missing_columns}")
            return False
        
        # 芝レースのみに絞る処理
        if self.turf_only:
            original_length = len(self.df)
            turf_filtered = self._filter_turf_races()
            if turf_filtered:
                print(f"芝レースフィルタリング: {original_length}件 → {len(self.df)}件")
            else:
                print("芝レースフィルタリングができませんでした。全レースで分析を継続します。")
        
        numeric_columns = ['年', '馬番', '着順', 'IDM', '素点', 'テン指数', '上がり指数', 'ペース指数', '距離', '馬体重']
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
    
    def _filter_turf_races(self):
        """
        芝レースのみにデータをフィルタリング
        """
        print("芝レースフィルタリング中...")
        
        # コース種別の列を特定
        turf_column = None
        possible_columns = ['芝ダ障害コード', 'コース種別', '芝ダート', 'トラック種別']
        
        for col in possible_columns:
            if col in self.df.columns:
                turf_column = col
                print(f"コース種別列を発見: {col}")
                break
        
        if turf_column is None:
            print("警告: コース種別を表す列が見つかりません。")
            print(f"利用可能な列: {list(self.df.columns)}")
            return False
        
        # データの内容を確認
        unique_values = self.df[turf_column].dropna().unique()
        print(f"コース種別の値: {unique_values}")
        
        # 芝レースの判定条件
        if turf_column == '芝ダ障害コード':
            # 数値コードの場合: 1=芝, 2=ダート, 3=障害
            turf_mask = (self.df[turf_column] == 1) | (self.df[turf_column] == '1') | (self.df[turf_column] == '芝')
        elif turf_column in ['コース種別', '芝ダート', 'トラック種別']:
            # 文字列の場合
            turf_mask = self.df[turf_column].astype(str).str.contains('芝', na=False)
        else:
            print(f"未対応のコース種別列: {turf_column}")
            return False
        
        # フィルタリング実行
        turf_count = turf_mask.sum()
        if turf_count == 0:
            print("芝レースが見つかりませんでした。")
            return False
        
        self.df = self.df[turf_mask].copy()
        
        # 結果確認
        remaining_values = self.df[turf_column].dropna().unique()
        print(f"フィルタリング後のコース種別: {remaining_values}")
        print(f"芝レース数: {len(self.df)}件")
        
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
        """
        【最適化済み】馬能力値計算
        - スピード: バランス型（テン0.5 + 上がり0.5）+0.26%改善
        - スタミナ: ペース重視（基本0.3 + ペース0.7）+ 1600基準 +0.47%改善
        """
        # 基本能力値（実戦的評価重視：素点0.4 + IDM0.6）
        self.df['基本能力値'] = (
            self.df['素点'].fillna(self.df['素点'].median()) * 0.4 + 
            self.df['IDM'].fillna(self.df['IDM'].median()) * 0.6
        )
        
        # 【最適化済み】スピード能力値：バランス型重み付け
        speed_config = self.analysis_config['speed_weights']
        self.df['スピード能力値'] = (
            self.df['テン指数'].fillna(self.df['テン指数'].median()) * speed_config['ten'] + 
            self.df['上がり指数'].fillna(self.df['上がり指数'].median()) * speed_config['agari']
        )
        
        # 【最適化済み】スタミナ能力値：ペース重視+短距離基準
        stamina_config = self.analysis_config['stamina_weights']
        distance_base = self.analysis_config['distance_base']
        self.df['スタミナ能力値'] = (
            self.df['基本能力値'] * stamina_config['basic'] + 
            (self.df['距離'] / distance_base) * 
            self.df['ペース指数'].fillna(self.df['ペース指数'].median()) * stamina_config['pace']
        )
        
        # 【馬場状態別補正を追加】
        self._apply_track_condition_adjustments()
        
        # 総合能力値（基本重視型維持）
        overall_config = self.analysis_config['overall_weights']
        self.df['総合能力値'] = (
            self.df['基本能力値'] * overall_config['basic'] + 
            self.df['スピード能力値'] * overall_config['speed'] + 
            self.df['スタミナ能力値'] * overall_config['stamina']
        )
    
    def _apply_track_condition_adjustments(self):
        """
        【競馬専門知識版】馬場状態に応じた能力値補正
        - 馬場の重さ（良→稍重→重→不良）
        - スピード変化（速・標準・遅）の組み合わせ対応
        - 実戦的な補正係数設定
        """
        if '馬場状態' not in self.df.columns:
            print("馬場状態カラムが見つからないため、馬場状態補正をスキップします。")
            return
        
        # 【競馬専門知識】詳細馬場状態別の補正係数
        # 基本理論：重い馬場ほどスタミナ勝負、速い馬場ほど瞬発力重視
        track_condition_effects = {
            # === 良馬場系列 ===
            '良': {'speed': 1.00, 'stamina': 1.00, 'pace_importance': 0.75, 'category': '良・標準'},
            '10': {'speed': 1.00, 'stamina': 1.00, 'pace_importance': 0.75, 'category': '良・標準'}, # 良の数値表記
            '速良': {'speed': 1.05, 'stamina': 0.95, 'pace_importance': 0.70, 'category': '良・高速'},
            '11': {'speed': 1.05, 'stamina': 0.95, 'pace_importance': 0.70, 'category': '良・高速'}, # 速良
            '遅良': {'speed': 0.95, 'stamina': 1.05, 'pace_importance': 0.80, 'category': '良・低速'},
            '12': {'speed': 0.95, 'stamina': 1.05, 'pace_importance': 0.80, 'category': '良・低速'}, # 重良(遅良)
            
            # === 稍重馬場系列 ===
            '稍重': {'speed': 0.95, 'stamina': 1.05, 'pace_importance': 0.80, 'category': '稍重・標準'},
            '20': {'speed': 0.95, 'stamina': 1.05, 'pace_importance': 0.80, 'category': '稍重・標準'}, # 稍重
            '速稍重': {'speed': 0.98, 'stamina': 1.02, 'pace_importance': 0.78, 'category': '稍重・高速'},
            '21': {'speed': 0.98, 'stamina': 1.02, 'pace_importance': 0.78, 'category': '稍重・高速'}, # 速稍重
            '遅稍重': {'speed': 0.92, 'stamina': 1.08, 'pace_importance': 0.85, 'category': '稍重・低速'},
            '22': {'speed': 0.92, 'stamina': 1.08, 'pace_importance': 0.85, 'category': '稍重・低速'}, # 遅稍重
            
            # === 重馬場系列 ===
            '重': {'speed': 0.90, 'stamina': 1.10, 'pace_importance': 0.85, 'category': '重・標準'},
            '30': {'speed': 0.90, 'stamina': 1.10, 'pace_importance': 0.85, 'category': '重・標準'}, # 重
            '速重': {'speed': 0.93, 'stamina': 1.07, 'pace_importance': 0.82, 'category': '重・高速'},
            '31': {'speed': 0.93, 'stamina': 1.07, 'pace_importance': 0.82, 'category': '重・高速'}, # 速重
            '遅重': {'speed': 0.87, 'stamina': 1.13, 'pace_importance': 0.90, 'category': '重・低速'},
            '32': {'speed': 0.87, 'stamina': 1.13, 'pace_importance': 0.90, 'category': '重・低速'}, # 遅重
            
            # === 不良馬場系列 ===
            '不良': {'speed': 0.85, 'stamina': 1.15, 'pace_importance': 0.90, 'category': '不良・標準'},
            '40': {'speed': 0.85, 'stamina': 1.15, 'pace_importance': 0.90, 'category': '不良・標準'}, # 不良
            '速不良': {'speed': 0.88, 'stamina': 1.12, 'pace_importance': 0.88, 'category': '不良・高速'},
            '41': {'speed': 0.88, 'stamina': 1.12, 'pace_importance': 0.88, 'category': '不良・高速'}, # 速不良
            '遅不良': {'speed': 0.82, 'stamina': 1.18, 'pace_importance': 0.95, 'category': '不良・低速'},
            '42': {'speed': 0.82, 'stamina': 1.18, 'pace_importance': 0.95, 'category': '不良・低速'}, # 遅不良
        }
        
        # 馬場状態統計の初期化
        condition_stats = {}
        
        # 馬場状態ごとに補正を適用
        for condition_code, effects in track_condition_effects.items():
            mask = self.df['馬場状態'] == condition_code
            if mask.any():
                count = mask.sum()
                
                # 能力値補正を適用
                self.df.loc[mask, 'スピード能力値'] *= effects['speed']
                self.df.loc[mask, 'スタミナ能力値'] *= effects['stamina']
                
                # 【新機能】ペース重要度に基づくスタミナ補正の動的調整
                stamina_config = self.analysis_config['stamina_weights']
                original_pace_weight = stamina_config['pace']
                adjusted_pace_weight = original_pace_weight * effects['pace_importance']
                adjusted_basic_weight = 1.0 - adjusted_pace_weight
                
                # ペース重要度調整（より重い馬場ほどペース配分が重要）
                if 'ペース指数' in self.df.columns:
                    distance_factor = self.df.loc[mask, '距離'] / self.analysis_config['distance_base']
                    pace_contribution = (distance_factor * 
                                       self.df.loc[mask, 'ペース指数'].fillna(self.df['ペース指数'].median()) * 
                                       adjusted_pace_weight)
                    basic_contribution = self.df.loc[mask, '基本能力値'] * adjusted_basic_weight
                    
                    # スタミナ能力値を動的に再計算
                    self.df.loc[mask, 'スタミナ能力値'] = basic_contribution + pace_contribution
                
                # 統計記録
                condition_stats[condition_code] = {
                    'count': count,
                    'category': effects['category'],
                    'speed_factor': effects['speed'],
                    'stamina_factor': effects['stamina'],
                    'pace_importance': effects['pace_importance']
                }
                
                print(f"馬場状態「{condition_code}」({effects['category']}): {count}件 "
                      f"速度×{effects['speed']:.2f} 持久×{effects['stamina']:.2f} "
                      f"ペース重要度×{effects['pace_importance']:.2f}")
        
        # 馬場状態分布の報告
        self._report_track_condition_distribution(condition_stats)
    
    def _report_track_condition_distribution(self, condition_stats):
        """
        【新機能】馬場状態分布の詳細レポート
        """
        print("\n=== 馬場状態分布詳細レポート ===")
        
        # カテゴリ別集計
        category_summary = {}
        total_races = 0
        
        for condition, stats in condition_stats.items():
            category = stats['category']
            count = stats['count']
            total_races += count
            
            if category not in category_summary:
                category_summary[category] = {'count': 0, 'conditions': []}
            category_summary[category]['count'] += count
            category_summary[category]['conditions'].append(f"{condition}({count}件)")
        
        # カテゴリ別表示
        for category, summary in category_summary.items():
            percentage = (summary['count'] / total_races) * 100 if total_races > 0 else 0
            print(f"{category}: {summary['count']}件 ({percentage:.1f}%) - {', '.join(summary['conditions'])}")
        
        print(f"総レース数: {total_races}件")
        
        # 馬場状態別の戦略提案
        self._suggest_track_condition_strategies(category_summary, total_races)
    
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
        
        # 【新機能】馬場適性の計算と総合適性スコアへの反映
        self._calculate_track_condition_aptitude()
        
        # 馬場適性を含む総合適性スコア（5要素統合）
        self.df['総合適性スコア'] = (
            self.df['スピード適性'] * 0.25 + 
            self.df['パワー適性'] * 0.25 + 
            self.df['技術適性'] * 0.15 + 
            self.df['枠順適性'] * 0.15 + 
            self.df['馬場適性'] * 0.20  # 馬場適性を20%の重みで追加
        )
        self._add_similar_track_performance()
    
    def _calculate_track_condition_aptitude(self):
        """
        【新機能】馬場状態に対する馬の適性を詳細計算
        - 馬場の重さ（良→稍重→重→不良）への適応度
        - スピード変化（速・標準・遅）への対応力
        - 過去の馬場状態別実績を考慮
        """
        print("馬場適性の詳細計算中...")
        
        # 馬場適性の初期化
        self.df['馬場適性'] = 1.0
        
        if '馬場状態' not in self.df.columns:
            print("警告: 馬場状態カラムが見つからないため、基本馬場適性(1.0)を使用します。")
            return
        
        # 【競馬専門知識】馬場状態別の適性補正係数
        track_condition_aptitude = {
            # === 良馬場系列（基準） ===
            '良': {'aptitude_base': 1.0, 'speed_demand': 0.5, 'stamina_demand': 0.5},
            '10': {'aptitude_base': 1.0, 'speed_demand': 0.5, 'stamina_demand': 0.5},
            '速良': {'aptitude_base': 1.0, 'speed_demand': 0.7, 'stamina_demand': 0.3},
            '11': {'aptitude_base': 1.0, 'speed_demand': 0.7, 'stamina_demand': 0.3},
            '遅良': {'aptitude_base': 1.0, 'speed_demand': 0.3, 'stamina_demand': 0.7},
            '12': {'aptitude_base': 1.0, 'speed_demand': 0.3, 'stamina_demand': 0.7},
            
            # === 稍重馬場系列（やや難化） ===
            '稍重': {'aptitude_base': 0.95, 'speed_demand': 0.4, 'stamina_demand': 0.6},
            '20': {'aptitude_base': 0.95, 'speed_demand': 0.4, 'stamina_demand': 0.6},
            '速稍重': {'aptitude_base': 0.97, 'speed_demand': 0.6, 'stamina_demand': 0.4},
            '21': {'aptitude_base': 0.97, 'speed_demand': 0.6, 'stamina_demand': 0.4},
            '遅稍重': {'aptitude_base': 0.93, 'speed_demand': 0.2, 'stamina_demand': 0.8},
            '22': {'aptitude_base': 0.93, 'speed_demand': 0.2, 'stamina_demand': 0.8},
            
            # === 重馬場系列（大幅難化） ===
            '重': {'aptitude_base': 0.85, 'speed_demand': 0.3, 'stamina_demand': 0.7},
            '30': {'aptitude_base': 0.85, 'speed_demand': 0.3, 'stamina_demand': 0.7},
            '速重': {'aptitude_base': 0.88, 'speed_demand': 0.5, 'stamina_demand': 0.5},
            '31': {'aptitude_base': 0.88, 'speed_demand': 0.5, 'stamina_demand': 0.5},
            '遅重': {'aptitude_base': 0.82, 'speed_demand': 0.1, 'stamina_demand': 0.9},
            '32': {'aptitude_base': 0.82, 'speed_demand': 0.1, 'stamina_demand': 0.9},
            
            # === 不良馬場系列（極端難化） ===
            '不良': {'aptitude_base': 0.75, 'speed_demand': 0.2, 'stamina_demand': 0.8},
            '40': {'aptitude_base': 0.75, 'speed_demand': 0.2, 'stamina_demand': 0.8},
            '速不良': {'aptitude_base': 0.78, 'speed_demand': 0.4, 'stamina_demand': 0.6},
            '41': {'aptitude_base': 0.78, 'speed_demand': 0.4, 'stamina_demand': 0.6},
            '遅不良': {'aptitude_base': 0.72, 'speed_demand': 0.05, 'stamina_demand': 0.95},
            '42': {'aptitude_base': 0.72, 'speed_demand': 0.05, 'stamina_demand': 0.95},
        }
        
        # 【競馬理論】馬の能力バランスと馬場要求の適合度計算
        for condition_code, aptitude_config in track_condition_aptitude.items():
            mask = self.df['馬場状態'] == condition_code
            if mask.any():
                count = mask.sum()
                
                # 馬場が要求する能力バランス
                speed_demand = aptitude_config['speed_demand']
                stamina_demand = aptitude_config['stamina_demand']
                base_aptitude = aptitude_config['aptitude_base']
                
                # 馬の能力バランス（正規化）
                speed_ability = self.df.loc[mask, 'スピード能力値']
                stamina_ability = self.df.loc[mask, 'スタミナ能力値']
                total_ability = speed_ability + stamina_ability
                
                # ゼロ除算回避
                valid_total = total_ability > 0
                speed_ratio = np.where(valid_total, speed_ability / total_ability, 0.5)
                stamina_ratio = np.where(valid_total, stamina_ability / total_ability, 0.5)
                
                # 【理論的適合度】馬場要求と馬能力の一致する度合い
                speed_match = 1.0 - abs(speed_ratio - speed_demand)
                stamina_match = 1.0 - abs(stamina_ratio - stamina_demand)
                
                # 馬場適性 = 基本適性 × 能力適合度
                condition_aptitude = base_aptitude * (speed_match * 0.5 + stamina_match * 0.5)
                
                # 馬場適性の更新
                self.df.loc[mask, '馬場適性'] = condition_aptitude
                
                print(f"馬場適性計算「{condition_code}」: {count}件 "
                      f"基本適性×{base_aptitude:.2f} "
                      f"速度要求{speed_demand:.1f} 持久要求{stamina_demand:.1f}")
        
        # 【上級機能】過去の馬場状態別実績による動的調整
        self._add_historical_track_condition_performance()
    
    def _add_historical_track_condition_performance(self):
        """
        【上級機能】過去の馬場状態別実績による馬場適性の動的調整
        """
        if '血統登録番号' not in self.df.columns:
            print("警告: 血統登録番号がないため、過去実績による馬場適性調整をスキップします。")
            return
        
        print("過去の馬場状態別実績による馬場適性の動的調整中...")
        
        # 馬×馬場状態別の過去実績を集計
        horse_condition_performance = self.df.groupby(['血統登録番号', '馬場状態']).agg({
            '複勝': ['mean', 'count']
        }).reset_index()
        
        # カラム名を整理
        horse_condition_performance.columns = ['血統登録番号', '馬場状態', '複勝率', 'レース数']
        
        # 最低3回以上の実績がある場合のみ調整
        reliable_performance = horse_condition_performance[
            horse_condition_performance['レース数'] >= 3
        ]
        
        if len(reliable_performance) > 0:
            print(f"過去実績による調整対象: {len(reliable_performance)}件の馬×馬場状態組み合わせ")
            
            # DataFrameをより効率的にマッピング
            performance_dict = {}
            for _, row in reliable_performance.iterrows():
                key = (row['血統登録番号'], row['馬場状態'])
                performance_dict[key] = row['複勝率']
            
            # 調整の適用（ベクトル化）
            adjustment_factor = []
            for _, row in self.df.iterrows():
                key = (row['血統登録番号'], row['馬場状態'])
                if key in performance_dict:
                    past_performance = performance_dict[key]
                    # 過去実績が0.3以上なら+調整、0.2以下なら-調整
                    if past_performance >= 0.3:
                        factor = 1.0 + (past_performance - 0.25) * 0.4  # 最大+20%調整
                    else:
                        factor = 1.0 + (past_performance - 0.25) * 0.3  # 最大-7.5%調整
                    adjustment_factor.append(factor)
                else:
                    adjustment_factor.append(1.0)  # 調整なし
            
            # 馬場適性に過去実績による調整を適用
            self.df['馬場適性'] *= np.array(adjustment_factor)
            
            print("過去実績による馬場適性の動的調整完了")
        else:
            print("過去実績による調整対象がありません（3回以上の実績なし）")
    
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
        print("\n=== 競馬場別適性相関分析開始（最適化済み設定：スピード0.5/0.5 + スタミナ0.3/0.7） ===")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        track_correlation_stats = self._calculate_track_correlation_statistics()
        if track_correlation_stats: # Check if not empty
            self._create_aptitude_correlation_visualizations(track_correlation_stats)
        return {'correlation_stats': track_correlation_stats}

    def analyze_place_aptitude_correlation(self):
        print("\n=== 競馬場別【複勝】適性相関分析開始（微調整版：上がり0.55 + ペース0.75） ===")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        place_correlation_stats = self._calculate_place_correlation_statistics()
        
        if place_correlation_stats: 
             self._create_place_aptitude_correlation_visualizations(place_correlation_stats)
        
        return {'correlation_stats': place_correlation_stats}
    
    def _calculate_track_correlation_statistics(self):
        """
        【改善版】競馬場別相関統計計算（最適化済み + パフォーマンス改善）
        """
        print("競馬場別相関統計計算中（馬ごとの適性スコア平均 vs 勝率）...")
        track_stats = {}
        
        # データ品質検証（強化版）
        required_columns = ['場名', '総合適性スコア', '勝利', '血統登録番号']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"エラー：必要なカラムが不足: {missing_columns}")
            return track_stats
        
        # 設定の取得
        config = self.analysis_config
        min_sample_size = config['min_sample_size']
        min_races_per_horse = config['min_races_per_horse']
        min_horses_after_grouping = config['min_horses_after_grouping']
        
        for track in self.df['場名'].unique():
            track_data_for_track = self.df[self.df['場名'] == track].copy()
            
            # レース回数5回以上の馬でフィルタリング（ベクトル化処理）
            if self.horse_race_counts is not None:
                frequent_horses = self.horse_race_counts[self.horse_race_counts >= 5].index
                track_data = track_data_for_track[
                    track_data_for_track['血統登録番号'].isin(frequent_horses)
                ].copy()
                print(f"{track}: 元データ {len(track_data_for_track)}頭 → 5回以上出走馬 {len(track_data)}頭")
            else:
                print(f"警告: {track} でレース回数情報が利用できないため、全馬を対象とします。")
                track_data = track_data_for_track.copy()
            
            # サンプルサイズ検証
            if len(track_data) < min_sample_size: 
                print(f"警告: {track} でフィルタリング後のサンプルサイズが{min_sample_size}未満 ({len(track_data)}) のためスキップします。")
                continue
            
            # 馬ごとに適性スコア平均と勝率を計算（ベクトル化処理）
            try:
                horse_stats = track_data.groupby('血統登録番号').agg({
                    '総合適性スコア': 'mean',  # 各馬の適性スコア平均
                    '勝利': ['mean', 'count']   # 各馬の勝率とレース数
                }).reset_index()
                
                # カラム名を整理
                horse_stats.columns = ['血統登録番号', '適性スコア平均', '勝率', 'レース数']
                
                # 最低レース数のフィルタリング
                horse_stats_filtered = horse_stats[
                    horse_stats['レース数'] >= min_races_per_horse
                ].copy()
                
                if len(horse_stats_filtered) < min_horses_after_grouping:
                    print(f"警告: {track} で馬ごと集計後のサンプル数が{min_horses_after_grouping}未満 ({len(horse_stats_filtered)}) のためスキップします。")
                continue
            
                # データ検証と相関計算（強化版）
                aptitude_scores = horse_stats_filtered['適性スコア平均'].values
                win_rates = horse_stats_filtered['勝率'].values
                race_counts = horse_stats_filtered['レース数'].values
                
                # NaN・inf・重複値の包括的チェック
                valid_mask = (
                    ~np.isnan(aptitude_scores) & ~np.isinf(aptitude_scores) & 
                    ~np.isnan(win_rates) & ~np.isinf(win_rates)
                )
                
                aptitude_scores_valid = aptitude_scores[valid_mask]
                win_rates_valid = win_rates[valid_mask]
                race_counts_valid = race_counts[valid_mask]

                # データの多様性検証
                if (len(aptitude_scores_valid) < 2 or 
                    len(np.unique(aptitude_scores_valid)) < 2 or 
                    len(np.unique(win_rates_valid)) < 2):
                    print(f"警告: {track}で有効なデータが不足しているため相関計算をスキップします。")
                    continue

                correlation, p_value = stats.pearsonr(aptitude_scores_valid, win_rates_valid)
                
                track_stats[track] = {
                    'sample_size': len(aptitude_scores_valid),
                    'correlation': correlation, 'p_value': p_value,
                    'aptitude_data': aptitude_scores_valid, 
                    'win_data': win_rates_valid,
                    'race_counts': race_counts_valid,
                    'aptitude_stats': {
                        'min': aptitude_scores_valid.min(), 
                        'max': aptitude_scores_valid.max(),
                        'mean': aptitude_scores_valid.mean(),
                        'std': aptitude_scores_valid.std()
                    }
                }
                
            except Exception as e:
                print(f"エラー: {track}での統計計算中にエラーが発生: {e}")
                continue
        
        return track_stats

    def _calculate_place_correlation_statistics(self):
        """
        【改善版】競馬場別【複勝】相関統計計算（最適化済み + パフォーマンス改善）
        """
        print("競馬場別【複勝】相関統計計算中（馬ごとの適性スコア平均 vs 複勝率）...")
        track_stats = {}
        
        # データ品質検証（強化版）
        required_columns = ['場名', '総合適性スコア', '複勝', '血統登録番号']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"エラー：必要なカラムが不足: {missing_columns}")
            return track_stats

        # 設定の取得
        config = self.analysis_config
        min_sample_size = config['min_sample_size']
        min_races_per_horse = config['min_races_per_horse']
        min_horses_after_grouping = config['min_horses_after_grouping']

        for track in self.df['場名'].unique():
            track_data_for_track = self.df[self.df['場名'] == track].copy()
            
            # レース回数5回以上の馬でフィルタリング（ベクトル化処理）
            if self.horse_race_counts is not None:
                frequent_horses = self.horse_race_counts[self.horse_race_counts >= 5].index
                track_data = track_data_for_track[
                    track_data_for_track['血統登録番号'].isin(frequent_horses)
                ].copy()
                print(f"{track}: 【複勝】元データ {len(track_data_for_track)}頭 → 5回以上出走馬 {len(track_data)}頭")
            else:
                print(f"警告: {track} でレース回数情報が利用できないため、【複勝】分析は全馬を対象とします。")
                track_data = track_data_for_track.copy()
            
            # サンプルサイズ検証
            if len(track_data) < min_sample_size: 
                print(f"警告: {track} で【複勝】フィルタリング後のサンプルサイズが{min_sample_size}未満 ({len(track_data)}) のためスキップします。")
                continue
            
            # 馬ごとに適性スコア平均と複勝率を計算（ベクトル化処理）
            try:
                horse_stats = track_data.groupby('血統登録番号').agg({
                    '総合適性スコア': 'mean',  # 各馬の適性スコア平均
                    '複勝': ['mean', 'count']   # 各馬の複勝率とレース数
                }).reset_index()
                
                # カラム名を整理
                horse_stats.columns = ['血統登録番号', '適性スコア平均', '複勝率', 'レース数']
                
                # 最低レース数のフィルタリング
                horse_stats_filtered = horse_stats[
                    horse_stats['レース数'] >= min_races_per_horse
                ].copy()
                
                if len(horse_stats_filtered) < min_horses_after_grouping:
                    print(f"警告: {track} で【複勝】馬ごと集計後のサンプル数が{min_horses_after_grouping}未満 ({len(horse_stats_filtered)}) のためスキップします。")
                    continue
                
                # データ検証と相関計算（強化版）
                aptitude_scores = horse_stats_filtered['適性スコア平均'].values
                place_rates = horse_stats_filtered['複勝率'].values
                race_counts = horse_stats_filtered['レース数'].values
                
                # NaN・inf・重複値の包括的チェック
                valid_mask = (
                    ~np.isnan(aptitude_scores) & ~np.isinf(aptitude_scores) & 
                    ~np.isnan(place_rates) & ~np.isinf(place_rates)
                )
                
                # 【軽微な改善】外れ値の除去（IQR法）
                if len(aptitude_scores[valid_mask]) > 20:  # 十分なサンプルがある場合のみ
                    q25, q75 = np.percentile(aptitude_scores[valid_mask], [25, 75])
                    iqr = q75 - q25
                    outlier_mask = (
                        (aptitude_scores >= q25 - 1.5 * iqr) & 
                        (aptitude_scores <= q75 + 1.5 * iqr)
                    )
                    valid_mask = valid_mask & outlier_mask
                
                aptitude_scores_valid = aptitude_scores[valid_mask]
                place_rates_valid = place_rates[valid_mask]
                race_counts_valid = race_counts[valid_mask]

                # データの多様性検証
                if (len(aptitude_scores_valid) < 2 or 
                    len(np.unique(aptitude_scores_valid)) < 2 or 
                    len(np.unique(place_rates_valid)) < 2):
                    print(f"警告: {track}で有効なデータが不足しているため【複勝】相関計算をスキップします。")
                    continue

                correlation, p_value = stats.pearsonr(aptitude_scores_valid, place_rates_valid)
                
                track_stats[track] = {
                    'sample_size': len(aptitude_scores_valid),
                    'correlation': correlation, 'p_value': p_value,
                    'aptitude_data': aptitude_scores_valid, 
                    'place_data': place_rates_valid,
                    'race_counts': race_counts_valid,
                    'aptitude_stats': {
                        'min': aptitude_scores_valid.min(), 
                        'max': aptitude_scores_valid.max(),
                        'mean': aptitude_scores_valid.mean(),
                        'std': aptitude_scores_valid.std()
                    }
                }
                
            except Exception as e:
                print(f"エラー: {track}での【複勝】統計計算中にエラーが発生: {e}")
                continue
        
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
        fig.suptitle('競馬場別 馬ごとの適性スコア平均×勝率 相関分析（最適化済み設定：スピード0.5/0.5+スタミナ0.3/0.7）', fontproperties=self.font_prop, fontsize=16)
        
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
            ax.set_xlabel('馬ごとの総合適性スコア平均（最適化済み設定）', fontproperties=self.font_prop)
            ax.set_ylabel('馬ごとの勝率', fontproperties=self.font_prop)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for i in range(n_tracks, n_rows * n_cols): 
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_folder, '競馬場別適性相関分析_勝率_最適化済み設定.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"競馬場別適性相関分析_勝率_最適化済み設定.png を保存しました。")

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
        fig.suptitle('競馬場別 馬ごとの適性スコア平均×複勝率 相関分析（微調整版：上がり0.55+ペース0.75）', fontproperties=self.font_prop, fontsize=16)
        
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
            ax.set_xlabel('馬ごとの総合適性スコア平均（最適化済み設定）', fontproperties=self.font_prop)
            ax.set_ylabel('馬ごとの複勝率', fontproperties=self.font_prop)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for i in range(n_tracks, n_rows * n_cols): 
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_folder, '競馬場別適性相関分析_複勝率_微調整版.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"競馬場別適性相関分析_複勝率_微調整版.png を保存しました。")

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
    
    def analyze_by_track_condition(self, analysis_type='place'):
        """
        【新機能】馬場状態別の適性相関分析
        """
        print(f"\n=== 馬場状態別【{'複勝' if analysis_type == 'place' else '単勝'}】適性相関分析開始（微調整版） ===")
        
        if '馬場状態' not in self.df.columns:
            print("エラー: 馬場状態カラムが見つかりません。")
            return {}
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        track_conditions = self.df['馬場状態'].dropna().unique()
        condition_results = {}
        
        for condition in track_conditions:
            print(f"\n--- 馬場状態「{condition}」の分析開始 ---")
            condition_data = self.df[self.df['馬場状態'] == condition].copy()
            
            if len(condition_data) < 100:
                print(f"警告: 馬場状態「{condition}」のデータ数が少ない({len(condition_data)}件)")
                continue
            
            # 一時的にdfを条件別データに置き換え
            original_df = self.df.copy()
            self.df = condition_data
            
            try:
                if analysis_type == 'place':
                    correlation_stats = self._calculate_place_correlation_statistics()
                else:
                    correlation_stats = self._calculate_track_correlation_statistics()
                
                condition_results[condition] = correlation_stats
                
                # 馬場状態別の可視化作成
                self._create_track_condition_visualizations(correlation_stats, condition, analysis_type)
                
            except Exception as e:
                print(f"エラー: 馬場状態「{condition}」の分析中にエラー: {e}")
            finally:
                # dfを元に戻す
                self.df = original_df
        
        # 馬場状態比較可視化の作成
        self._create_track_condition_comparison(condition_results, analysis_type)
        
        print(f"馬場状態別【{'複勝' if analysis_type == 'place' else '単勝'}】適性相関分析完了")
        return condition_results

    def _create_track_condition_visualizations(self, track_stats, condition, analysis_type):
        """
        【新機能】馬場状態別の可視化作成
        """
        print(f"馬場状態「{condition}」の可視化作成中...")
        n_tracks = len(track_stats)
        if n_tracks == 0:
            print(f"馬場状態「{condition}」で可視化対象の競馬場データがありません。")
            return
        
        n_cols = 3
        n_rows = (n_tracks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        target_label = '複勝率' if analysis_type == 'place' else '勝率'
        fig.suptitle(f'馬場状態「{condition}」別 競馬場×{target_label} 相関分析（微調整版）', 
                    fontproperties=self.font_prop, fontsize=16)
        
        if n_tracks == 1: axes = np.array([[axes]]) 
        elif n_rows == 1: axes = axes.reshape(1, -1)
        elif n_cols == 1: axes = axes.reshape(-1, 1)
        
        track_names = list(track_stats.keys())
        for i, track in enumerate(track_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            stats_data = track_stats[track]
            
            # データプロット
            aptitude_scores = stats_data['aptitude_data']
            rates = stats_data.get('place_data' if analysis_type == 'place' else 'win_data', 
                                 stats_data.get('win_data', []))
            race_counts = stats_data['race_counts']
            
            if len(aptitude_scores) > 0:
                # 点のサイズをレース数に比例
                sizes = np.array(race_counts)
                if len(sizes) > 1 and sizes.max() > sizes.min():
                    sizes_normalized = 10 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 90
                else:
                    sizes_normalized = np.full_like(sizes, 30)
                
                color = 'darkgreen' if analysis_type == 'place' else 'darkblue'
                scatter = ax.scatter(aptitude_scores, rates, 
                                   s=sizes_normalized, alpha=0.6, 
                                   color=color, edgecolors='black', linewidth=0.3)
            
            # 回帰直線
            if len(aptitude_scores) > 1 and len(rates) > 1:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(aptitude_scores, rates)
                    x_line = np.array([aptitude_scores.min(), aptitude_scores.max()])
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color='red', linewidth=2, 
                           label=f'回帰直線 (R²={r_value**2:.3f})')
                except Exception as e:
                    print(f"回帰直線計算エラー ({track}@{condition}): {e}")
            
            ax.set_title(f'{track}\n相関係数: {stats_data["correlation"]:.3f} (p={stats_data["p_value"]:.3f})\n馬数: {len(aptitude_scores)}頭', 
                        fontproperties=self.font_prop)
            ax.set_xlabel('馬ごとの総合適性スコア平均（馬場状態補正済み）', fontproperties=self.font_prop)
            ax.set_ylabel(f'馬ごとの{target_label}', fontproperties=self.font_prop)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for i in range(n_tracks, n_rows * n_cols): 
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f'競馬場別適性相関分析_{target_label}_馬場状態_{condition}_微調整版.png'
        plt.savefig(os.path.join(self.output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{filename} を保存しました。")
    
    def _create_track_condition_comparison(self, condition_results, analysis_type):
        """
        【新機能】馬場状態間の比較可視化
        """
        print("馬場状態間の比較可視化作成中...")
        if not condition_results:
            print("比較対象の馬場状態データがありません。")
            return
        
        target_label = '複勝率' if analysis_type == 'place' else '勝率'
        
        # 各馬場状態の平均相関係数を計算
        condition_correlations = {}
        for condition, tracks_data in condition_results.items():
            correlations = [stats['correlation'] for stats in tracks_data.values() 
                          if not np.isnan(stats['correlation'])]
            if correlations:
                condition_correlations[condition] = {
                    'mean_correlation': np.mean(correlations),
                    'std_correlation': np.std(correlations),
                    'track_count': len(correlations)
                }
        
        if not condition_correlations:
            print("比較可能な相関データがありません。")
            return
        
        # 棒グラフで馬場状態別の平均相関係数を比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        conditions = list(condition_correlations.keys())
        mean_corrs = [condition_correlations[c]['mean_correlation'] for c in conditions]
        std_corrs = [condition_correlations[c]['std_correlation'] for c in conditions]
        track_counts = [condition_correlations[c]['track_count'] for c in conditions]
        
        # 平均相関係数の比較
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(conditions)]
        bars1 = ax1.bar(conditions, mean_corrs, yerr=std_corrs, capsize=5, 
                       color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title(f'馬場状態別の平均相関係数比較（{target_label}予測）', fontproperties=self.font_prop)
        ax1.set_ylabel('平均相関係数', fontproperties=self.font_prop)
        ax1.set_xlabel('馬場状態', fontproperties=self.font_prop)
        ax1.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for i, (bar, corr, count) in enumerate(zip(bars1, mean_corrs, track_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}\n({count}場)', ha='center', va='bottom', 
                    fontproperties=self.font_prop, fontsize=10)
        
        # サンプル数の比較
        bars2 = ax2.bar(conditions, track_counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('馬場状態別の分析対象競馬場数', fontproperties=self.font_prop)
        ax2.set_ylabel('競馬場数', fontproperties=self.font_prop)
        ax2.set_xlabel('馬場状態', fontproperties=self.font_prop)
        ax2.grid(True, alpha=0.3)
        
        for bar, count in zip(bars2, track_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{count}', ha='center', va='bottom', 
                    fontproperties=self.font_prop, fontsize=12)
        
        plt.tight_layout()
        filename = f'馬場状態別比較_{target_label}_微調整版.png'
        plt.savefig(os.path.join(self.output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{filename} を保存しました。")

    def _suggest_track_condition_strategies(self, category_summary, total_races):
        """
        【競馬専門知識】馬場状態別の投資戦略提案
        """
        print("\n=== 馬券投資戦略提案 ===")
        
        for category, summary in category_summary.items():
            percentage = (summary['count'] / total_races) * 100 if total_races > 0 else 0
            
            # カテゴリ別戦略アドバイス
            if '良・高速' in category:
                strategy = "瞬発力重視。先行・差し馬が有利。スピード指数重要度: 高"
            elif '良・標準' in category:
                strategy = "バランス型。オールラウンダー有利。総合能力重要度: 高"
            elif '良・低速' in category:
                strategy = "スタミナ重視。逃げ・追込み馬が有利。ペース配分重要度: 高"
            elif '稍重・高速' in category:
                strategy = "軽微なパワー要求+瞬発力。中距離得意馬有利。"
            elif '稍重・標準' in category:
                strategy = "パワー+バランス型。実績重視。条件戦強い馬有利。"
            elif '稍重・低速' in category:
                strategy = "持続力重視。長距離適性+ペース配分能力重要。"
            elif '重・高速' in category:
                strategy = "パワー+瞬発力。重賞実績ある馬が安定。"
            elif '重・標準' in category:
                strategy = "真のスタミナ勝負。長距離・重賞実績重視。"
            elif '重・低速' in category:
                strategy = "超持続力勝負。逃げ粘り・大逃げ型が台頭。距離延長OK馬。"
            elif '不良' in category:
                strategy = "パワー・スタミナ最重要。荒れる可能性大。穴馬狙い有効。"
            else:
                strategy = "データ要確認。慎重な馬選択を推奨。"
            
            print(f"【{category}】({percentage:.1f}%): {strategy}")
        
        # 全体的な傾向分析
        high_speed_ratio = sum(v['count'] for k, v in category_summary.items() if '高速' in k) / total_races * 100
        low_speed_ratio = sum(v['count'] for k, v in category_summary.items() if '低速' in k) / total_races * 100
        
        print(f"\n=== 全体傾向 ===")
        print(f"高速馬場: {high_speed_ratio:.1f}% → スピード・瞬発力重視戦略")
        print(f"低速馬場: {low_speed_ratio:.1f}% → スタミナ・ペース重視戦略")
        print(f"標準馬場: {100-high_speed_ratio-low_speed_ratio:.1f}% → バランス型戦略")

    def analyze_track_condition_details(self, analysis_type='place'):
        """
        【新機能】馬場状態の詳細分析
        """
        print(f"\n=== 馬場状態詳細分析開始 ===")
        
        if '馬場状態' not in self.df.columns:
            print("エラー: 馬場状態カラムが見つかりません。")
            return {}
        
        # 馬場状態の分布分析
        condition_distribution = self.df['馬場状態'].value_counts()
        print("\n=== 馬場状態分布 ===")
        for condition, count in condition_distribution.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"{condition}: {count}件 ({percentage:.2f}%)")
        
        # 馬場状態別の勝率・複勝率分析
        target_column = '複勝' if analysis_type == 'place' else '勝利'
        target_name = '複勝率' if analysis_type == 'place' else '勝率'
        
        condition_performance = self.df.groupby('馬場状態')[target_column].agg(['mean', 'count']).reset_index()
        condition_performance = condition_performance[condition_performance['count'] >= 50]  # 最低50レース
        condition_performance = condition_performance.sort_values('mean', ascending=False)
        
        print(f"\n=== 馬場状態別{target_name}ランキング（50レース以上） ===")
        for _, row in condition_performance.head(10).iterrows():
            print(f"{row['馬場状態']}: {target_name}{row['mean']:.3f} ({row['count']}レース)")
        
        # 競馬場×馬場状態のクロス分析
        track_condition_analysis = self.df.groupby(['場名', '馬場状態']).agg({
            target_column: ['mean', 'count'],
            '総合適性スコア': 'mean'
        }).reset_index()
        
        track_condition_analysis.columns = ['場名', '馬場状態', f'{target_name}', 'レース数', '平均適性スコア']
        track_condition_analysis = track_condition_analysis[track_condition_analysis['レース数'] >= 20]
        
        print(f"\n=== 競馬場×馬場状態 最高{target_name}組み合わせ（20レース以上） ===")
        top_combinations = track_condition_analysis.sort_values(f'{target_name}', ascending=False).head(10)
        for _, row in top_combinations.iterrows():
            print(f"{row['場名']}×{row['馬場状態']}: {target_name}{row[f'{target_name}']:.3f} "
                  f"({row['レース数']}レース, 適性スコア{row['平均適性スコア']:.2f})")
        
        return {
            'distribution': condition_distribution,
            'performance': condition_performance,
            'track_condition_cross': track_condition_analysis
        }

    def analyze_logistic_regression(self, target_type='place'):
        """
        【新機能】ロジスティック回帰による勝率・複勝率予測分析
        """
        print(f"\n=== ロジスティック回帰分析開始 ({target_type}) ===")
        
        # scikit-learnの必要なモジュールをインポート
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
            import numpy as np
        except ImportError:
            print("❌ scikit-learnが必要です。pip install scikit-learn でインストールしてください。")
            return
        
        # 目的変数の設定
        target_column = '複勝' if target_type == 'place' else '勝利'
        target_name = '複勝率' if target_type == 'place' else '勝率'
        
        # 特徴量カラムの定義
        feature_columns = ['スピード適性', 'パワー適性', '技術適性', '枠順適性', '馬場適性', '総合適性スコア']
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        if len(available_features) < 4:
            print(f"❌ 必要な特徴量が不足しています。利用可能: {available_features}")
            return
        
        print(f"📊 使用特徴量: {available_features}")
        print(f"🎯 予測対象: {target_name}")
        
        # 全体的なロジスティック回帰分析
        overall_results = self._perform_overall_logistic_regression(available_features, target_column, target_name)
        
        # 競馬場別ロジスティック回帰分析
        track_results = self._perform_track_wise_logistic_regression(available_features, target_column, target_name)
        
        # 馬場状態別ロジスティック回帰分析
        condition_results = self._perform_condition_wise_logistic_regression(available_features, target_column, target_name)
        
        # 結果の可視化
        self._create_logistic_regression_visualizations(overall_results, track_results, condition_results, target_name)
        
        # 詳細レポートの生成
        self._generate_logistic_regression_report(overall_results, track_results, condition_results, target_name)
        
        print(f"✅ ロジスティック回帰分析({target_name})が完了しました。")

    def _perform_overall_logistic_regression(self, features, target_column, target_name):
        """全体データでのロジスティック回帰分析"""
        print(f"\n--- 全体ロジスティック回帰分析 ---")
        
        # scikit-learnのインポート
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
        import numpy as np
        
        # データ準備
        data = self.df[features + [target_column]].dropna()
        X = data[features].values
        y = data[target_column].values
        
        print(f"サンプル数: {len(data)}件, 正例率: {y.mean():.3f}")
        
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 訓練・テストデータ分割
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # ロジスティック回帰モデル構築
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 各種評価指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # クロスバリデーション
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        
        # 特徴量重要度（係数）
        feature_importance = dict(zip(features, model.coef_[0]))
        
        print(f"精度: {accuracy:.3f}")
        print(f"適合率: {precision:.3f}")
        print(f"再現率: {recall:.3f}")
        print(f"F1スコア: {f1:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"CV-AUC平均: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # ROC曲線用データ
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            },
            'feature_importance': feature_importance,
            'roc_data': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'sample_size': len(data),
            'positive_rate': y.mean()
        }

    def _perform_track_wise_logistic_regression(self, features, target_column, target_name):
        """競馬場別ロジスティック回帰分析"""
        print(f"\n--- 競馬場別ロジスティック回帰分析 ---")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np
        
        track_results = {}
        
        for track_name in self.df['場名'].unique():
            track_data = self.df[self.df['場名'] == track_name]
            track_subset = track_data[features + [target_column]].dropna()
            
            # 最低サンプル数チェック
            if len(track_subset) < 200:
                continue
            
            X = track_subset[features].values
            y = track_subset[target_column].values
            
            # 正例が少なすぎる場合はスキップ
            if y.sum() < 20:
                continue
            
            # データ標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # モデル構築
            model = LogisticRegression(random_state=42, max_iter=1000)
            
            try:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1]
                
                accuracy = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_prob)
                feature_importance = dict(zip(features, model.coef_[0]))
                
                track_results[track_name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'feature_importance': feature_importance,
                    'sample_size': len(track_subset),
                    'positive_rate': y.mean()
                }
                
                print(f"  {track_name}: AUC={auc:.3f}, 精度={accuracy:.3f}, サンプル={len(track_subset)}")
                
            except Exception as e:
                print(f"  {track_name}: エラー - {e}")
                continue
        
        return track_results

    def _perform_condition_wise_logistic_regression(self, features, target_column, target_name):
        """馬場状態別ロジスティック回帰分析"""
        print(f"\n--- 馬場状態別ロジスティック回帰分析 ---")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np
        
        condition_results = {}
        
        # 主要な馬場状態のみ分析
        main_conditions = ['良', '稍重', '重', '不良']
        
        for condition in self.df['馬場状態'].unique():
            if not any(mc in str(condition) for mc in main_conditions):
                continue
                
            condition_data = self.df[self.df['馬場状態'] == condition]
            condition_subset = condition_data[features + [target_column]].dropna()
            
            # 最低サンプル数チェック
            if len(condition_subset) < 100:
                continue
            
            X = condition_subset[features].values
            y = condition_subset[target_column].values
            
            # 正例が少なすぎる場合はスキップ
            if y.sum() < 10:
                continue
        
            # データ標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # モデル構築
            model = LogisticRegression(random_state=42, max_iter=1000)
            
            try:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1]
                
                accuracy = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_prob)
                feature_importance = dict(zip(features, model.coef_[0]))
                
                condition_results[condition] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'feature_importance': feature_importance,
                    'sample_size': len(condition_subset),
                    'positive_rate': y.mean()
                }
                
                print(f"  {condition}: AUC={auc:.3f}, 精度={accuracy:.3f}, サンプル={len(condition_subset)}")
                
            except Exception as e:
                print(f"  {condition}: エラー - {e}")
                continue
        
        return condition_results

    def _create_logistic_regression_visualizations(self, overall_results, track_results, condition_results, target_name):
        """ロジスティック回帰分析結果の可視化"""
        print(f"\n📈 ロジスティック回帰可視化作成中...")
        
        import numpy as np
        
        # 2x2のサブプロット作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROC曲線
        roc_data = overall_results['roc_data']
        ax1.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC曲線 (AUC = {overall_results["metrics"]["auc"]:.3f})',
                color='darkorange', lw=2)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='ランダム予測')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('偽陽性率 (False Positive Rate)')
        ax1.set_ylabel('真陽性率 (True Positive Rate)')
        ax1.set_title(f'1. ROC曲線 - {target_name}予測')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. 特徴量重要度（全体）
        importance = overall_results['feature_importance']
        features = list(importance.keys())
        values = list(importance.values())
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        bars = ax2.barh(features, values, color=colors, alpha=0.7)
        ax2.set_xlabel('ロジスティック回帰係数')
        ax2.set_title(f'2. 特徴量重要度 - {target_name}予測')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 係数値をバーに表示
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax2.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        # 3. 競馬場別AUC比較
        if track_results:
            track_names = list(track_results.keys())
            track_aucs = [track_results[name]['auc'] for name in track_names]
            
            bars = ax3.bar(range(len(track_names)), track_aucs, alpha=0.7, color='lightgreen')
            ax3.set_xlabel('競馬場')
            ax3.set_ylabel('AUC')
            ax3.set_title(f'3. 競馬場別予測精度 - {target_name}')
            ax3.set_xticks(range(len(track_names)))
            ax3.set_xticklabels(track_names, rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム予測')
            
            # AUC値をバーに表示
            for bar, auc in zip(bars, track_aucs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom')
        
        # 4. 馬場状態別AUC比較
        if condition_results:
            condition_names = list(condition_results.keys())
            condition_aucs = [condition_results[name]['auc'] for name in condition_names]
            
            bars = ax4.bar(range(len(condition_names)), condition_aucs, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('馬場状態')
            ax4.set_ylabel('AUC')
            ax4.set_title(f'4. 馬場状態別予測精度 - {target_name}')
            ax4.set_xticks(range(len(condition_names)))
            ax4.set_xticklabels(condition_names, rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム予測')
            
            # AUC値をバーに表示
            for bar, auc in zip(bars, condition_aucs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # ファイル保存
        target_suffix = 'place' if target_name == '複勝率' else 'win'
        filename = f"ロジスティック回帰分析_{target_suffix}.png"
        plt.savefig(os.path.join(self.output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{filename} を保存しました。")

    def _generate_logistic_regression_report(self, overall_results, track_results, condition_results, target_name):
        """ロジスティック回帰分析の詳細レポート生成"""
        print(f"📄 ロジスティック回帰レポート生成中...")
        
        target_suffix = 'place' if target_name == '複勝率' else 'win'
        report_filename = os.path.join(self.output_folder, f"ロジスティック回帰分析レポート_{target_suffix}.txt")
        
        report_lines = [
            f"=== ロジスティック回帰分析レポート - {target_name}予測 ===\n",
            f"分析実行日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "【全体分析結果】",
            f"サンプル数: {overall_results['sample_size']:,}件",
            f"正例率: {overall_results['positive_rate']:.3f}",
            f"精度: {overall_results['metrics']['accuracy']:.3f}",
            f"適合率: {overall_results['metrics']['precision']:.3f}",
            f"再現率: {overall_results['metrics']['recall']:.3f}",
            f"F1スコア: {overall_results['metrics']['f1']:.3f}",
            f"AUC: {overall_results['metrics']['auc']:.3f}",
            f"交差検証AUC: {overall_results['metrics']['cv_auc_mean']:.3f} (±{overall_results['metrics']['cv_auc_std']:.3f})\n",
            
            "【特徴量重要度】"
        ]
        
        # 特徴量重要度をソート
        importance_sorted = sorted(overall_results['feature_importance'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in importance_sorted:
            effect = "正の影響" if coef > 0 else "負の影響"
            report_lines.append(f"{feature}: {coef:.4f} ({effect})")
        
        # 競馬場別結果
        if track_results:
            report_lines.extend([
                "\n【競馬場別分析結果】",
                "競馬場名: AUC, 精度, サンプル数"
            ])
            
            track_sorted = sorted(track_results.items(), key=lambda x: x[1]['auc'], reverse=True)
            for track_name, results in track_sorted:
                report_lines.append(
                    f"{track_name}: AUC={results['auc']:.3f}, "
                    f"精度={results['accuracy']:.3f}, "
                    f"サンプル={results['sample_size']:,}件"
                )
        
        # 馬場状態別結果
        if condition_results:
            report_lines.extend([
                "\n【馬場状態別分析結果】",
                "馬場状態: AUC, 精度, サンプル数"
            ])
            
            condition_sorted = sorted(condition_results.items(), key=lambda x: x[1]['auc'], reverse=True)
            for condition, results in condition_sorted:
                report_lines.append(
                    f"{condition}: AUC={results['auc']:.3f}, "
                    f"精度={results['accuracy']:.3f}, "
                    f"サンプル={results['sample_size']:,}件"
                )
        
        # 実用的な解釈とアドバイス
        overall_auc = overall_results['metrics']['auc']
        if overall_auc >= 0.7:
            interpretation = "優秀 - 実用的な予測性能"
        elif overall_auc >= 0.6:
            interpretation = "良好 - 参考として活用可能"
        elif overall_auc >= 0.55:
            interpretation = "やや有効 - 他の指標と組み合わせて使用"
        else:
            interpretation = "要改善 - 特徴量やモデルの見直しが必要"
        
        report_lines.extend([
            f"\n【総合評価】",
            f"予測性能: {interpretation}",
            f"推奨用途: {'メイン予想材料として活用' if overall_auc >= 0.65 else '補助的な判断材料として活用'}",
            f"\n【活用のポイント】",
            "- 係数が正の特徴量は{target_name}向上に寄与",
            "- 係数が負の特徴量は{target_name}低下に寄与", 
            "- AUCが0.6以上の競馬場・馬場状態で特に有効",
            "- サンプル数が多い結果ほど信頼性が高い"
        ])
        
        # レポート保存
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 {report_filename} を保存しました。")
        
        # 重要な結果をコンソール出力
        print(f"\n=== {target_name}予測 - 重要な知見 ===")
        print(f"🎯 全体AUC: {overall_auc:.3f} ({interpretation})")
        print("🔑 最重要特徴量:")
        for feature, coef in importance_sorted[:3]:
            effect = "↗️" if coef > 0 else "↘️"
            print(f"   {feature}: {coef:.3f} {effect}")
        
        if track_results:
            best_track = max(track_results.items(), key=lambda x: x[1]['auc'])
            print(f"🏆 最高AUC競馬場: {best_track[0]} (AUC: {best_track[1]['auc']:.3f})")

    def analyze_aptitude_logistic_curve(self, target_type='place'):
        """適性スコア vs 勝率/複勝率のロジスティック回帰曲線分析"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            import numpy as np
        except ImportError:
            print("❌ scikit-learn が必要です。")
            return
        
        print(f"\n=== 適性スコア ロジスティック曲線分析開始 ===")
        
        target_column = '複勝' if target_type == 'place' else '勝利'
        target_name = '複勝率' if target_type == 'place' else '勝率'
        
        # データ準備
        data = self.df[['総合適性スコア', target_column]].dropna()
        X = data['総合適性スコア'].values.reshape(-1, 1)
        y = data[target_column].values
        
        print(f"サンプル数: {len(data)}件")
        print(f"{target_name}全体平均: {y.mean():.3f}")
        
        # データ標準化とモデル構築
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # 予測精度
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
        print(f"予測精度 (AUC): {auc:.3f}")
        
        # 可視化用データ生成
        score_min = X.min() - (X.max() - X.min()) * 0.2  # 20%下方拡張
        score_max = X.max() + (X.max() - X.min()) * 0.3  # 30%上方拡張
        score_range = np.linspace(score_min, score_max, 1000)
        score_range_scaled = scaler.transform(score_range.reshape(-1, 1))
        predicted_probs = model.predict_proba(score_range_scaled)[:, 1]
        
        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 実際のデータポイント
        success_mask = y == 1
        fail_mask = y == 0
        
        success_scores = X[success_mask].flatten()
        fail_scores = X[fail_mask].flatten()
        
        # ジッター追加
        np.random.seed(42)  # 再現性のため
        success_y = np.ones(len(success_scores)) + np.random.normal(0, 0.02, len(success_scores))
        fail_y = np.zeros(len(fail_scores)) + np.random.normal(0, 0.02, len(fail_scores))
        
        ax.scatter(success_scores, success_y, alpha=0.3, s=8, color='black', marker='o', label=f'{target_name}成功')
        ax.scatter(fail_scores, fail_y, alpha=0.3, s=8, color='gray', marker='o', label=f'{target_name}失敗')
        
        # ロジスティック回帰曲線
        ax.plot(score_range, predicted_probs, color='red', linewidth=3, 
               label=f'ロジスティック回帰曲線 (AUC = {auc:.3f})')
        
        # 50%確率ライン
        prob_50_idx = np.argmin(np.abs(predicted_probs - 0.5))
        score_50 = score_range[prob_50_idx]
        ax.axvline(x=score_50, color='orange', linestyle='--', alpha=0.7, 
                  label=f'50%確率ライン ({score_50:.3f})')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
        
        # 平均スコアライン
        score_mean = X.mean()
        ax.axvline(x=score_mean, color='green', linestyle='--', alpha=0.7, 
                  label=f'平均適性スコア ({score_mean:.3f})')
        
        # グラフ装飾
        ax.set_xlabel('総合適性スコア', fontsize=14)
        ax.set_ylabel(f'{target_name}確率', fontsize=14)
        ax.set_title(f'総合適性スコア vs {target_name} - ロジスティック回帰分析', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center right')
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        # ファイル保存
        target_suffix = 'place' if target_type == 'place' else 'win'
        filename = f"適性スコア_ロジスティック曲線_{target_suffix}.png"
        plt.savefig(os.path.join(self.output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{filename} を保存しました。")
        
        # 重要な統計値の表示
        print(f"\n=== 重要な知見 ===")
        print(f"🎯 50%確率の適性スコア: {score_50:.3f}")
        print(f"📈 平均適性スコアでの予測{target_name}: {model.predict_proba(scaler.transform([[score_mean]]))[0][1]:.3f}")
        
        # パーセンタイル別予測確率
        percentiles = [25, 50, 75, 90]
        score_values = np.percentile(X, percentiles)
        score_values_scaled = scaler.transform(score_values.reshape(-1, 1))
        predicted_probs_pct = model.predict_proba(score_values_scaled)[:, 1]
        
        print("📊 適性スコア別予測確率:")
        for pct, score, prob in zip(percentiles, score_values, predicted_probs_pct):
            print(f"  {pct}%ile (スコア{score:.3f}): 予測{target_name} {prob:.3f}")
        
        print(f"✅ 適性スコア ロジスティック曲線分析が完了しました。")
    
    def _create_aptitude_logistic_curve_visualization(self, target_type='place'):
        """【改良版】適性スコア vs 勝率/複勝率のロジスティック回帰曲線可視化"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            import numpy as np
        except ImportError:
            print("❌ scikit-learn が必要です。")
            return
        
        print(f"\n=== 適性スコア ロジスティック曲線分析開始 ===")
        
        # 必要な列の存在確認
        required_columns = ['IDM', 'テン指数', '上がり指数', 'ペース指数', '着順']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"❌ 必要なカラムが不足: {missing_columns}")
            return
        
        # データ準備
        data = self.df[required_columns].dropna()
        if len(data) == 0:
            print("❌ 分析可能なデータがありません。")
            return
        
        print(f"データ読み込み完了: {len(data)}件")
        
        # 適性スコア計算
        print("適性スコア計算中...")
        weights = {'IDM': 0.3, 'テン指数': 0.25, '上がり指数': 0.25, 'ペース指数': 0.2}
        aptitude_score = np.zeros(len(data))
        
        for column, weight in weights.items():
            if column in data.columns:
                values = np.array(data[column].fillna(data[column].median()))
                aptitude_score += values * weight
                print(f"✓ {column}を使用")
        
        print(f"✓ 適性スコア計算完了（平均: {aptitude_score.mean():.2f}, 標準偏差: {aptitude_score.std():.2f}）")
        
        # 目的変数作成
        print("目的変数作成中...")
        win_target = np.array((data['着順'] == 1).astype(int))
        place_target = np.array(((data['着順'] >= 1) & (data['着順'] <= 3)).astype(int))
        
        print(f"✓ 勝利率: {win_target.mean():.3f}")
        print(f"✓ 複勝率: {place_target.mean():.3f}")
        
        # 分析対象の設定
        if target_type == 'place':
            y = place_target
            target_name = '複勝率'
        else:
            y = win_target
            target_name = '勝率'
        
        # 有効データのフィルタリング
        aptitude_score = np.array(aptitude_score)  # 型を確実にnumpy arrayに
        y = np.array(y)  # 型を確実にnumpy arrayに
        valid_mask = ~(np.isnan(aptitude_score) | np.isnan(y))
        X = aptitude_score[valid_mask].reshape(-1, 1)
        y = y[valid_mask]
        
        print(f"サンプル数: {len(X)}件")
        print(f"{target_name}全体平均: {y.mean():.3f}")
        
        # ロジスティック回帰モデル構築
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # 予測精度
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
        print(f"予測精度 (AUC): {auc:.3f}")
        
        # 可視化用データ生成
        score_min = X.min() - (X.max() - X.min()) * 0.2  # 20%下方拡張
        score_max = X.max() + (X.max() - X.min()) * 0.3  # 30%上方拡張
        score_range = np.linspace(score_min, score_max, 1000)
        score_range_scaled = scaler.transform(score_range.reshape(-1, 1))
        predicted_probs = model.predict_proba(score_range_scaled)[:, 1]
        
        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 実際のデータポイント（ジッター付き）
        np.random.seed(42)
        success_mask = y == 1
        fail_mask = y == 0
        
        success_scores = X[success_mask].flatten()
        fail_scores = X[fail_mask].flatten()
        
        success_y = np.ones(len(success_scores)) + np.random.normal(0, 0.02, len(success_scores))
        fail_y = np.zeros(len(fail_scores)) + np.random.normal(0, 0.02, len(fail_scores))
        
        ax.scatter(success_scores, success_y, alpha=0.4, s=12, color='black', marker='o', label=f'{target_name}成功')
        ax.scatter(fail_scores, fail_y, alpha=0.2, s=8, color='gray', marker='o', label=f'{target_name}失敗')
        
        # ロジスティック回帰曲線（S字カーブ）
        ax.plot(score_range, predicted_probs, color='red', linewidth=3, 
               label=f'ロジスティック回帰曲線 (AUC = {auc:.3f})')
        
        # 50%確率ライン
        prob_50_idx = np.argmin(np.abs(predicted_probs - 0.5))
        score_50 = score_range[prob_50_idx]
        ax.axvline(x=score_50, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                  label=f'50%確率ライン (スコア{score_50:.1f})')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.6)
        
        # 平均スコアライン
        score_mean = X.mean()
        pred_mean = model.predict_proba(scaler.transform([[score_mean]]))[0][1]
        ax.axvline(x=score_mean, color='green', linestyle='--', linewidth=2, alpha=0.8, 
                  label=f'平均適性スコア (スコア{score_mean:.1f})')
        
        # グラフ装飾
        ax.set_xlabel('適性スコア（IDM30% + テン指数25% + 上がり指数25% + ペース指数20%）', 
                     fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel(f'{target_name}確率', fontproperties=self.font_prop, fontsize=12)
        ax.set_title(f'適性スコア vs {target_name} - ロジスティック回帰曲線', 
                    fontproperties=self.font_prop, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(prop=self.font_prop, loc='center right')
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        # 結果保存ディレクトリ作成
        curve_folder = os.path.join("results", "logistic_curve")
        if not os.path.exists(curve_folder):
            os.makedirs(curve_folder)
        
        # ファイル保存
        target_suffix = 'place' if target_type == 'place' else 'win'
        filename = f"適性スコア_ロジスティック曲線_{target_suffix}.png"
        filepath = os.path.join(curve_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 {filepath} を保存しました。")
        
        # 重要な統計値の表示
        print(f"\n=== 重要な知見 ===")
        print(f"🎯 50%確率の適性スコア: {score_50:.1f}")
        print(f"📈 平均適性スコアでの予測{target_name}: {pred_mean:.3f}")
        
        # パーセンタイル別予測確率
        percentiles = [25, 50, 75, 90]
        score_values = np.percentile(X, percentiles)
        score_values_scaled = scaler.transform(score_values.reshape(-1, 1))
        predicted_probs_pct = model.predict_proba(score_values_scaled)[:, 1]
        
        print(f"📊 適性スコア別予測確率:")
        for pct, score, prob in zip(percentiles, score_values, predicted_probs_pct):
            print(f"  {pct}%ile (スコア{score:.1f}): 予測{target_name} {prob:.3f}")
        
        print(f"✅ 適性スコア ロジスティック曲線分析が完了しました。")


def main():
    import sys
    
    print("=== 競馬場特徴×馬能力適性分析システム（馬場差考慮版）開始 ===")
    
    # 芝レース限定オプション
    turf_only_choice = input("芝レースのみに限定しますか？ (y/n): ").strip().lower()
    turf_only = turf_only_choice in ['y', 'yes', 'はい', '1']
    
    if turf_only:
        print("🌱 芝レース限定モードで分析を実行します。")
        analyzer = TrackHorseAbilityAnalyzer(
            output_folder="results/track_horse_ability_analysis_turf_only",
            turf_only=True
        )
    else:
        print("🏇 全レース（芝・ダート）で分析を実行します。")
        analyzer = TrackHorseAbilityAnalyzer(turf_only=False)
    
    if not analyzer.load_and_preprocess_data():
        print("データの読み込みに失敗しました。")
        return
    

    
    surface_label = "【芝レース限定】" if turf_only else "【芝・ダート全レース】"
    print(f"\n=== 【馬場差考慮版】{surface_label}分析機能メニュー ===")
    print("1. 競馬場別適性相関分析（勝率）")
    print("2. 競馬場別適性相関分析（複勝率）")
    print("3. 馬場状態別適性相関分析（複勝率）")
    print("4. 馬場状態詳細分析")
    print("5. ロジスティック回帰分析（複勝率予測）")
    print("6. ロジスティック回帰分析（勝率予測）")
    print("7. 適性スコア ロジスティック曲線分析（複勝率）")
    print("8. 適性スコア ロジスティック曲線分析（勝率）")
    print("9. 全分析実行")
    
    choice = input("実行する分析を選択してください (1-9): ").strip()
    
    try:
        if choice == '1':
            print("\n=== 競馬場別適性相関分析（勝率）実行 ===")
            analyzer.analyze_track_aptitude_correlation()
            
        elif choice == '2':
            print("\n=== 競馬場別適性相関分析（複勝率）実行 ===")
            analyzer.analyze_place_aptitude_correlation()
            
        elif choice == '3':
            print("\n=== 馬場状態別適性相関分析（複勝率）実行 ===")
            analyzer.analyze_by_track_condition('place')
            
        elif choice == '4':
            print("\n=== 馬場状態詳細分析実行 ===")
            analyzer.analyze_track_condition_details('place')
            
        elif choice == '5':
            print("\n=== ロジスティック回帰分析（複勝率予測）実行 ===")
            analyzer.analyze_logistic_regression('place')
            
        elif choice == '6':
            print("\n=== ロジスティック回帰分析（勝率予測）実行 ===")
            analyzer.analyze_logistic_regression('win')
            
        elif choice == '7':
            print("\n=== 適性スコア ロジスティック曲線分析（複勝率）実行 ===")
            analyzer._create_aptitude_logistic_curve_visualization('place')
            
        elif choice == '8':
            print("\n=== 適性スコア ロジスティック曲線分析（勝率）実行 ===")
            analyzer._create_aptitude_logistic_curve_visualization('win')
            
        elif choice == '9':
            print("\n=== 全分析実行 ===")
            
            # 1. 競馬場別分析
            print("\n--- 1/8: 競馬場別適性相関分析（勝率） ---")
            analyzer.analyze_track_aptitude_correlation()
            
            print("\n--- 2/8: 競馬場別適性相関分析（複勝率） ---")
            analyzer.analyze_place_aptitude_correlation()
            
            # 2. 馬場状態別分析
            print("\n--- 3/8: 馬場状態別適性相関分析（複勝率） ---")
            analyzer.analyze_by_track_condition('place')
            
            # 3. 馬場状態詳細分析
            print("\n--- 4/8: 馬場状態詳細分析 ---")
            analyzer.analyze_track_condition_details('place')
            
            # 4. ロジスティック回帰分析（複勝率）
            print("\n--- 5/8: ロジスティック回帰分析（複勝率予測） ---")
            analyzer.analyze_logistic_regression('place')
            
            # 5. ロジスティック回帰分析（勝率）
            print("\n--- 6/8: ロジスティック回帰分析（勝率予測） ---")
            analyzer.analyze_logistic_regression('win')
            
            # 6. ロジスティック曲線分析（複勝率）
            print("\n--- 7/8: 適性スコア ロジスティック曲線分析（複勝率） ---")
            analyzer._create_aptitude_logistic_curve_visualization('place')
            
            # 7. ロジスティック曲線分析（勝率）
            print("\n--- 8/8: 適性スコア ロジスティック曲線分析（勝率） ---")
            analyzer._create_aptitude_logistic_curve_visualization('win')
            
            # 【新機能】馬場適性統計サマリーの出力
            print("\n=== 馬場適性統計サマリー ===")
            if '馬場適性' in analyzer.df.columns:
                track_aptitude_stats = analyzer.df.groupby('馬場状態')['馬場適性'].agg(['mean', 'std', 'count']).round(3)
                print("馬場状態別の平均馬場適性:")
                print(track_aptitude_stats)
                
                # 馬場適性の分布確認
                overall_aptitude_stats = analyzer.df['馬場適性'].describe()
                print(f"\n全体の馬場適性統計:")
                print(f"平均: {overall_aptitude_stats['mean']:.3f}")
                print(f"標準偏差: {overall_aptitude_stats['std']:.3f}")
                print(f"最小値: {overall_aptitude_stats['min']:.3f}")
                print(f"最大値: {overall_aptitude_stats['max']:.3f}")
                
                # 総合適性スコアの構成要素確認
                print(f"\n総合適性スコアの構成要素平均:")
                component_stats = analyzer.df[['スピード適性', 'パワー適性', '技術適性', '枠順適性', '馬場適性']].mean()
                for component, value in component_stats.items():
                    print(f"{component}: {value:.3f}")
                
                print("\n【馬場差考慮版の特徴】")
                print("✅ 馬場適性が総合適性スコアの20%を占める")
                print("✅ 12種類の詳細馬場状態に対応")
                print("✅ 過去実績による動的調整機能")
                print("✅ 馬場要求と馬能力の理論的適合度計算")
            else:
                print("警告: 馬場適性カラムが見つかりません。")
        
        else:
            print("無効な選択です。1-9の番号を入力してください。")
            return
            
    except Exception as e:
        print(f"分析実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n=== 【馬場差考慮版】{surface_label}分析完了 ===")
    print(f"結果は {analyzer.output_folder} フォルダに保存されました。")
    if turf_only:
        print("\n【芝レース限定分析の特徴】")
        print("🌱 芝レースのみの高精度分析")
        print("🌱 芝馬場特有の傾向を抽出")
        print("🌱 ダートレースのノイズを除去")
    print("\n【馬場差考慮の効果】")
    print("- 馬場状態による能力値補正")
    print("- 馬の能力バランスと馬場要求の適合度評価")
    print("- 過去の馬場状態別実績による動的調整")
    print("- 総合適性スコアへの馬場適性の統合（20%重み）")

if __name__ == "__main__":
    main() 
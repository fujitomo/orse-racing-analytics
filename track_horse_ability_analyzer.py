import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class TrackHorseAbilityAnalyzer:
    """
    競馬場特徴×馬能力適性分析システム
    
    機能:
    1. 競馬場の物理的特徴の数値化
    2. 馬の総合能力値の算出
    3. 競馬場適性の定量化
    4. 機械学習による勝率予測
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/track_horse_ability_analysis"):
        """
        初期化
        
        Args:
            data_folder (str): データフォルダのパス
            output_folder (str): 結果出力先フォルダのパス
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        self.track_features = None
        self.scaler = StandardScaler()
        
        # 日本語フォント設定
        self._setup_japanese_font()
        
        # 競馬場特徴の定義
        self._define_track_characteristics()
        
    def _setup_japanese_font(self):
        """日本語フォント設定（確実版）"""
        import matplotlib
        import matplotlib.font_manager as fm
        import platform
        
        try:
            print("日本語フォント設定を開始（確実版）...")
            
            if platform.system() == 'Windows':
                # Windowsフォントディレクトリ
                windows_fonts_dir = r'C:\Windows\Fonts'
                
                # 利用可能な日本語フォントファイルを直接指定
                font_candidates = [
                    (os.path.join(windows_fonts_dir, 'YuGothM.ttc'), 'Yu Gothic Medium'),
                    (os.path.join(windows_fonts_dir, 'YuGothB.ttc'), 'Yu Gothic Bold'),
                    (os.path.join(windows_fonts_dir, 'msgothic.ttc'), 'MS Gothic'),
                    (os.path.join(windows_fonts_dir, 'meiryo.ttc'), 'Meiryo'),
                    (os.path.join(windows_fonts_dir, 'msmincho.ttc'), 'MS Mincho'),
                ]
                
                self.font_prop = None
                for font_path, font_name in font_candidates:
                    if os.path.exists(font_path):
                        try:
                            # FontPropertiesを直接作成
                            self.font_prop = fm.FontProperties(fname=font_path)
                            
                            # matplotlib設定も更新
                            matplotlib.rcParams['font.family'] = [font_name]
                            matplotlib.rcParams['axes.unicode_minus'] = False
                            
                            print(f"フォント設定成功: {font_path} ({font_name})")
                            break
                        except Exception as e:
                            print(f"フォント読み込みエラー: {font_path} - {e}")
                            continue
                
                if self.font_prop is None:
                    print("日本語フォントが見つかりません。デフォルトフォントを使用します。")
                    self.font_prop = fm.FontProperties()
            else:
                # Windows以外の環境
                self.font_prop = fm.FontProperties()
                
        except Exception as e:
            print(f"フォント設定エラー: {e}")
            self.font_prop = fm.FontProperties()
            
        print("フォント設定完了")
    
    def _define_track_characteristics(self):
        """
        競馬場の物理的特徴を定義
        
        ユーザー提供情報:
        - 中山、阪神：坂がきつい
        - 中京：急カーブかつ下り坂（外を回されると膨らみやすくかなりきつい）
        - 東京、阪神、京都で実績ありは札幌でも実績出しやすい
        """
        self.track_characteristics = {
            '中山': {
                'slope_difficulty': 0.9,    # 坂の厳しさ (0-1)
                'curve_tightness': 0.6,     # カーブの急さ
                'bias_impact': 0.7,         # バイアスの影響度
                'stamina_demand': 0.8,      # スタミナ要求度
                'speed_sustainability': 0.7, # スピード持続性要求
                'outside_disadvantage': 0.6, # 外枠不利度
                'track_type': 'power',      # パワー型
                'similar_tracks': ['札幌']   # 類似コース
            },
            '阪神': {
                'slope_difficulty': 0.9,
                'curve_tightness': 0.7,
                'bias_impact': 0.8,
                'stamina_demand': 0.8,
                'speed_sustainability': 0.8,
                'outside_disadvantage': 0.7,
                'track_type': 'power',
                'similar_tracks': ['札幌']
            },
            '中京': {
                'slope_difficulty': 0.8,    # 下り坂
                'curve_tightness': 0.95,    # 急カーブ（最高レベル）
                'bias_impact': 0.9,         # 外に膨らみやすい
                'stamina_demand': 0.7,
                'speed_sustainability': 0.6,
                'outside_disadvantage': 0.9, # 外枠非常に不利
                'track_type': 'technical',   # 技術型
                'similar_tracks': []
            },
            '東京': {
                'slope_difficulty': 0.5,
                'curve_tightness': 0.4,
                'bias_impact': 0.6,
                'stamina_demand': 0.7,
                'speed_sustainability': 0.8,
                'outside_disadvantage': 0.5,
                'track_type': 'speed',      # スピード型
                'similar_tracks': ['札幌']
            },
            '京都': {
                'slope_difficulty': 0.6,
                'curve_tightness': 0.5,
                'bias_impact': 0.7,
                'stamina_demand': 0.7,
                'speed_sustainability': 0.8,
                'outside_disadvantage': 0.6,
                'track_type': 'balanced',   # バランス型
                'similar_tracks': ['札幌']
            },
            '新潟': {
                'slope_difficulty': 0.3,
                'curve_tightness': 0.2,     # 直線的
                'bias_impact': 0.4,
                'stamina_demand': 0.6,
                'speed_sustainability': 0.9,
                'outside_disadvantage': 0.2, # 外枠有利
                'track_type': 'speed',
                'similar_tracks': []
            },
            '福島': {
                'slope_difficulty': 0.4,
                'curve_tightness': 0.5,
                'bias_impact': 0.6,
                'stamina_demand': 0.6,
                'speed_sustainability': 0.7,
                'outside_disadvantage': 0.5,
                'track_type': 'balanced',
                'similar_tracks': []
            },
            '函館': {
                'slope_difficulty': 0.3,
                'curve_tightness': 0.6,
                'bias_impact': 0.5,
                'stamina_demand': 0.5,
                'speed_sustainability': 0.7,
                'outside_disadvantage': 0.4,
                'track_type': 'speed',
                'similar_tracks': []
            },
            '小倉': {
                'slope_difficulty': 0.4,
                'curve_tightness': 0.7,
                'bias_impact': 0.6,
                'stamina_demand': 0.6,
                'speed_sustainability': 0.7,
                'outside_disadvantage': 0.6,
                'track_type': 'balanced',
                'similar_tracks': []
            },
            '札幌': {
                'slope_difficulty': 0.4,
                'curve_tightness': 0.5,
                'bias_impact': 0.5,
                'stamina_demand': 0.6,
                'speed_sustainability': 0.8,
                'outside_disadvantage': 0.4,
                'track_type': 'speed',
                'similar_tracks': ['東京', '阪神', '京都'] # 類似実績反映
            }
        }
    
    def load_and_preprocess_data(self):
        """
        SEDファイル群からデータを読み込み、前処理を実行
        """
        print("データ読み込み・前処理を開始...")
        
        # SEDファイルの読み込み
        sed_files = glob.glob(os.path.join(self.data_folder, "SED*_formatted_with_bias.csv"))
        
        if not sed_files:
            print(f"エラー: {self.data_folder} にSEDファイルが見つかりません。")
            return False
        
        print(f"見つかったSEDファイル: {len(sed_files)}個")
        
        data_list = []
        for file_path in sed_files[:50]:  # 処理時間短縮のため最初の50ファイル
            try:
                for encoding in ['utf-8', 'shift-jis', 'cp932']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        data_list.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"ファイル読み込みエラー: {file_path} - {e}")
        
        if not data_list:
            print("データの読み込みに失敗しました。")
            return False
        
        # データ結合
        self.df = pd.concat(data_list, ignore_index=True)
        print(f"総データ数: {len(self.df)}行")
        
        # 前処理実行
        return self._preprocess_data()
    
    def _preprocess_data(self):
        """データの前処理"""
        print("データ前処理を実行中...")
        
        # 必要なカラムの確認
        required_columns = ['場名', '年', '馬番', '着順', 'IDM', '素点']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"エラー: 必要なカラムが不足: {missing_columns}")
            return False
        
        # データ型変換
        numeric_columns = ['年', '馬番', '着順', 'IDM', '素点', 'テン指数', '上がり指数', 'ペース指数', '距離', '馬体重']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 勝利フラグ作成
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        
        # 馬の総合能力値算出
        self._calculate_horse_ability_scores()
        
        # 競馬場特徴量追加
        self._add_track_features()
        
        # 適性スコア算出
        self._calculate_track_aptitude()
        
        # 無効データ除去
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['年', '馬番', '着順', '場名', '総合能力値'])
        after_count = len(self.df)
        
        print(f"データクリーニング: {before_count}行 → {after_count}行")
        print(f"分析対象競馬場: {list(self.df['場名'].unique())}")
        
        return True
    
    def _calculate_horse_ability_scores(self):
        """
        馬の総合能力値を算出
        複数の指数を統合して馬の総合力を数値化
        """
        print("馬の総合能力値を算出中...")
        
        # 基本能力値（IDM、素点をベース）
        self.df['基本能力値'] = (
            self.df['IDM'].fillna(self.df['IDM'].median()) * 0.4 +
            self.df['素点'].fillna(self.df['素点'].median()) * 0.6
        )
        
        # スピード能力値
        self.df['スピード能力値'] = (
            self.df['テン指数'].fillna(self.df['テン指数'].median()) * 0.6 +
            self.df['上がり指数'].fillna(self.df['上がり指数'].median()) * 0.4
        )
        
        # スタミナ能力値（距離適性を考慮）
        self.df['スタミナ能力値'] = (
            self.df['基本能力値'] * 0.7 +
            (self.df['距離'] / 2000) * self.df['ペース指数'].fillna(self.df['ペース指数'].median()) * 0.3
        )
        
        # 総合能力値（重み付き平均）
        self.df['総合能力値'] = (
            self.df['基本能力値'] * 0.5 +
            self.df['スピード能力値'] * 0.3 +
            self.df['スタミナ能力値'] * 0.2
        )
        
        print("能力値算出完了")
    
    def _add_track_features(self):
        """競馬場の特徴量をデータに追加"""
        print("競馬場特徴量を追加中...")
        
        # 競馬場特徴量を追加
        for feature in ['slope_difficulty', 'curve_tightness', 'bias_impact', 
                       'stamina_demand', 'speed_sustainability', 'outside_disadvantage']:
            self.df[f'track_{feature}'] = self.df['場名'].map(
                lambda x: self.track_characteristics.get(x, {}).get(feature, 0.5)
            )
        
        # コースタイプのエンコーディング
        track_type_map = {'speed': 0, 'balanced': 1, 'power': 2, 'technical': 3}
        self.df['track_type_encoded'] = self.df['場名'].map(
            lambda x: track_type_map.get(
                self.track_characteristics.get(x, {}).get('track_type', 'balanced'), 1
            )
        )
        
        print("競馬場特徴量追加完了")
    
    def _calculate_track_aptitude(self):
        """
        馬の競馬場適性スコアを算出
        馬の能力特性と競馬場要求能力のマッチング度を数値化
        """
        print("競馬場適性スコアを算出中...")
        
        # スピード適性
        self.df['スピード適性'] = (
            self.df['スピード能力値'] * (1 - self.df['track_slope_difficulty']) * 
            (1 - self.df['track_curve_tightness'])
        )
        
        # パワー適性
        self.df['パワー適性'] = (
            self.df['基本能力値'] * self.df['track_slope_difficulty'] * 
            self.df['track_stamina_demand']
        )
        
        # 技術適性
        self.df['技術適性'] = (
            self.df['総合能力値'] * self.df['track_curve_tightness'] * 
            (1 - self.df['track_bias_impact'])
        )
        
        # 枠順適性（馬番による影響）
        self.df['枠順適性'] = (
            1 - (self.df['馬番'] - 1) / 17 * self.df['track_outside_disadvantage']
        )
        
        # 総合適性スコア
        self.df['総合適性スコア'] = (
            self.df['スピード適性'] * 0.3 +
            self.df['パワー適性'] * 0.3 +
            self.df['技術適性'] * 0.2 +
            self.df['枠順適性'] * 0.2
        )
        
        # 類似競馬場での実績を考慮
        self._add_similar_track_performance()
        
        print("適性スコア算出完了")
    
    def _add_similar_track_performance(self):
        """類似競馬場での実績を適性スコアに反映"""
        print("類似競馬場実績を反映中...")
        
        # 馬ごとの競馬場別実績を集計
        horse_track_performance = self.df.groupby(['血統登録番号', '場名']).agg({
            '勝利': 'mean',
            '着順': 'mean',
            '総合能力値': 'mean'
        }).reset_index()
        
        # 類似競馬場での実績を加味
        for track, features in self.track_characteristics.items():
            similar_tracks = features.get('similar_tracks', [])
            if similar_tracks:
                track_mask = self.df['場名'] == track
                
                for _, row in self.df[track_mask].iterrows():
                    horse_id = row['血統登録番号']
                    
                    # 類似競馬場での平均勝率を取得
                    similar_performance = horse_track_performance[
                        (horse_track_performance['血統登録番号'] == horse_id) &
                        (horse_track_performance['場名'].isin(similar_tracks))
                    ]['勝利'].mean()
                    
                    if not pd.isna(similar_performance):
                        # 適性スコアに類似競馬場実績を反映
                        adjustment_factor = 1 + (similar_performance - 0.1) * 0.3
                        self.df.loc[row.name, '総合適性スコア'] *= adjustment_factor
        
        print("類似競馬場実績反映完了")
    
    def analyze_track_aptitude_correlation(self):
        """
        競馬場ごとの勝率と馬の適性について相関分析
        相関係数、散布図、p値、回帰直線、ロジスティック回帰を実施
        """
        print("\n=== 競馬場別適性相関分析開始 ===")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # 1. 競馬場別統計分析
        track_correlation_stats = self._calculate_track_correlation_statistics()
        
        # 2. 散布図と回帰分析の可視化
        self._create_aptitude_correlation_visualizations(track_correlation_stats)
        
        # 3. ロジスティック回帰分析
        logistic_results = self._perform_logistic_regression_analysis()
        
        # 4. 詳細統計レポート生成
        self._generate_aptitude_correlation_report(track_correlation_stats, logistic_results)
        
        return {
            'correlation_stats': track_correlation_stats,
            'logistic_results': logistic_results
        }
    
    def _calculate_track_correlation_statistics(self):
        """競馬場別の適性と勝率の相関統計を計算"""
        print("競馬場別相関統計計算中...")
        
        track_stats = {}
        
        for track in self.df['場名'].unique():
            track_data = self.df[self.df['場名'] == track].copy()
            
            if len(track_data) < 50:  # 最低サンプルサイズ
                print(f"警告: {track}のサンプルサイズが少ない({len(track_data)})")
                continue
            
            # 適性スコアと勝率のデータ
            aptitude_scores = track_data['総合適性スコア'].values
            win_flags = track_data['勝利'].values
            win_rates = track_data.groupby('総合適性スコア')['勝利'].transform('mean').values
            
            # ピアソン相関係数とp値
            correlation, p_value = stats.pearsonr(aptitude_scores, win_flags)
            
            # スピアマンの順位相関係数
            spearman_corr, spearman_p = stats.spearmanr(aptitude_scores, win_flags)
            
            # 線形回帰
            X = aptitude_scores.reshape(-1, 1)
            y = win_flags
            
            reg_model = LinearRegression()
            reg_model.fit(X, y)
            
            y_pred = reg_model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # 回帰直線の係数
            slope = reg_model.coef_[0]
            intercept = reg_model.intercept_
            
            # 統計的有意性判定
            is_significant = p_value < 0.05
            significance_level = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
            
            # 適性スコアの分布統計
            aptitude_stats = {
                'mean': aptitude_scores.mean(),
                'std': aptitude_scores.std(),
                'min': aptitude_scores.min(),
                'max': aptitude_scores.max(),
                'q25': np.percentile(aptitude_scores, 25),
                'median': np.median(aptitude_scores),
                'q75': np.percentile(aptitude_scores, 75)
            }
            
            track_stats[track] = {
                'sample_size': len(track_data),
                'correlation': correlation,
                'p_value': p_value,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'r_squared': r2,
                'slope': slope,
                'intercept': intercept,
                'win_rate_mean': win_flags.mean(),
                'win_rate_std': win_flags.std(),
                'is_significant': is_significant,
                'significance_level': significance_level,
                'aptitude_data': aptitude_scores,
                'win_data': win_flags,
                'aptitude_stats': aptitude_stats
            }
        
        return track_stats
    
    def _create_aptitude_correlation_visualizations(self, track_stats):
        """適性相関関係の可視化"""
        print("適性相関関係可視化作成中...")
        
        # 競馬場数に応じてサブプロット配置を決定
        n_tracks = len(track_stats)
        n_cols = 3
        n_rows = (n_tracks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('競馬場別 馬適性×勝率 相関分析', fontproperties=self.font_prop, fontsize=16)
        
        # axesが1次元の場合に対応
        if n_tracks == 1:
            axes = axes.reshape(1, -1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        track_names = list(track_stats.keys())
        
        for i, track in enumerate(track_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            stats_data = track_stats[track]
            
            # 散布図（適性スコアでビニング）
            aptitude_bins = np.linspace(stats_data['aptitude_stats']['min'], 
                                      stats_data['aptitude_stats']['max'], 20)
            bin_indices = np.digitize(stats_data['aptitude_data'], aptitude_bins)
            
            # ビンごとの勝率を計算
            bin_win_rates = []
            bin_centers = []
            for bin_idx in range(1, len(aptitude_bins)):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_win_rates.append(stats_data['win_data'][mask].mean())
                    bin_centers.append((aptitude_bins[bin_idx-1] + aptitude_bins[bin_idx]) / 2)
            
            # 散布図
            ax.scatter(stats_data['aptitude_data'], stats_data['win_data'], 
                      alpha=0.3, s=20, color='lightblue', label='個別データ')
            
            # ビン集計の勝率
            if bin_centers:
                ax.scatter(bin_centers, bin_win_rates, 
                          s=100, color='darkblue', marker='o', label='ビン平均勝率')
            
            # 回帰直線
            x_line = np.linspace(stats_data['aptitude_stats']['min'], 
                               stats_data['aptitude_stats']['max'], 100)
            y_line = stats_data['slope'] * x_line + stats_data['intercept']
            ax.plot(x_line, y_line, color='red', linewidth=2, 
                   label=f'回帰直線 (R²={stats_data["r_squared"]:.3f})')
            
            # タイトルと統計情報
            title = (f'{track}\n'
                    f'相関係数: {stats_data["correlation"]:.3f}{stats_data["significance_level"]} '
                    f'(p={stats_data["p_value"] if stats_data["p_value"] > 0 else 0:.4f})\n'
                    f'スピアマン: {stats_data["spearman_correlation"]:.3f}')
            ax.set_title(title, fontproperties=self.font_prop, fontsize=10)
            
            ax.set_xlabel('総合適性スコア', fontproperties=self.font_prop, fontsize=9)
            ax.set_ylabel('勝利フラグ', fontproperties=self.font_prop, fontsize=9)
            ax.legend(prop=self.font_prop, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        
        # 空のサブプロットを非表示
        for i in range(len(track_names), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '競馬場別適性相関分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 相関係数比較チャート
        self._create_aptitude_correlation_comparison_chart(track_stats)
    
    def _create_aptitude_correlation_comparison_chart(self, track_stats):
        """適性相関係数比較チャート"""
        
        # データ準備
        tracks = list(track_stats.keys())
        correlations = [track_stats[track]['correlation'] for track in tracks]
        spearman_correlations = [track_stats[track]['spearman_correlation'] for track in tracks]
        p_values = [track_stats[track]['p_value'] for track in tracks]
        r_squared = [track_stats[track]['r_squared'] for track in tracks]
        
        # 有意性による色分け
        colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray' 
                 for p in p_values]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('競馬場別適性相関分析 総合比較', fontproperties=self.font_prop, fontsize=16)
        
        # 1. ピアソン相関係数比較
        bars1 = ax1.barh(tracks, correlations, color=colors, alpha=0.7)
        ax1.set_xlabel('ピアソン相関係数', fontproperties=self.font_prop)
        ax1.set_title('競馬場別適性相関係数（ピアソン）', fontproperties=self.font_prop, fontsize=12)
        ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        for i, (bar, corr) in enumerate(zip(bars1, correlations)):
            ax1.text(corr + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', va='center', fontproperties=self.font_prop, fontsize=9)
        
        for label in ax1.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 2. スピアマン相関係数比較
        bars2 = ax2.barh(tracks, spearman_correlations, color=colors, alpha=0.7)
        ax2.set_xlabel('スピアマン相関係数', fontproperties=self.font_prop)
        ax2.set_title('競馬場別適性相関係数（スピアマン）', fontproperties=self.font_prop, fontsize=12)
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        for i, (bar, corr) in enumerate(zip(bars2, spearman_correlations)):
            ax2.text(corr + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', va='center', fontproperties=self.font_prop, fontsize=9)
        
        for label in ax2.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 3. p値比較（対数スケール）
        ax3.barh(tracks, [-np.log10(p) if p > 0 else 10 for p in p_values], color=colors, alpha=0.7)
        ax3.set_xlabel('-log10(p値)', fontproperties=self.font_prop)
        ax3.set_title('統計的有意性 (-log10(p値))', fontproperties=self.font_prop, fontsize=12)
        ax3.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.axvline(-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
        ax3.axvline(-np.log10(0.001), color='darkred', linestyle='--', alpha=0.7, label='p=0.001')
        ax3.legend(prop=self.font_prop)
        
        for label in ax3.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 4. R²値比較
        bars4 = ax4.barh(tracks, r_squared, color=colors, alpha=0.7)
        ax4.set_xlabel('決定係数 (R²)', fontproperties=self.font_prop)
        ax4.set_title('回帰モデル説明力 (R²値)', fontproperties=self.font_prop, fontsize=12)
        
        for i, (bar, r2) in enumerate(zip(bars4, r_squared)):
            ax4.text(r2 + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{r2:.3f}', va='center', fontproperties=self.font_prop, fontsize=9)
        
        for label in ax4.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '適性相関係数比較.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_logistic_regression_analysis(self):
        """ロジスティック回帰分析の実施"""
        print("ロジスティック回帰分析実行中...")
        
        logistic_results = {}
        
        for track in self.df['場名'].unique():
            track_data = self.df[self.df['場名'] == track].copy()
            
            if len(track_data) < 50:
                continue
            
            # 特徴量準備
            feature_cols = ['総合適性スコア', 'スピード適性', 'パワー適性', '技術適性', '枠順適性']
            X = track_data[feature_cols].fillna(track_data[feature_cols].median())
            y = track_data['勝利']
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # ロジスティック回帰
            log_reg = LogisticRegression(random_state=42, max_iter=1000)
            log_reg.fit(X_train_scaled, y_train)
            
            # 予測と評価
            y_pred = log_reg.predict(X_test_scaled)
            y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
            
            # 評価指標
            accuracy = log_reg.score(X_test_scaled, y_test)
            
            # 係数
            coefficients = dict(zip(feature_cols, log_reg.coef_[0]))
            
            logistic_results[track] = {
                'accuracy': accuracy,
                'coefficients': coefficients,
                'intercept': log_reg.intercept_[0],
                'sample_size': len(track_data),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'actual': y_test,
                'feature_importance': abs(log_reg.coef_[0])
            }
        
        # ロジスティック回帰結果の可視化
        self._visualize_logistic_regression_results(logistic_results)
        
        return logistic_results
    
    def _visualize_logistic_regression_results(self, logistic_results):
        """ロジスティック回帰結果の可視化"""
        print("ロジスティック回帰結果可視化中...")
        
        # 係数比較のヒートマップ
        tracks = list(logistic_results.keys())
        feature_names = ['総合適性スコア', 'スピード適性', 'パワー適性', '技術適性', '枠順適性']
        
        coef_matrix = []
        for track in tracks:
            coef_matrix.append([logistic_results[track]['coefficients'][feat] for feat in feature_names])
        
        coef_df = pd.DataFrame(coef_matrix, index=tracks, columns=feature_names)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(coef_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
        plt.title('競馬場別ロジスティック回帰係数', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('特徴量', fontproperties=self.font_prop, fontsize=12)
        plt.ylabel('競馬場', fontproperties=self.font_prop, fontsize=12)
        
        # 軸ラベルにフォント適用
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'ロジスティック回帰係数.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 精度比較
        accuracies = [logistic_results[track]['accuracy'] for track in tracks]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tracks, accuracies, alpha=0.7, color='steelblue')
        plt.title('競馬場別ロジスティック回帰精度', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('競馬場', fontproperties=self.font_prop, fontsize=12)
        plt.ylabel('精度', fontproperties=self.font_prop, fontsize=12)
        plt.xticks(rotation=45)
        
        # 精度の値を表示
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom', fontproperties=self.font_prop, fontsize=9)
        
        # x軸ラベルにフォント適用
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'ロジスティック回帰精度.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_aptitude_correlation_report(self, track_stats, logistic_results):
        """適性相関分析レポートの生成"""
        report_path = os.path.join(self.output_folder, '競馬場別適性相関分析レポート.md')
        
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write("# 競馬場別 馬適性×勝率 相関分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 分析目的\n\n")
            f.write("各競馬場において、馬の総合適性スコアと勝率の間の相関関係を\n")
            f.write("多角的に分析し、競馬場ごとの特性を明らかにしました。\n\n")
            
            f.write("## 📊 相関分析結果\n\n")
            f.write("| 競馬場 | サンプル数 | ピアソン相関 | スピアマン相関 | p値 | R² | 有意性 |\n")
            f.write("|--------|------------|--------------|----------------|-----|----|---------|\n")
            
            # 相関係数の高い順にソート
            sorted_tracks = sorted(track_stats.items(), 
                                 key=lambda x: x[1]['correlation'], reverse=True)
            
            for track, stats in sorted_tracks:
                f.write(f"| {track} | {stats['sample_size']} | "
                       f"{stats['correlation']:.3f} | {stats['spearman_correlation']:.3f} | "
                       f"{stats['p_value']:.4f} | {stats['r_squared']:.3f} | "
                       f"{stats['significance_level']} |\n")
            
            f.write("\n**有意性の記号**\n")
            f.write("- *** : p < 0.001 (極めて高い有意性)\n")
            f.write("- ** : p < 0.01 (高い有意性)\n") 
            f.write("- * : p < 0.05 (有意)\n")
            f.write("- n.s. : p ≥ 0.05 (有意差なし)\n\n")
            
            f.write("## 🤖 ロジスティック回帰分析結果\n\n")
            f.write("| 競馬場 | 精度 | 総合適性係数 | スピード係数 | パワー係数 | 技術係数 | 枠順係数 |\n")
            f.write("|--------|------|--------------|--------------|------------|----------|----------|\n")
            
            for track, results in logistic_results.items():
                coef = results['coefficients']
                f.write(f"| {track} | {results['accuracy']:.3f} | "
                       f"{coef['総合適性スコア']:.3f} | {coef['スピード適性']:.3f} | "
                       f"{coef['パワー適性']:.3f} | {coef['技術適性']:.3f} | "
                       f"{coef['枠順適性']:.3f} |\n")
            
            f.write("\n## 🏆 主要な発見\n\n")
            
            # 最高・最低相関
            highest_track, highest_stats = max(track_stats.items(), 
                                             key=lambda x: x[1]['correlation'])
            lowest_track, lowest_stats = min(track_stats.items(), 
                                           key=lambda x: x[1]['correlation'])
            
            f.write(f"### 適性と勝率の相関が最も強い競馬場\n")
            f.write(f"- **{highest_track}**: ピアソン相関係数 {highest_stats['correlation']:.3f}\n")
            f.write(f"  - スピアマン相関係数: {highest_stats['spearman_correlation']:.3f}\n")
            f.write(f"  - p値: {highest_stats['p_value']:.4f}\n")
            f.write(f"  - R²: {highest_stats['r_squared']:.3f}\n\n")
            
            f.write(f"### 適性と勝率の相関が最も弱い競馬場\n")
            f.write(f"- **{lowest_track}**: ピアソン相関係数 {lowest_stats['correlation']:.3f}\n")
            f.write(f"  - スピアマン相関係数: {lowest_stats['spearman_correlation']:.3f}\n")
            f.write(f"  - p値: {lowest_stats['p_value']:.4f}\n")
            f.write(f"  - R²: {lowest_stats['r_squared']:.3f}\n\n")
            
            # ロジスティック回帰の最高精度
            best_logistic = max(logistic_results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"### ロジスティック回帰で最も予測精度が高い競馬場\n")
            f.write(f"- **{best_logistic[0]}**: 精度 {best_logistic[1]['accuracy']:.3f}\n\n")
            
            # 有意性による分類
            significant_tracks = [track for track, stats in track_stats.items() 
                                if stats['is_significant']]
            non_significant_tracks = [track for track, stats in track_stats.items() 
                                    if not stats['is_significant']]
            
            f.write(f"### 統計的有意性の結果\n")
            f.write(f"- **有意な相関がある競馬場** ({len(significant_tracks)}場): {', '.join(significant_tracks)}\n")
            f.write(f"- **有意な相関がない競馬場** ({len(non_significant_tracks)}場): {', '.join(non_significant_tracks) if non_significant_tracks else 'なし'}\n\n")
            
            f.write("## 💡 実用的解釈\n\n")
            
            f.write("### 適性スコアの有効性\n")
            avg_correlation = np.mean([stats['correlation'] for stats in track_stats.values()])
            f.write(f"- **全競馬場平均相関係数**: {avg_correlation:.3f}\n")
            
            if avg_correlation > 0.3:
                f.write("- 馬の適性スコアは勝率予測に**非常に有効**\n")
            elif avg_correlation > 0.2:
                f.write("- 馬の適性スコアは勝率予測に**有効**\n")
            elif avg_correlation > 0.1:
                f.write("- 馬の適性スコアは勝率予測に**やや有効**\n")
            else:
                f.write("- 馬の適性スコアは勝率予測への寄与が**限定的**\n")
            
            f.write("\n### 競馬場別の予測戦略\n")
            
            # 相関の強い競馬場
            strong_corr_tracks = [track for track, stats in track_stats.items() 
                                if stats['correlation'] > avg_correlation + 0.1]
            weak_corr_tracks = [track for track, stats in track_stats.items() 
                              if stats['correlation'] < avg_correlation - 0.1]
            
            f.write(f"- **適性重視型競馬場** (相関が強い): {', '.join(strong_corr_tracks)}\n")
            f.write(f"  → 馬の適性スコアを重視した予想が有効\n")
            f.write(f"- **多要因型競馬場** (相関が弱い): {', '.join(weak_corr_tracks)}\n")
            f.write(f"  → 適性以外の要因（展開、馬場状態等）も考慮が必要\n\n")
            
            f.write("### 実践的アドバイス\n")
            f.write("1. **高相関競馬場**では適性スコアによる予想の信頼性が高い\n")
            f.write("2. **低相関競馬場**では他の要因との複合的な分析が必要\n")
            f.write("3. **ロジスティック回帰の係数**から各適性要素の重要度を判断\n")
            f.write("4. **スピアマン相関**が高い場合は順位関係の予測に有効\n\n")
        
        print(f"適性相関分析レポート保存: {report_path}")
    
    def analyze_track_horse_compatibility(self):
        """
        競馬場と馬の相性を詳細分析
        """
        print("競馬場×馬相性分析を実行中...")
        
        # 競馬場別の平均能力値と勝率
        track_analysis = self.df.groupby('場名').agg({
            '総合能力値': ['mean', 'std'],
            '総合適性スコア': ['mean', 'std'],
            '勝利': 'mean',
            '着順': 'mean'
        }).round(4)
        
        # 能力値レンジ別の勝率分析
        self.df['能力値ランク'] = pd.qcut(self.df['総合能力値'], q=5, labels=['D', 'C', 'B', 'A', 'S'])
        
        ability_track_analysis = self.df.groupby(['場名', '能力値ランク']).agg({
            '勝利': 'mean',
            '着順': 'mean',
            '総合適性スコア': 'mean'
        }).reset_index()
        
        # 可視化
        self._create_compatibility_visualizations(track_analysis, ability_track_analysis)
        
        # レポート生成
        self._generate_compatibility_report(track_analysis, ability_track_analysis)
        
        print("競馬場×馬相性分析完了")
    
    def _create_compatibility_visualizations(self, track_analysis, ability_analysis):
        """相性分析の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('競馬場×馬能力適性分析', fontproperties=self.font_prop, fontsize=16)
        
        # 1. 競馬場別平均適性スコア
        ax1 = axes[0, 0]
        tracks = track_analysis.index
        scores = track_analysis[('総合適性スコア', 'mean')]
        bars = ax1.bar(tracks, scores, alpha=0.7)
        ax1.set_title('競馬場別平均適性スコア', fontproperties=self.font_prop, fontsize=12)
        ax1.set_ylabel('適性スコア', fontproperties=self.font_prop, fontsize=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        for label in ax1.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 2. 競馬場別勝率vs適性スコア
        ax2 = axes[0, 1]
        win_rates = track_analysis[('勝利', 'mean')]
        ax2.scatter(scores, win_rates, s=100, alpha=0.7)
        for i, track in enumerate(tracks):
            ax2.annotate(track, (scores.iloc[i], win_rates.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontproperties=self.font_prop, fontsize=9)
        ax2.set_xlabel('平均適性スコア', fontproperties=self.font_prop, fontsize=10)
        ax2.set_ylabel('平均勝率', fontproperties=self.font_prop, fontsize=10)
        ax2.set_title('適性スコア vs 勝率', fontproperties=self.font_prop, fontsize=12)
        
        # 3. 能力ランク別勝率（ヒートマップ）
        ax3 = axes[1, 0]
        heatmap_data = ability_analysis.pivot(index='場名', columns='能力値ランク', values='勝利')
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax3, fmt='.3f')
        ax3.set_title('能力ランク別勝率', fontproperties=self.font_prop, fontsize=12)
        ax3.set_xlabel('能力値ランク', fontproperties=self.font_prop, fontsize=10)
        ax3.set_ylabel('競馬場', fontproperties=self.font_prop, fontsize=10)
        
        # ヒートマップの軸ラベルにフォント適用
        for label in ax3.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        for label in ax3.get_yticklabels():
            label.set_fontproperties(self.font_prop)
        
        # 4. 競馬場特徴比較（バーチャート）
        ax4 = axes[1, 1]
        self._create_track_comparison_chart(ax4, ['中山', '中京', '東京', '新潟'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '競馬場適性分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 別途レーダーチャートを作成
        self._create_radar_chart_separately(['中山', '中京', '東京', '新潟'])
    
    def _create_track_comparison_chart(self, ax, tracks):
        """競馬場特徴の比較チャート作成（バーチャート版）"""
        categories = ['坂難易度', 'カーブ急度', 'バイアス影響', 'スタミナ要求', 'スピード持続', '外枠不利']
        
        x = np.arange(len(categories))
        width = 0.2
        
        for i, track in enumerate(tracks):
            if track in self.track_characteristics:
                values = [
                    self.track_characteristics[track]['slope_difficulty'],
                    self.track_characteristics[track]['curve_tightness'],
                    self.track_characteristics[track]['bias_impact'],
                    self.track_characteristics[track]['stamina_demand'],
                    self.track_characteristics[track]['speed_sustainability'],
                    self.track_characteristics[track]['outside_disadvantage']
                ]
                
                ax.bar(x + i * width, values, width, label=track, alpha=0.8)
        
        ax.set_xlabel('競馬場特徴', fontproperties=self.font_prop, fontsize=10)
        ax.set_ylabel('数値', fontproperties=self.font_prop, fontsize=10)
        ax.set_title('競馬場特徴比較', fontproperties=self.font_prop, fontsize=12)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(categories, rotation=45, fontproperties=self.font_prop, fontsize=9)
        
        # 凡例のフォント設定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontproperties(self.font_prop)
            
        ax.set_ylim(0, 1)
    
    def _create_radar_chart_separately(self, tracks):
        """競馬場特徴のレーダーチャートを別途作成"""
        categories = ['坂難易度', 'カーブ急度', 'バイアス影響', 'スタミナ要求', 'スピード持続', '外枠不利']
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # カテゴリラベルにフォント適用（正しいメソッド使用）
        try:
            for label in ax.get_xticklabels():
                if hasattr(label, 'set_fontproperties'):
                    label.set_fontproperties(self.font_prop)
                    label.set_fontsize(10)
        except Exception as e:
            print(f"ラベルフォント設定エラー: {e}")
        
        for track in tracks:
            if track in self.track_characteristics:
                values = [
                    self.track_characteristics[track]['slope_difficulty'],
                    self.track_characteristics[track]['curve_tightness'],
                    self.track_characteristics[track]['bias_impact'],
                    self.track_characteristics[track]['stamina_demand'],
                    self.track_characteristics[track]['speed_sustainability'],
                    self.track_characteristics[track]['outside_disadvantage']
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=track)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_ylim(0, 1)
        ax.set_title('競馬場特徴レーダーチャート', fontproperties=self.font_prop, fontsize=14, pad=20)
        
        # 凡例のフォント設定
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        for text in legend.get_texts():
            text.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '競馬場特徴レーダーチャート.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("レーダーチャート作成完了")
    
    def _generate_compatibility_report(self, track_analysis, ability_analysis):
        """相性分析レポートの生成"""
        report_path = os.path.join(self.output_folder, '競馬場適性分析レポート.md')
        
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write("# 競馬場×馬能力適性分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 分析概要\n\n")
            f.write("競馬場の物理的特徴と馬の能力特性を数値化し、相性を定量的に分析しました。\n\n")
            
            f.write("## 競馬場特徴定義\n\n")
            f.write("| 競馬場 | 坂難易度 | カーブ急度 | バイアス影響 | スタミナ要求 | スピード持続 | 外枠不利 | タイプ |\n")
            f.write("|--------|----------|------------|-------------|-------------|-------------|----------|--------|\n")
            
            for track, features in self.track_characteristics.items():
                f.write(f"| {track} | {features['slope_difficulty']:.2f} | "
                       f"{features['curve_tightness']:.2f} | {features['bias_impact']:.2f} | "
                       f"{features['stamina_demand']:.2f} | {features['speed_sustainability']:.2f} | "
                       f"{features['outside_disadvantage']:.2f} | {features['track_type']} |\n")
            
            f.write("\n## 競馬場別分析結果\n\n")
            f.write("| 競馬場 | 平均適性スコア | 平均勝率 | 平均着順 |\n")
            f.write("|--------|----------------|----------|----------|\n")
            
            for track in track_analysis.index:
                aptitude = track_analysis.loc[track, ('総合適性スコア', 'mean')]
                win_rate = track_analysis.loc[track, ('勝利', 'mean')]
                avg_rank = track_analysis.loc[track, ('着順', 'mean')]
                f.write(f"| {track} | {aptitude:.4f} | {win_rate:.4f} | {avg_rank:.2f} |\n")
            
            # 重要な発見
            f.write("\n## 重要な発見\n\n")
            
            best_aptitude_track = track_analysis[('総合適性スコア', 'mean')].idxmax()
            worst_aptitude_track = track_analysis[('総合適性スコア', 'mean')].idxmin()
            
            f.write(f"- **最高適性競馬場**: {best_aptitude_track}\n")
            f.write(f"- **最低適性競馬場**: {worst_aptitude_track}\n")
            
            # 能力ランク別の特徴
            high_ability_best = ability_analysis[ability_analysis['能力値ランク'] == 'S'].nlargest(3, '勝利')
            f.write(f"\n### 高能力馬（Sランク）が最も活躍する競馬場TOP3\n")
            for i, (_, row) in enumerate(high_ability_best.iterrows(), 1):
                f.write(f"{i}. {row['場名']}: 勝率{row['勝利']:.3f}\n")
        
        print(f"レポート保存: {report_path}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='競馬場特徴×馬能力適性分析')
    parser.add_argument('--data-folder', type=str, default="export/with_bias",
                       help='データフォルダのパス')
    parser.add_argument('--output-folder', type=str, default="results/track_horse_ability_analysis",
                       help='結果出力先フォルダのパス')
    
    args = parser.parse_args()
    
    # 分析器を初期化
    analyzer = TrackHorseAbilityAnalyzer(
        data_folder=args.data_folder,
        output_folder=args.output_folder
    )
    
    # データ読み込み・前処理
    if not analyzer.load_and_preprocess_data():
        print("データ読み込みに失敗しました。")
        return
    
    # 適性相関分析実行
    correlation_results = analyzer.analyze_track_aptitude_correlation()
    
    # 競馬場×馬相性分析
    analyzer.analyze_track_horse_compatibility()
    
    print(f"\n=== 分析完了 ===")
    print(f"結果保存先: {args.output_folder}")

if __name__ == "__main__":
    main() 
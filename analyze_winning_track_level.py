#!/usr/bin/env python
"""
競馬場レベル別勝率分析コマンドラインツール
馬ごとの重み付けポイントと勝率の関係を3年ごとに分析します。
複勝した場合のみポイントを加算する仕様に変更。
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import sys
import japanize_matplotlib
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
import copy
import random
warnings.filterwarnings('ignore')

# メインロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrackWinRateAnalyzer:
    """競馬場別勝率分析クラス（馬ごと重み付け・3年間隔分析・複勝時のみポイント加算）"""
    
    def __init__(self, config):
        """初期化"""
        self.config = config
        self.df = None
        self.track_hierarchy = {}
        self.original_track_hierarchy = {}  # オリジナルの重み付けを保存
        self.is_random_weights = config.get('random_weights', False)  # ランダム重み付けフラグ
        
        # 日本語フォント設定
        self._setup_japanese_font()
        
        # 競馬場階層の設定
        self._setup_track_hierarchy()
        
        # ランダム重み付けの場合は重みを変更
        if self.is_random_weights:
            self._apply_random_weights()
    
    def _setup_japanese_font(self):
        """日本語フォント設定"""
        plt.rcParams['font.family'] = 'MS Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
    
    def _setup_track_hierarchy(self):
        """競馬場の階層定義と重み付けポイントシステム"""
        self.track_hierarchy = {
            # 中央競馬 - 重み付けポイントシステム
            '東京': {
                'level': 10, 'type': '中央', 'category': '最高格式',
                'weight_points': 100,  # 最高格式G1多数開催
                'grade_weight': 10,    # G1レース価値
                'prestige_weight': 10, # 威信度
                'facility_weight': 10  # 設備・規模
            },
            '中山': {
                'level': 10, 'type': '中央', 'category': '最高格式',
                'weight_points': 95,
                'grade_weight': 9.5,
                'prestige_weight': 9.5,
                'facility_weight': 9.5
            },
            '京都': {
                'level': 9, 'type': '中央', 'category': '関西主要',
                'weight_points': 90,
                'grade_weight': 9,
                'prestige_weight': 9,
                'facility_weight': 9
            },
            '阪神': {
                'level': 9, 'type': '中央', 'category': '関西主要',
                'weight_points': 85,
                'grade_weight': 8.5,
                'prestige_weight': 8.5,
                'facility_weight': 8.5
            },
            '中京': {
                'level': 7, 'type': '中央', 'category': '中央地方',
                'weight_points': 70,
                'grade_weight': 7,
                'prestige_weight': 7,
                'facility_weight': 7
            },
            '新潟': {
                'level': 6, 'type': '中央', 'category': '中央地方',
                'weight_points': 60,
                'grade_weight': 6,
                'prestige_weight': 6,
                'facility_weight': 6
            },
            '小倉': {
                'level': 6, 'type': '中央', 'category': '中央地方',
                'weight_points': 58,
                'grade_weight': 5.8,
                'prestige_weight': 5.8,
                'facility_weight': 5.8
            },
            '福島': {
                'level': 5, 'type': '中央', 'category': '中央地方',
                'weight_points': 50,
                'grade_weight': 5,
                'prestige_weight': 5,
                'facility_weight': 5
            },
            '函館': {
                'level': 5, 'type': '中央', 'category': '中央地方',
                'weight_points': 48,
                'grade_weight': 4.8,
                'prestige_weight': 4.8,
                'facility_weight': 4.8
            },
            '札幌': {
                'level': 5, 'type': '中央', 'category': '中央地方',
                'weight_points': 45,
                'grade_weight': 4.5,
                'prestige_weight': 4.5,
                'facility_weight': 4.5
            },
            
            # 地方競馬 - NAR重賞価値を考慮
            '大井': {
                'level': 4, 'type': '地方', 'category': '首都圏主要',
                'weight_points': 40,  # 東京大賞典等G1開催
                'grade_weight': 4,
                'prestige_weight': 4,
                'facility_weight': 4
            },
            '川崎': {
                'level': 4, 'type': '地方', 'category': '首都圏主要',
                'weight_points': 38,  # 川崎記念等G1開催
                'grade_weight': 3.8,
                'prestige_weight': 3.8,
                'facility_weight': 3.8
            },
            '船橋': {
                'level': 3, 'type': '地方', 'category': '首都圏地方',
                'weight_points': 30,  # かしわ記念等G1開催
                'grade_weight': 3,
                'prestige_weight': 3,
                'facility_weight': 3
            },
            '浦和': {
                'level': 3, 'type': '地方', 'category': '首都圏地方',
                'weight_points': 28,
                'grade_weight': 2.8,
                'prestige_weight': 2.8,
                'facility_weight': 2.8
            },
            '園田': {
                'level': 3, 'type': '地方', 'category': '関西地方',
                'weight_points': 25,
                'grade_weight': 2.5,
                'prestige_weight': 2.5,
                'facility_weight': 2.5
            },
            '姫路': {
                'level': 2, 'type': '地方', 'category': '関西地方',
                'weight_points': 20,
                'grade_weight': 2,
                'prestige_weight': 2,
                'facility_weight': 2
            },
            '名古屋': {
                'level': 3, 'type': '地方', 'category': '中部地方',
                'weight_points': 28,
                'grade_weight': 2.8,
                'prestige_weight': 2.8,
                'facility_weight': 2.8
            },
            '笠松': {
                'level': 2, 'type': '地方', 'category': '中部地方',
                'weight_points': 18,
                'grade_weight': 1.8,
                'prestige_weight': 1.8,
                'facility_weight': 1.8
            },
            '金沢': {
                'level': 2, 'type': '地方', 'category': '北陸地方',
                'weight_points': 15,
                'grade_weight': 1.5,
                'prestige_weight': 1.5,
                'facility_weight': 1.5
            },
            '佐賀': {
                'level': 2, 'type': '地方', 'category': '九州地方',
                'weight_points': 15,
                'grade_weight': 1.5,
                'prestige_weight': 1.5,
                'facility_weight': 1.5
            },
            '高知': {
                'level': 2, 'type': '地方', 'category': '四国地方',
                'weight_points': 12,
                'grade_weight': 1.2,
                'prestige_weight': 1.2,
                'facility_weight': 1.2
            },
            '門別': {
                'level': 3, 'type': '地方', 'category': '北海道地方',
                'weight_points': 25,  # 生産地価値加算
                'grade_weight': 2.5,
                'prestige_weight': 2.5,
                'facility_weight': 2.5
            },
            '盛岡': {
                'level': 2, 'type': '地方', 'category': '東北地方',
                'weight_points': 12,
                'grade_weight': 1.2,
                'prestige_weight': 1.2,
                'facility_weight': 1.2
            },
            '水沢': {
                'level': 2, 'type': '地方', 'category': '東北地方',
                'weight_points': 10,
                'grade_weight': 1,
                'prestige_weight': 1,
                'facility_weight': 1
            },
            '宇都宮': {
                'level': 2, 'type': '地方', 'category': '関東地方',
                'weight_points': 10,
                'grade_weight': 1,
                'prestige_weight': 1,
                'facility_weight': 1
            },
            '足利': {
                'level': 1, 'type': '地方', 'category': '関東地方',
                'weight_points': 5,
                'grade_weight': 0.5,
                'prestige_weight': 0.5,
                'facility_weight': 0.5
            },
            '高崎': {
                'level': 1, 'type': '地方', 'category': '関東地方',
                'weight_points': 5,
                'grade_weight': 0.5,
                'prestige_weight': 0.5,
                'facility_weight': 0.5
            },
            '福山': {
                'level': 1, 'type': '地方', 'category': '中国地方',
                'weight_points': 3,
                'grade_weight': 0.3,
                'prestige_weight': 0.3,
                'facility_weight': 0.3
            },
            '益田': {
                'level': 1, 'type': '地方', 'category': '中国地方',
                'weight_points': 2,
                'grade_weight': 0.2,
                'prestige_weight': 0.2,
                'facility_weight': 0.2
            }
        }
    
    def _apply_random_weights(self):
        """競馬場の重み付けをランダムに変更"""
        import random
        
        logger.info("🎲 競馬場重み付けをランダムに変更中...")
        
        # オリジナルの重み付けを保存
        self.original_track_hierarchy = copy.deepcopy(self.track_hierarchy)
        
        # 全競馬場のリストを取得
        track_names = list(self.track_hierarchy.keys())
        
        # オリジナルの重み付けポイントを取得してシャッフル
        original_points = [info['weight_points'] for info in self.track_hierarchy.values()]
        random.shuffle(original_points)
        
        # シャッフルされたポイントを各競馬場に再割り当て
        for i, track_name in enumerate(track_names):
            new_weight = original_points[i]
            self.track_hierarchy[track_name]['weight_points'] = new_weight
            
            # 他の重みも比例して調整
            ratio = new_weight / self.original_track_hierarchy[track_name]['weight_points']
            self.track_hierarchy[track_name]['grade_weight'] = round(
                self.original_track_hierarchy[track_name]['grade_weight'] * ratio, 2)
            self.track_hierarchy[track_name]['prestige_weight'] = round(
                self.original_track_hierarchy[track_name]['prestige_weight'] * ratio, 2)
            self.track_hierarchy[track_name]['facility_weight'] = round(
                self.original_track_hierarchy[track_name]['facility_weight'] * ratio, 2)
        
        # ランダム化の結果をログ出力
        logger.info("🎲 ランダム重み付け結果（抜粋）:")
        for track_name in ['東京', '中山', '大井', '川崎']:
            if track_name in self.track_hierarchy:
                original = self.original_track_hierarchy[track_name]['weight_points']
                new = self.track_hierarchy[track_name]['weight_points']
                logger.info(f"  {track_name}: {original} → {new}")
    
    def load_data(self):
        """データ読み込み"""
        logger.info("データ読み込み開始...")
        
        input_path = Path(self.config['input_path'])
        
        if input_path.is_file():
            df = self._read_csv_file(input_path)
        elif input_path.is_dir():
            csv_files = list(input_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"CSVファイルが見つかりません: {input_path}")
            
            df_list = []
            for file_path in csv_files:
                try:
                    df_temp = self._read_csv_file(file_path)
                    df_list.append(df_temp)
                except Exception as e:
                    logger.warning(f"ファイル読み込みエラー: {file_path} - {e}")
            
            if not df_list:
                raise ValueError("有効なCSVファイルがありません")
            
            df = pd.concat(df_list, ignore_index=True)
        else:
            raise FileNotFoundError(f"指定されたパスが存在しません: {input_path}")
        
        logger.info(f"データ読み込み完了: {len(df)}行")
        return df
    
    def _read_csv_file(self, file_path):
        """CSVファイル読み込み（エンコーディング自動判定）"""
        encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp']
        
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"ファイルを読み込めませんでした: {file_path}")
    
    def preprocess_data(self):
        """データ前処理"""
        logger.info("データ前処理開始...")
        
        # 必要カラムの確認
        required_columns = ['馬名', '場名', '着順']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"必要なカラムが不足しています: {missing_columns}")
        
        # 数値変換
        self.df['着順'] = pd.to_numeric(self.df['着順'], errors='coerce')
        
        # 勝利・複勝フラグ
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        self.df['複勝'] = (self.df['着順'] <= 3).astype(int)
        
        # 競馬場情報の追加
        self.df['競馬場レベル'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('level', 0)
        )
        self.df['競馬場タイプ'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('type', '不明')
        )
        self.df['競馬場カテゴリ'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('category', '不明')
        )
        
        # 競馬場の重み付けポイント情報の追加（後で馬ごとに集計）
        self.df['競馬場重み付けポイント'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('weight_points', 0)
        )
        self.df['競馬場グレード重み'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('grade_weight', 0)
        )
        self.df['競馬場威信度重み'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('prestige_weight', 0)
        )
        self.df['競馬場設備重み'] = self.df['場名'].map(
            lambda x: self.track_hierarchy.get(x, {}).get('facility_weight', 0)
        )
        
        # 年の処理（データに年があるかチェック）
        self._process_year_data()
        
        # データクリーニング
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['馬名', '場名', '着順'])
        self.df = self.df[self.df['競馬場レベル'] > 0]  # 未定義競馬場を除外
        after_count = len(self.df)
        
        logger.info(f"データクリーニング: {before_count}行 → {after_count}行")
        logger.info(f"対象競馬場: {sorted(self.df['場名'].unique())}")
        
        return self.df
    
    def _process_year_data(self):
        """年データの処理"""
        # 年カラムの検出
        year_columns = [col for col in self.df.columns if '年' in col.lower() or 'year' in col.lower()]
        date_columns = [col for col in self.df.columns if '日付' in col or 'date' in col.lower() or '開催日' in col]
        
        if year_columns:
            self.df['年'] = pd.to_numeric(self.df[year_columns[0]], errors='coerce')
            logger.info(f"年カラムを検出しました: {year_columns[0]}")
        elif date_columns:
            try:
                # 日付カラムから年を抽出
                date_col = date_columns[0]
                self.df['年'] = pd.to_datetime(self.df[date_col], errors='coerce').dt.year
                logger.info(f"日付カラムから年を抽出しました: {date_col}")
            except:
                logger.warning("日付カラムからの年抽出に失敗しました")
                self.df['年'] = None
        else:
            logger.warning("年データが見つかりませんでした。全期間での分析を実行します。")
            self.df['年'] = None
    
    def _calculate_horse_weights(self, df_period):
        """馬ごとの重み付けポイント計算（従来通り全出走で計算）"""
        # 馬ごとの重み付けポイント平均を計算
        horse_weights = df_period.groupby('馬名').agg({
            '競馬場重み付けポイント': 'mean',
            '競馬場グレード重み': 'mean',
            '競馬場威信度重み': 'mean',
            '競馬場設備重み': 'mean',
            '勝利': ['count', 'sum', 'mean'],
            '複勝': ['sum', 'mean'],  # 複勝回数と複勝率を追加
            '着順': 'mean'
        }).round(4)
        
        # カラム名を整理
        horse_weights.columns = [
            '平均重み付けポイント', '平均グレード重み', '平均威信度重み', '平均設備重み',
            '出走数', '勝利数', '勝率', '複勝回数', '複勝率', '平均着順'
        ]
        
        # 複合重み付けポイント（馬ごと）
        horse_weights['複合重みポイント'] = (
            horse_weights['平均グレード重み'] * 0.4 +
            horse_weights['平均威信度重み'] * 0.3 +
            horse_weights['平均設備重み'] * 0.3
        )
        
        # データの検証ログ
        logger.info(f"馬ごと統計:")
        logger.info(f"  - 勝率0.0の馬: {(horse_weights['勝率'] == 0.0).sum()}頭")
        logger.info(f"  - 勝利経験のある馬: {(horse_weights['勝率'] > 0.0).sum()}頭")
        logger.info(f"  - 平均勝率: {horse_weights['勝率'].mean():.3f}")
        logger.info(f"  - 平均複勝率: {horse_weights['複勝率'].mean():.3f}")
        
        return horse_weights
    
    def _calculate_horse_race_point_stats(self, df_period):
        """馬ごとの複勝時重み付けポイント統計計算（複勝した場合のみポイント加算）"""
        horse_stats_list = []
        
        for horse_name in df_period['馬名'].unique():
            horse_data = df_period[df_period['馬名'] == horse_name]
            
            # 基本統計
            total_races = len(horse_data)
            win_count = horse_data['勝利'].sum()
            place_count = horse_data['複勝'].sum()
            win_rate = win_count / total_races if total_races > 0 else 0
            place_rate = place_count / total_races if total_races > 0 else 0
            avg_rank = horse_data['着順'].mean()
            
            # 複勝時のみの重み付けポイント計算
            place_races = horse_data[horse_data['複勝'] == 1]  # 複勝した場合のみ
            
            if len(place_races) > 0:
                # 複勝時の重み付けポイント平均
                avg_point = place_races['競馬場重み付けポイント'].mean()
                # 複勝時の累積ポイント
                cumulative_points = place_races['競馬場重み付けポイント'].sum()
            else:
                # 複勝経験がない場合は0
                avg_point = 0.0
                cumulative_points = 0.0
            
            horse_stats_list.append({
                '馬名': horse_name,
                '平均重み付けポイント': round(avg_point, 4),
                '累積重み付けポイント': round(cumulative_points, 4),
                '出走数': total_races,
                '勝利数': win_count,
                '勝率': round(win_rate, 4),
                '複勝回数': place_count,
                '複勝率': round(place_rate, 4),
                '平均着順': round(avg_rank, 4)
            })
        
        horse_stats = pd.DataFrame(horse_stats_list).set_index('馬名')
        
        # データの検証ログ
        logger.info(f"馬ごと複勝時重み付けポイント統計:")
        logger.info(f"  - 複勝経験なし（ポイント0）の馬: {(horse_stats['平均重み付けポイント'] == 0.0).sum()}頭")
        logger.info(f"  - 複勝経験あり（ポイント>0）の馬: {(horse_stats['平均重み付けポイント'] > 0.0).sum()}頭")
        logger.info(f"  - 平均重み付けポイント範囲: {horse_stats['平均重み付けポイント'].min():.2f} - {horse_stats['平均重み付けポイント'].max():.2f}")
        logger.info(f"  - 平均勝率: {horse_stats['勝率'].mean():.3f}")
        logger.info(f"  - 平均複勝率: {horse_stats['複勝率'].mean():.3f}")
        
        return horse_stats

    def _analyze_race_point_correlation(self, horse_stats):
        """平均重み付けポイントと勝率/複勝率の相関分析（複勝時のみ）"""
        if len(horse_stats) < 3:
            return {}
        
        results = {}
        
        try:
            # 複勝経験のある馬のみでの分析（ポイント>0）
            place_experienced_horses = horse_stats[horse_stats['平均重み付けポイント'] > 0.0]
            
            if len(place_experienced_horses) < 3:
                logger.warning("複勝経験のある馬が3頭未満のため、相関分析をスキップします")
                return {}
            
            logger.info(f"複勝経験馬での相関分析: {len(place_experienced_horses)}頭")
            
            # === 勝利数影響除去のための正規化指標計算 ===
            # 適切な正規化指標の計算
            place_experienced_horses = place_experienced_horses.copy()
            
            # 1. 複勝時の平均ポイント（これは既にある）
            # place_experienced_horses['平均重み付けポイント'] (既存)
            
            # 2. 出走回数あたりの期待ポイント（複勝率を考慮した期待値）
            place_experienced_horses['期待重み付けポイント'] = place_experienced_horses['平均重み付けポイント'] * place_experienced_horses['複勝率']
            
            # 3. 複勝経験レース数の影響を除去するため、複勝回数のランクベース正規化
            place_experienced_horses['複勝回数ランク'] = place_experienced_horses['複勝回数'].rank(pct=True)
            place_experienced_horses['累積ポイント_複勝回数調整'] = place_experienced_horses['累積重み付けポイント'] / (place_experienced_horses['複勝回数ランク'] + 0.1)  # 0除算防止
            
            # 4. 標準化された累積ポイント（Zスコア）
            cumulative_mean = place_experienced_horses['累積重み付けポイント'].mean()
            cumulative_std = place_experienced_horses['累積重み付けポイント'].std()
            if cumulative_std > 0:
                place_experienced_horses['標準化累積ポイント'] = (place_experienced_horses['累積重み付けポイント'] - cumulative_mean) / cumulative_std
            else:
                place_experienced_horses['標準化累積ポイント'] = 0
            
            # 5. 単純出走数正規化（複勝率と独立）- 削除予定だが比較のため残す
            place_experienced_horses['出走数正規化累積ポイント'] = place_experienced_horses['累積重み付けポイント'] / place_experienced_horses['出走数']
            place_experienced_horses['複勝回数正規化累積ポイント'] = place_experienced_horses['累積重み付けポイント'] / place_experienced_horses['複勝回数']
            
            # === 平均重み付けポイント分析 ===
            # 勝率分析
            win_rate_corr, win_rate_p = pearsonr(place_experienced_horses['平均重み付けポイント'], place_experienced_horses['勝率'])
            
            # 複勝率分析
            place_rate_corr, place_rate_p = pearsonr(place_experienced_horses['平均重み付けポイント'], place_experienced_horses['複勝率'])
            
            # 線形回帰分析
            X_avg = place_experienced_horses[['平均重み付けポイント']].values
            y_win = place_experienced_horses['勝率'].values
            y_place = place_experienced_horses['複勝率'].values
            
            reg_win_avg = LinearRegression()
            reg_win_avg.fit(X_avg, y_win)
            win_r2_avg = reg_win_avg.score(X_avg, y_win)
            
            reg_place_avg = LinearRegression()
            reg_place_avg.fit(X_avg, y_place)
            place_r2_avg = reg_place_avg.score(X_avg, y_place)
            
            # === 累積重み付けポイント（合計）分析 ===
            # 勝率分析
            cumulative_win_rate_corr, cumulative_win_rate_p = pearsonr(place_experienced_horses['累積重み付けポイント'], place_experienced_horses['勝率'])
            
            # 複勝率分析
            cumulative_place_rate_corr, cumulative_place_rate_p = pearsonr(place_experienced_horses['累積重み付けポイント'], place_experienced_horses['複勝率'])
            
            # 線形回帰分析
            X_cumulative = place_experienced_horses[['累積重み付けポイント']].values
            
            reg_win_cumulative = LinearRegression()
            reg_win_cumulative.fit(X_cumulative, y_win)
            win_r2_cumulative = reg_win_cumulative.score(X_cumulative, y_win)
            
            reg_place_cumulative = LinearRegression()
            reg_place_cumulative.fit(X_cumulative, y_place)
            place_r2_cumulative = reg_place_cumulative.score(X_cumulative, y_place)
            
            # === 正規化分析（勝利数影響除去） ===
            # 出走数正規化累積ポイント分析（注意：複勝率との相関は構造的に高くなる）
            normalized_win_rate_corr, normalized_win_rate_p = pearsonr(place_experienced_horses['出走数正規化累積ポイント'], place_experienced_horses['勝率'])
            normalized_place_rate_corr, normalized_place_rate_p = pearsonr(place_experienced_horses['出走数正規化累積ポイント'], place_experienced_horses['複勝率'])
            
            # 期待重み付けポイント分析（複勝率を考慮した期待値）
            expected_win_rate_corr, expected_win_rate_p = pearsonr(place_experienced_horses['期待重み付けポイント'], place_experienced_horses['勝率'])
            expected_place_rate_corr, expected_place_rate_p = pearsonr(place_experienced_horses['期待重み付けポイント'], place_experienced_horses['複勝率'])
            
            # 複勝回数調整累積ポイント分析
            adjusted_win_rate_corr, adjusted_win_rate_p = pearsonr(place_experienced_horses['累積ポイント_複勝回数調整'], place_experienced_horses['勝率'])
            adjusted_place_rate_corr, adjusted_place_rate_p = pearsonr(place_experienced_horses['累積ポイント_複勝回数調整'], place_experienced_horses['複勝率'])
            
            # 標準化累積ポイント分析
            std_win_rate_corr, std_win_rate_p = pearsonr(place_experienced_horses['標準化累積ポイント'], place_experienced_horses['勝率'])
            std_place_rate_corr, std_place_rate_p = pearsonr(place_experienced_horses['標準化累積ポイント'], place_experienced_horses['複勝率'])
            
            # 複勝回数正規化累積ポイント分析（これは平均と同じになるはず）
            place_normalized_win_rate_corr, place_normalized_win_rate_p = pearsonr(place_experienced_horses['複勝回数正規化累積ポイント'], place_experienced_horses['勝率'])
            place_normalized_place_rate_corr, place_normalized_place_rate_p = pearsonr(place_experienced_horses['複勝回数正規化累積ポイント'], place_experienced_horses['複勝率'])
            
            # 線形回帰分析（正規化）
            X_normalized = place_experienced_horses[['出走数正規化累積ポイント']].values
            X_expected = place_experienced_horses[['期待重み付けポイント']].values
            X_adjusted = place_experienced_horses[['累積ポイント_複勝回数調整']].values
            X_std = place_experienced_horses[['標準化累積ポイント']].values
            
            reg_win_normalized = LinearRegression()
            reg_win_normalized.fit(X_normalized, y_win)
            win_r2_normalized = reg_win_normalized.score(X_normalized, y_win)
            
            reg_place_normalized = LinearRegression()
            reg_place_normalized.fit(X_normalized, y_place)
            place_r2_normalized = reg_place_normalized.score(X_normalized, y_place)
            
            # 期待ポイントでの回帰
            reg_win_expected = LinearRegression()
            reg_win_expected.fit(X_expected, y_win)
            win_r2_expected = reg_win_expected.score(X_expected, y_win)
            
            reg_place_expected = LinearRegression()
            reg_place_expected.fit(X_expected, y_place)
            place_r2_expected = reg_place_expected.score(X_expected, y_place)
            
            # 複勝回数調整での回帰
            reg_win_adjusted = LinearRegression()
            reg_win_adjusted.fit(X_adjusted, y_win)
            win_r2_adjusted = reg_win_adjusted.score(X_adjusted, y_win)
            
            reg_place_adjusted = LinearRegression()
            reg_place_adjusted.fit(X_adjusted, y_place)
            place_r2_adjusted = reg_place_adjusted.score(X_adjusted, y_place)
            
            # === 部分相関分析（複勝回数の影響を統制） ===
            from scipy.stats import pearsonr as sp_pearsonr
            import numpy as np
            
            def partial_correlation(x, y, control):
                """部分相関係数を計算"""
                # xとcontrolの相関
                rx_control, _ = sp_pearsonr(x, control)
                # yとcontrolの相関
                ry_control, _ = sp_pearsonr(y, control)
                # xとyの相関
                rxy, _ = sp_pearsonr(x, y)
                
                # 部分相関係数の計算
                numerator = rxy - (rx_control * ry_control)
                denominator = np.sqrt((1 - rx_control**2) * (1 - ry_control**2))
                
                if denominator == 0:
                    return 0, 1  # 相関係数、p値
                
                partial_r = numerator / denominator
                
                # 簡易的なp値計算（正確ではないが近似値）
                n = len(x)
                t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r**2))
                # 簡易的にp値を推定（正確な統計的検定ではない）
                p_value = 0.05 if abs(t_stat) > 1.96 else 0.1  # 近似値
                
                return partial_r, p_value
            
            # 累積ポイントと勝率の関係（複勝回数を統制）
            partial_cumulative_win_corr, partial_cumulative_win_p = partial_correlation(
                place_experienced_horses['累積重み付けポイント'],
                place_experienced_horses['勝率'],
                place_experienced_horses['複勝回数']
            )
            
            # 累積ポイントと複勝率の関係（複勝回数を統制）
            partial_cumulative_place_corr, partial_cumulative_place_p = partial_correlation(
                place_experienced_horses['累積重み付けポイント'],
                place_experienced_horses['複勝率'],
                place_experienced_horses['複勝回数']
            )
            
            # === ロジスティック回帰（簡略版） ===
            # 平均ポイントのみでロジスティック回帰を実行
            logistic_data_avg = []
            for _, row in place_experienced_horses.iterrows():
                point = row['平均重み付けポイント']
                for _ in range(int(row['勝利数'])):
                    logistic_data_avg.append([point, 1])
                for _ in range(int(row['出走数'] - row['勝利数'])):
                    logistic_data_avg.append([point, 0])
            
            if len(logistic_data_avg) > 0:
                logistic_df_avg = pd.DataFrame(logistic_data_avg, columns=['平均重み付けポイント', '勝利フラグ'])
                X_logistic_avg = logistic_df_avg[['平均重み付けポイント']].values
                y_logistic_avg = logistic_df_avg['勝利フラグ'].values
                
                logistic_reg_avg = LogisticRegression()
                logistic_reg_avg.fit(X_logistic_avg, y_logistic_avg)
                
                results['logistic_regression_avg'] = {
                    'model': logistic_reg_avg,
                    'X': X_logistic_avg,
                    'y': y_logistic_avg
                }
            
            # 結果の格納と返却
            results = {
                'place_experienced_horses': place_experienced_horses,  # 正規化指標を含むデータフレーム
                'correlation_analysis': {
                    'avg_point': {
                        'win_rate': {
                            'correlation': win_rate_corr,
                            'p_value': win_rate_p,
                            'r2': win_r2_avg,
                            'regression': reg_win_avg
                        },
                        'place_rate': {
                            'correlation': place_rate_corr,
                            'p_value': place_rate_p,
                            'r2': place_r2_avg,
                            'regression': reg_place_avg
                        }
                    },
                    'cumulative': {
                        'win_rate': {
                            'correlation': cumulative_win_rate_corr,
                            'p_value': cumulative_win_rate_p,
                            'r2': win_r2_cumulative,
                            'regression': reg_win_cumulative
                        },
                        'place_rate': {
                            'correlation': cumulative_place_rate_corr,
                            'p_value': cumulative_place_rate_p,
                            'r2': place_r2_cumulative,
                            'regression': reg_place_cumulative
                        }
                    },
                    'normalized': {
                        'win_rate': {
                            'correlation': normalized_win_rate_corr,
                            'p_value': normalized_win_rate_p,
                            'r2': win_r2_normalized,
                            'regression': reg_win_normalized
                        },
                        'place_rate': {
                            'correlation': normalized_place_rate_corr,
                            'p_value': normalized_place_rate_p,
                            'r2': place_r2_normalized,
                            'regression': reg_place_normalized
                        }
                    },
                    'expected': {
                        'win_rate': {
                            'correlation': expected_win_rate_corr,
                            'p_value': expected_win_rate_p,
                            'r2': win_r2_expected,
                            'regression': reg_win_expected
                        },
                        'place_rate': {
                            'correlation': expected_place_rate_corr,
                            'p_value': expected_place_rate_p,
                            'r2': place_r2_expected,
                            'regression': reg_place_expected
                        }
                    },
                    'adjusted': {
                        'win_rate': {
                            'correlation': adjusted_win_rate_corr,
                            'p_value': adjusted_win_rate_p,
                            'r2': win_r2_adjusted,
                            'regression': reg_win_adjusted
                        },
                        'place_rate': {
                            'correlation': adjusted_place_rate_corr,
                            'p_value': adjusted_place_rate_p,
                            'r2': place_r2_adjusted,
                            'regression': reg_place_adjusted
                        }
                    },
                    'partial_correlation': {
                        'cumulative_vs_win_rate': partial_cumulative_win_corr,
                        'cumulative_vs_place_rate': partial_cumulative_place_corr
                    }
                }
            }
            
            logger.info(f"複勝時重み付けポイント相関分析:")
            logger.info(f"  - 平均ポイント vs 勝率: r={win_rate_corr:.3f}, p={win_rate_p:.3f}, R²={win_r2_avg:.3f}")
            logger.info(f"  - 平均ポイント vs 複勝率: r={place_rate_corr:.3f}, p={place_rate_p:.3f}, R²={place_r2_avg:.3f}")
            logger.info(f"  - 累積ポイント vs 勝率: r={cumulative_win_rate_corr:.3f}, p={cumulative_win_rate_p:.3f}, R²={win_r2_cumulative:.3f}")
            logger.info(f"  - 累積ポイント vs 複勝率: r={cumulative_place_rate_corr:.3f}, p={cumulative_place_rate_p:.3f}, R²={place_r2_cumulative:.3f}")
            logger.warning(f"⚠️  出走数正規化累積ポイント vs 複勝率: r={normalized_place_rate_corr:.3f} （構造的に高相関）")
            logger.info(f"  - 期待重み付けポイント vs 勝率: r={expected_win_rate_corr:.3f}, p={expected_win_rate_p:.3f}, R²={win_r2_expected:.3f}")
            logger.info(f"  - 複勝回数調整累積ポイント vs 勝率: r={adjusted_win_rate_corr:.3f}, p={adjusted_win_rate_p:.3f}, R²={win_r2_adjusted:.3f}")
            logger.info(f"  - 標準化累積ポイント vs 勝率: r={std_win_rate_corr:.3f}, p={std_win_rate_p:.3f}")
            logger.info(f"  - 部分相関（複勝回数統制）累積ポイント vs 勝率: r={partial_cumulative_win_corr:.3f}")
            logger.info(f"  - 部分相関（複勝回数統制）累積ポイント vs 複勝率: r={partial_cumulative_place_corr:.3f}")
        
        except Exception as e:
            logger.warning(f"重み付けポイント相関分析でエラー: {e}")
        
        return results

    def analyze_track_win_rates(self):
        """競馬場別勝率分析（3年間隔・複勝時のみポイント加算）"""
        logger.info("馬ごと複勝時重み付けポイント分析開始...")
        
        results = {}
        
        # 年データがある場合は3年間隔で分析
        if self.df['年'].notna().any():
            min_year = int(self.df['年'].min())
            max_year = int(self.df['年'].max())
            logger.info(f"年データ範囲: {min_year}年 - {max_year}年")
            
            # 3年間隔での期間設定
            periods = []
            for start_year in range(min_year, max_year + 1, 3):
                end_year = min(start_year + 2, max_year)
                periods.append((start_year, end_year))
            
            logger.info(f"分析期間: {periods}")
            
            for start_year, end_year in periods:
                period_name = f"{start_year}-{end_year}"
                logger.info(f"期間 {period_name} の分析開始...")
                
                # 期間データの抽出
                df_period = self.df[
                    (self.df['年'] >= start_year) & (self.df['年'] <= end_year)
                ].copy()
                
                if len(df_period) < self.config['min_races']:
                    logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(df_period)}行)")
                    continue
                
                # 馬ごとの重み付けポイント計算（従来通り）
                horse_weights = self._calculate_horse_weights(df_period)
                
                # 馬ごとの複勝時重み付けポイント統計計算（新機能）
                horse_race_point_stats = self._calculate_horse_race_point_stats(df_period)
                
                # 最小出走数でフィルタ
                horse_weights_filtered = horse_weights[
                    horse_weights['出走数'] >= self.config['min_races']
                ]
                
                horse_race_point_filtered = horse_race_point_stats[
                    horse_race_point_stats['出走数'] >= self.config['min_races']
                ]
                
                if len(horse_weights_filtered) < 3:
                    logger.warning(f"期間 {period_name}: 分析対象馬が不足のためスキップ")
                    continue
                
                # 競馬場別統計（従来通り）
                track_stats = self._calculate_track_stats(df_period)
                
                # 重み付けポイント相関分析（従来通り）
                weight_analysis = self._analyze_weight_correlation_horses(horse_weights_filtered)
                
                # 複勝時重み付けポイント相関分析（新機能）
                race_point_correlation = self._analyze_race_point_correlation(horse_race_point_filtered)
                
                results[period_name] = {
                    'horse_weights': horse_weights_filtered,
                    'horse_race_point_stats': horse_race_point_filtered,
                    'track_stats': track_stats,
                    'weight_analysis': weight_analysis,
                    'race_point_correlation': race_point_correlation,
                    'period': (start_year, end_year),
                    'total_races': len(df_period),
                    'total_horses': len(horse_weights_filtered)
                }
        
        else:
            # 年データがない場合は全期間で分析
            logger.info("全期間での分析を実行...")
            
            # 馬ごとの重み付けポイント計算
            horse_weights = self._calculate_horse_weights(self.df)
            
            # 馬ごとの複勝時重み付けポイント統計計算（新機能）
            horse_race_point_stats = self._calculate_horse_race_point_stats(self.df)
            
            # 最小出走数でフィルタ
            horse_weights_filtered = horse_weights[
                horse_weights['出走数'] >= self.config['min_races']
            ]
            
            horse_race_point_filtered = horse_race_point_stats[
                horse_race_point_stats['出走数'] >= self.config['min_races']
            ]
            
            # 競馬場別統計
            track_stats = self._calculate_track_stats(self.df)
            
            # 重み付けポイント相関分析
            weight_analysis = self._analyze_weight_correlation_horses(horse_weights_filtered)
            
            # 複勝時重み付けポイント相関分析（新機能）
            race_point_correlation = self._analyze_race_point_correlation(horse_race_point_filtered)
            
            results['全期間'] = {
                'horse_weights': horse_weights_filtered,
                'horse_race_point_stats': horse_race_point_filtered,
                'track_stats': track_stats,
                'weight_analysis': weight_analysis,
                'race_point_correlation': race_point_correlation,
                'period': None,
                'total_races': len(self.df),
                'total_horses': len(horse_weights_filtered)
            }
        
        return results
    
    def _calculate_track_stats(self, df_period):
        """競馬場別統計計算"""
        track_stats = df_period.groupby('場名').agg({
            '勝利': ['count', 'sum', 'mean'],
            '複勝': 'mean',
            '着順': 'mean',
            '馬名': 'nunique'
        }).round(4)
        
        track_stats.columns = ['出走数', '勝利数', '勝率', '複勝率', '平均着順', '出走馬数']
        
        # 競馬場情報を追加
        track_stats['競馬場レベル'] = track_stats.index.map(
            lambda x: self.track_hierarchy.get(x, {}).get('level', 0)
        )
        track_stats['競馬場タイプ'] = track_stats.index.map(
            lambda x: self.track_hierarchy.get(x, {}).get('type', '不明')
        )
        track_stats['重み付けポイント'] = track_stats.index.map(
            lambda x: self.track_hierarchy.get(x, {}).get('weight_points', 0)
        )
        
        return track_stats
    
    def _analyze_weight_correlation_horses(self, horse_weights):
        """馬ごと重み付けポイントと勝率の相関分析（従来通り）"""
        if len(horse_weights) < 3:
            return {}
        
        results = {}
        
        try:
            # 勝利経験のある馬のみでの分析も実施
            winning_horses = horse_weights[horse_weights['勝率'] > 0.0]
            logger.info(f"勝利経験馬での分析: {len(winning_horses)}頭")
            
            # 全馬での分析
            results['all_horses'] = self._perform_correlation_analysis(
                horse_weights, '全馬', use_placerate=False
            )
            
            # 勝利経験馬のみでの分析
            if len(winning_horses) >= 3:
                results['winning_horses'] = self._perform_correlation_analysis(
                    winning_horses, '勝利経験馬', use_placerate=False
                )
            
            # 複勝率での分析（全馬）
            results['placerate_analysis'] = self._perform_correlation_analysis(
                horse_weights, '複勝率分析', use_placerate=True
            )
            
            results['horse_weights'] = horse_weights
            results['winning_horses_data'] = winning_horses if len(winning_horses) >= 3 else None
        
        except Exception as e:
            logger.warning(f"重み付け相関分析でエラー: {e}")
        
        return results
    
    def _perform_correlation_analysis(self, data, analysis_name, use_placerate=False):
        """相関分析の実行"""
        target_rate = '複勝率' if use_placerate else '勝率'
        
        analysis_results = {}
        
        # 1. ピアソン相関分析
        weight_corr, weight_p = pearsonr(data['平均重み付けポイント'], data[target_rate])
        composite_corr, composite_p = pearsonr(data['複合重みポイント'], data[target_rate])
        
        analysis_results['pearson_correlation'] = {
            'weight_win_corr': weight_corr,
            'weight_win_p': weight_p,
            'composite_win_corr': composite_corr,
            'composite_win_p': composite_p,
            'target_rate': target_rate
        }
        
        # 2. スピアマン相関分析
        weight_spear, weight_spear_p = spearmanr(data['平均重み付けポイント'], data[target_rate])
        composite_spear, composite_spear_p = spearmanr(data['複合重みポイント'], data[target_rate])
        
        analysis_results['spearman_correlation'] = {
            'weight_win_corr': weight_spear,
            'weight_win_p': weight_spear_p,
            'composite_win_corr': composite_spear,
            'composite_win_p': composite_spear_p,
            'target_rate': target_rate
        }
        
        # 3. 線形回帰分析
        X_weight = data[['平均重み付けポイント']].values
        X_composite = data[['複合重みポイント']].values
        y = data[target_rate].values
        
        reg_weight = LinearRegression()
        reg_weight.fit(X_weight, y)
        weight_r2 = reg_weight.score(X_weight, y)
        
        reg_composite = LinearRegression()
        reg_composite.fit(X_composite, y)
        composite_r2 = reg_composite.score(X_composite, y)
        
        analysis_results['linear_regression'] = {
            'weight': {
                'r2': weight_r2,
                'coefficient': reg_weight.coef_[0],
                'intercept': reg_weight.intercept_
            },
            'composite': {
                'r2': composite_r2,
                'coefficient': reg_composite.coef_[0],
                'intercept': reg_composite.intercept_
            },
            'target_rate': target_rate
        }
        
        logger.info(f"{analysis_name}（{target_rate}）相関分析:")
        logger.info(f"  - ピアソン相関: r={weight_corr:.3f}, p={weight_p:.3f}")
        logger.info(f"  - 決定係数: R²={weight_r2:.3f}")
        
        return analysis_results
    
    def visualize_results(self, results):
        """結果の可視化（3年間隔・複勝時のみポイント加算対応）"""
        logger.info("結果の可視化開始...")
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for period_name, period_results in results.items():
            logger.info(f"期間 {period_name} の可視化...")
            
            period_output_dir = output_dir / period_name
            period_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 競馬場別分析
            if 'track_stats' in period_results:
                self._plot_track_analysis(period_results, period_output_dir, period_name)
            
            # 馬ごと重み付けポイント分析（従来通り）
            if 'weight_analysis' in period_results and 'horse_weights' in period_results['weight_analysis']:
                self._plot_horse_weight_analysis(period_results['weight_analysis'], period_output_dir, period_name)
            
            # 複勝時重み付けポイント相関分析（新機能）
            if ('horse_race_point_stats' in period_results and 
                'race_point_correlation' in period_results and
                'correlation_analysis' in period_results['race_point_correlation']):
                
                self._plot_race_point_correlation_analysis(
                    period_results['horse_race_point_stats'],
                    period_results['race_point_correlation'],
                    period_output_dir,
                    period_name
                )
        
        logger.info("可視化完了")
    
    def _plot_track_analysis(self, period_results, output_dir, period_name):
        """競馬場分析可視化"""
        track_stats = period_results['track_stats']
        
        # 出走数でフィルタ
        track_stats_filtered = track_stats[track_stats['出走数'] >= self.config['min_races']]
        
        if len(track_stats_filtered) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'競馬場別分析 ({period_name})', fontsize=16)
        
        # 勝率
        ax1 = axes[0, 0]
        track_sorted = track_stats_filtered.sort_values('勝率', ascending=True)
        colors = ['red' if x == '地方' else 'blue' for x in track_sorted['競馬場タイプ']]
        ax1.barh(range(len(track_sorted)), track_sorted['勝率'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(track_sorted)))
        ax1.set_yticklabels(track_sorted.index, fontsize=8)
        ax1.set_xlabel('勝率')
        ax1.set_title('競馬場別勝率')
        ax1.grid(True, alpha=0.3)
        
        # 出走数
        ax2 = axes[0, 1]
        ax2.bar(range(len(track_sorted)), track_sorted['出走数'], alpha=0.7, color='orange')
        ax2.set_xticks(range(len(track_sorted)))
        ax2.set_xticklabels(track_sorted.index, rotation=45, fontsize=8)
        ax2.set_ylabel('出走数')
        ax2.set_title('競馬場別出走数')
        ax2.grid(True, alpha=0.3)
        
        # 重み付けポイント vs 勝率
        ax3 = axes[1, 0]
        central_tracks = track_stats_filtered[track_stats_filtered['競馬場タイプ'] == '中央']
        local_tracks = track_stats_filtered[track_stats_filtered['競馬場タイプ'] == '地方']
        
        if len(central_tracks) > 0:
            ax3.scatter(central_tracks['重み付けポイント'], central_tracks['勝率'], 
                       color='blue', alpha=0.7, label='中央', s=60)
        if len(local_tracks) > 0:
            ax3.scatter(local_tracks['重み付けポイント'], local_tracks['勝率'], 
                       color='red', alpha=0.7, label='地方', s=60)
        
        ax3.set_xlabel('重み付けポイント')
        ax3.set_ylabel('勝率')
        ax3.set_title('競馬場重み付けポイント vs 勝率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 平均着順
        ax4 = axes[1, 1]
        ax4.barh(range(len(track_sorted)), track_sorted['平均着順'], color=colors, alpha=0.7)
        ax4.set_yticks(range(len(track_sorted)))
        ax4.set_yticklabels(track_sorted.index, fontsize=8)
        ax4.set_xlabel('平均着順')
        ax4.set_title('競馬場別平均着順')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'競馬場別分析_{period_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_horse_weight_analysis(self, weight_analysis, output_dir, period_name):
        """馬ごと重み付けポイント分析可視化（従来通り）"""
        
        # weight_analysisの構造に応じて適切なデータを取得
        if 'horse_weights' in weight_analysis:
            horse_weights = weight_analysis['horse_weights']
        else:
            logger.warning(f"期間 {period_name}: horse_weightsデータが見つかりません")
            return
        
        # main_analysisが存在するかチェック、なければall_horsesを使用
        if 'main_analysis' in weight_analysis:
            main_analysis = weight_analysis['main_analysis']
        elif 'all_horses' in weight_analysis:
            main_analysis = weight_analysis['all_horses']
        else:
            logger.warning(f"期間 {period_name}: 相関分析データが見つかりません")
            main_analysis = None
        
        # 勝利経験のある馬のみを抽出（プロット用）
        plot_data = horse_weights[horse_weights['勝利数'] > 0].copy()
        
        if len(plot_data) == 0:
            logger.warning(f"期間 {period_name}: 勝利経験のある馬がいないため、散布図をスキップします")
            return
        
        # 6つのサブプロットを作成
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 平均重み付けポイント vs 勝率
        ax1 = axes[0, 0]
        
        # レース回数（出走数）に基づくサイズ設定（より明確に）
        min_size = 30
        max_size = 200
        race_counts = plot_data['出走数']
        # レース回数を正規化してサイズに変換
        normalized_sizes = min_size + (race_counts - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
        
        scatter1 = ax1.scatter(plot_data['平均重み付けポイント'], plot_data['勝率'], 
                             c=plot_data['出走数'], cmap='viridis', 
                             alpha=0.7, s=normalized_sizes, edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter1, ax=ax1, label='出走数')
        ax1.set_xlabel('平均重み付けポイント')
        ax1.set_ylabel('勝率')
        ax1.set_title('平均重み付けポイント vs 勝率')
        ax1.grid(True, alpha=0.3)
        
        # 2. 累積重み付けポイント vs 勝率（horse_weightsに累積ポイントがあるかチェック）
        ax2 = axes[0, 1]
        
        if '累積重み付けポイント' in plot_data.columns:
            # 累積ポイント用のサイズ設定
            cumulative_counts = plot_data['出走数']
            normalized_sizes_cum = min_size + (cumulative_counts - cumulative_counts.min()) / (cumulative_counts.max() - cumulative_counts.min()) * (max_size - min_size)
            
            scatter2 = ax2.scatter(plot_data['累積重み付けポイント'], plot_data['勝率'], 
                                 c=plot_data['出走数'], cmap='plasma', 
                                 alpha=0.7, s=normalized_sizes_cum, edgecolors='black', linewidth=0.5)
            
            plt.colorbar(scatter2, ax=ax2, label='出走数')
            ax2.set_xlabel('累積重み付けポイント')
            ax2.set_ylabel('勝率')
            ax2.set_title('累積重み付けポイント vs 勝率')
        else:
            # 累積ポイントがない場合は代替として平均ポイント×出走数を表示
            cumulative_proxy = plot_data['平均重み付けポイント'] * plot_data['出走数']
            cumulative_counts = plot_data['出走数']
            normalized_sizes_cum = min_size + (cumulative_counts - cumulative_counts.min()) / (cumulative_counts.max() - cumulative_counts.min()) * (max_size - min_size)
            
            scatter2 = ax2.scatter(cumulative_proxy, plot_data['勝率'], 
                                 c=plot_data['出走数'], cmap='plasma', 
                                 alpha=0.7, s=normalized_sizes_cum, edgecolors='black', linewidth=0.5)
            
            plt.colorbar(scatter2, ax=ax2, label='出走数')
            ax2.set_xlabel('推定累積重み付けポイント（平均×出走数）')
            ax2.set_ylabel('勝率')
            ax2.set_title('推定累積重み付けポイント vs 勝率')
        ax2.grid(True, alpha=0.3)
        
        # 3. 複合重み付けポイント vs 勝率
        ax3 = axes[0, 2]
        
        # 複合ポイント用のサイズ設定
        composite_counts = plot_data['出走数']
        normalized_sizes_comp = min_size + (composite_counts - composite_counts.min()) / (composite_counts.max() - composite_counts.min()) * (max_size - min_size)
        
        scatter3 = ax3.scatter(plot_data['複合重みポイント'], plot_data['勝率'], 
                             c=plot_data['出走数'], cmap='coolwarm', 
                             alpha=0.7, s=normalized_sizes_comp, edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter3, ax=ax3, label='出走数')
        ax3.set_xlabel('複合重み付けポイント')
        ax3.set_ylabel('勝率')
        ax3.set_title('複合重み付けポイント vs 勝率')
        ax3.grid(True, alpha=0.3)
        
        # 4. 出走数 vs 勝率（重み付けポイント別）
        ax4 = axes[1, 0]
        
        # 出走数vs勝率用のサイズ設定（平均重み付けポイントに基づく）
        point_based_sizes = min_size + (plot_data['平均重み付けポイント'] - plot_data['平均重み付けポイント'].min()) / (plot_data['平均重み付けポイント'].max() - plot_data['平均重み付けポイント'].min()) * (max_size - min_size)
        
        scatter4 = ax4.scatter(plot_data['出走数'], plot_data['勝率'], 
                             c=plot_data['平均重み付けポイント'], cmap='viridis', 
                             alpha=0.7, s=point_based_sizes, edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter4, ax=ax4, label='平均重み付けポイント')
        ax4.set_xlabel('出走数')
        ax4.set_ylabel('勝率')
        ax4.set_title('出走数 vs 勝率（重み付けポイント別）')
        ax4.grid(True, alpha=0.3)
        
        # 5. 相関係数比較
        ax5 = axes[1, 1]
        
        if main_analysis and 'pearson_correlation' in main_analysis and 'spearman_correlation' in main_analysis:
            pearson_data = [
                main_analysis['pearson_correlation']['weight_win_corr'],
                main_analysis['pearson_correlation']['composite_win_corr']
            ]
            spearman_data = [
                main_analysis['spearman_correlation']['weight_win_corr'],
                main_analysis['spearman_correlation']['composite_win_corr']
            ]
            
            x_pos = np.arange(2)
            width = 0.35
            
            ax5.bar(x_pos - width/2, pearson_data, width, label='ピアソン相関', alpha=0.7)
            ax5.bar(x_pos + width/2, spearman_data, width, label='スピアマン相関', alpha=0.7)
            
            ax5.set_xlabel('重み付けタイプ')
            ax5.set_ylabel('相関係数')
            ax5.set_title('相関係数比較')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(['平均重み付け', '複合重み付け'])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            # 相関分析データがない場合のメッセージ
            ax5.text(0.5, 0.5, '相関分析データなし', 
                    transform=ax5.transAxes, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax5.set_title('相関係数比較')
            ax5.axis('off')
        
        # 6. 勝率分布
        ax6 = axes[1, 2]
        ax6.hist(horse_weights['勝率'], bins=20, alpha=0.7, color='skyblue', edgecolor='black', label='全馬')
        if len(plot_data) < len(horse_weights):  # 勝利経験馬のみの場合
            ax6.hist(plot_data['勝率'], bins=20, alpha=0.7, color='orange', edgecolor='black', label='勝利経験馬')
        
        ax6.set_xlabel('勝率')
        ax6.set_ylabel('馬数')
        ax6.set_title('勝率分布')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 全体のタイトルに出走数情報を追加
        fig.suptitle(f'馬ごと重み付け分析 ({period_name})\n出走数範囲: {int(race_counts.min())}～{int(race_counts.max())}回 (平均: {race_counts.mean():.1f}回)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'馬ごと重み付け分析_{period_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"馬ごと重み付け分析図を保存: {output_dir / f'馬ごと重み付け分析_{period_name}.png'}")
    
    def _plot_race_point_correlation_analysis(self, horse_stats, correlation_results, output_dir, period_name):
        """複勝時重み付けポイント相関分析の可視化（指定ファイル名）"""
        if 'correlation_analysis' not in correlation_results:
            return
        
        correlation_data = correlation_results['correlation_analysis']
        place_experienced_horses = correlation_results.get('place_experienced_horses', horse_stats)
        
        # === 平均重み付けポイント分析 ===
        # 1. 重み付けポイント（平均）と勝率の関係（複勝経験馬のみ）
        self._plot_correlation_scatter(
            place_experienced_horses, 
            '重み付けポイント（平均）と勝率の関係',
            '平均重み付けポイント',
            '勝率',
            correlation_data['avg_point']['win_rate'],
            output_dir / f'重み付けポイント（平均）と勝率の関係_correlation.png'
        )
        
        # 2. 重み付けポイント（平均）と複勝率の関係（複勝経験馬のみ）
        self._plot_correlation_scatter(
            place_experienced_horses,
            '重み付けポイント（平均）と複勝率の関係', 
            '平均重み付けポイント',
            '複勝率',
            correlation_data['avg_point']['place_rate'],
            output_dir / f'重み付けポイント（平均）と複勝率の関係_correlation.png'
        )
        
        # === 累積重み付けポイント（合計）分析 ===
        # 3. 重み付けポイント（合計）と勝率の関係（複勝経験馬のみ）
        self._plot_correlation_scatter(
            place_experienced_horses,
            '重み付けポイント（合計）と勝率の関係',
            '累積重み付けポイント',
            '勝率',
            correlation_data['cumulative']['win_rate'],
            output_dir / f'重み付けポイント（合計）と勝率の関係_correlation.png'
        )
        
        # 4. 重み付けポイント（合計）と複勝率の関係（複勝経験馬のみ）
        self._plot_correlation_scatter(
            place_experienced_horses,
            '重み付けポイント（合計）と複勝率の関係',
            '累積重み付けポイント',
            '複勝率',
            correlation_data['cumulative']['place_rate'],
            output_dir / f'重み付けポイント（合計）と複勝率の関係_correlation.png'
        )
        
        # === 正規化分析（勝利数影響除去） ===
        # 5. 出走数正規化重み付けポイントと勝率の関係
        self._plot_correlation_scatter(
            place_experienced_horses,
            '期待重み付けポイントと勝率の関係（複勝率考慮）',
            '期待重み付けポイント',
            '勝率',
            correlation_data['expected']['win_rate'],
            output_dir / f'期待重み付けポイントと勝率の関係_correlation.png'
        )
        
        # 6. 出走数正規化重み付けポイントと複勝率の関係（警告付き）
        self._plot_correlation_scatter_with_warning(
            place_experienced_horses,
            '出走数正規化重み付けポイントと複勝率の関係（⚠️構造的高相関）',
            '出走数正規化累積ポイント',
            '複勝率',
            correlation_data['normalized']['place_rate'],
            output_dir / f'出走数正規化重み付けポイントと複勝率の関係_correlation.png'
        )
        
        # 7. 複勝回数調整累積ポイントと勝率の関係
        self._plot_correlation_scatter(
            place_experienced_horses,
            '複勝回数調整累積ポイントと勝率の関係（バイアス除去）',
            '累積ポイント_複勝回数調整',
            '勝率',
            correlation_data['adjusted']['win_rate'],
            output_dir / f'複勝回数調整累積ポイントと勝率の関係_correlation.png'
        )
        
        # === ロジスティック回帰曲線 ===
        # 8. 平均ポイントでのロジスティック回帰曲線
        if 'logistic_regression_avg' in correlation_results:
            self._plot_logistic_regression_curve(
                correlation_results['logistic_regression_avg'],
                output_dir / f'重み付けポイント（平均）と勝率の関係（ロジスティック回帰）_logistic_regression_curve.png'
            )
        
        # 9. 累積ポイントでのロジスティック回帰曲線
        if 'logistic_regression_cumulative' in correlation_results:
            self._plot_logistic_regression_curve(
                correlation_results['logistic_regression_cumulative'],
                output_dir / f'重み付けポイント（合計）と勝率の関係（ロジスティック回帰）_logistic_regression_curve.png'
            )
        
        # 10. 正規化ポイントでのロジスティック回帰曲線
        if 'logistic_regression_normalized' in correlation_results:
            self._plot_logistic_regression_curve(
                correlation_results['logistic_regression_normalized'],
                output_dir / f'出走数正規化重み付けポイントと勝率の関係（ロジスティック回帰）_logistic_regression_curve.png'
            )

    def _plot_correlation_scatter(self, horse_stats, title, x_col, y_col, correlation_data, output_path):
        """相関散布図の描画（複勝経験馬のみ）"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # データの準備
        x = horse_stats[x_col]
        y = horse_stats[y_col]
        
        # レース回数（出走数）に基づくサイズ設定（より明確に）
        min_size = 30
        max_size = 300
        race_counts = horse_stats['出走数']
        # レース回数を正規化してサイズに変換
        normalized_sizes = min_size + (race_counts - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
        
        # 散布図
        scatter = ax.scatter(x, y, c=horse_stats['出走数'], s=normalized_sizes, alpha=0.6, 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('出走数', fontsize=12)
        
        # 回帰直線
        regression = correlation_data['regression']
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred = regression.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r--', linewidth=2, 
               label=f'回帰直線 (R² = {correlation_data["r2"]:.3f})')
        
        # 統計情報の表示
        corr = correlation_data['correlation']
        p_val = correlation_data['p_value']
        
        stats_text = f'相関係数: {corr:.3f}\n'
        stats_text += f'p値: {p_val:.3f}\n' 
        stats_text += f'有意性: {"有意" if p_val < 0.05 else "非有意"}\n'
        stats_text += f'対象: 複勝経験馬のみ ({len(horse_stats)}頭)'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # レース回数のサイズ凡例を追加
        sizes_for_legend = [race_counts.min(), race_counts.quantile(0.5), race_counts.max()]
        labels_for_legend = [f'{int(size)}回' for size in sizes_for_legend]
        
        # サイズ凡例用のマーカーサイズを計算
        legend_sizes = []
        for size in sizes_for_legend:
            normalized_size = min_size + (size - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
            legend_sizes.append(normalized_size)
        
        # サイズ凡例の作成
        legend_elements = []
        for size, label, marker_size in zip(sizes_for_legend, labels_for_legend, legend_sizes):
            legend_elements.append(plt.scatter([], [], s=marker_size, c='gray', alpha=0.6, 
                                             edgecolors='black', linewidth=0.5, label=label))
        
        # 既存の凡例と組み合わせ
        legend1 = plt.legend(handles=legend_elements, title="出走数（点のサイズ）", 
                           loc='upper left', bbox_to_anchor=(0, 0.85), frameon=True, fancybox=True, shadow=True)
        plt.gca().add_artist(legend1)
        
        # 回帰直線の凡例
        legend2 = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # グラフ左下に出走数の範囲情報を追加
        info_text = f"出走数範囲: {int(race_counts.min())}～{int(race_counts.max())}回\n平均: {race_counts.mean():.1f}回"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom', fontsize=10)
        
        # ラベルとタイトル
        ax.set_xlabel(f'{x_col} ※複勝した場合のみポイント加算', fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{title}（複勝経験馬のみ）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相関図を保存: {output_path}")

    def _plot_logistic_regression_curve(self, logistic_data, output_path):
        """ロジスティック回帰曲線の描画（複勝経験馬のみ）"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        model = logistic_data['model']
        X = logistic_data['X']
        y = logistic_data['y']
        
        # 実データの散布図（ジッター付き）
        x_vals = X.flatten()
        y_jittered = y + np.random.normal(0, 0.02, len(y))  # ジッター追加
        
        # 勝利・敗北で色分け
        win_mask = y == 1
        lose_mask = y == 0
        
        ax.scatter(x_vals[win_mask], y_jittered[win_mask], 
                  color='red', alpha=0.4, s=20, label='実データ（赤:勝, 青:敗）')
        ax.scatter(x_vals[lose_mask], y_jittered[lose_mask], 
                  color='blue', alpha=0.4, s=20)
        
        # ロジスティック回帰曲線
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
        
        ax.plot(x_range, y_prob, 'g-', linewidth=3, label='ロジスティック回帰曲線')
        
        # 50%ライン
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%ライン')
        
        # 予測曲線の説明
        ax.text(0.05, 0.95, '複勝経験馬のみの予測曲線', transform=ax.transAxes,
               verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # ラベルとタイトル
        ax.set_xlabel('重み付けポイント（複勝時のみ）', fontsize=12)
        ax.set_ylabel('勝利確率', fontsize=12)
        ax.set_title('ロジスティック回帰分析\n複勝経験馬の重み付けポイントと勝利確率', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ロジスティック回帰曲線を保存: {output_path}")

    def generate_report(self, results):
        """分析レポート生成（3年間隔・複勝時のみポイント加算対応）"""
        logger.info("分析レポート生成開始...")
        
        output_dir = Path(self.config['output_dir'])
        
        # 全体レポート
        main_report_path = output_dir / '馬ごと複勝時重み付けポイント分析レポート.md'
        
        with open(main_report_path, 'w', encoding='utf-8') as f:
            f.write("# 馬ごと複勝時重み付けポイント分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 分析概要\n\n")
            f.write("**複勝した場合のみ**重み付けポイントを加算し、馬の勝率との関係を分析しました。\n")
            f.write("これにより、実際に好成績を残した競馬場の格式度と勝率の関係がより明確になります。\n")
            f.write("3年間隔での時系列分析により、傾向の変化も把握できます。\n\n")
            f.write("**平均重み付けポイント**と**累積重み付けポイント（合計）**の両方で分析を実施し、\n")
            f.write("馬の質（格式高い競馬場での安定した成績）と量（総実績ポイント）の両面から評価しています。\n\n")
            f.write("### 🔍 分析手法の改善\n")
            f.write("累積ポイント分析では**勝利数の影響による偏り**を除去するため、以下の補正分析を実施：\n")
            f.write("- **出走数正規化分析**: 累積ポイントを出走数で割った値での相関分析\n")
            f.write("- **部分相関分析**: 複勝回数の影響を統計的に統制した相関分析\n")
            f.write("これにより、単なる「複勝回数の多さ」ではない、真の競馬場格式度と成績の関係を評価します。\n\n")
            
            f.write("## 📊 分析期間一覧\n\n")
            f.write("| 期間 | 対象馬数 | 総レース数 | 複勝経験馬数 | 主要統計 |\n")
            f.write("|------|----------|-----------|-------------|----------|\n")
            
            for period_name, period_results in results.items():
                total_horses = period_results.get('total_horses', 0)
                total_races = period_results.get('total_races', 0)
                
                # 複勝経験馬数
                if 'race_point_correlation' in period_results and 'place_experienced_horses' in period_results['race_point_correlation']:
                    place_exp_horses = len(period_results['race_point_correlation']['place_experienced_horses'])
                else:
                    place_exp_horses = 0
                
                # 主要統計
                if ('race_point_correlation' in period_results and 
                    'correlation_analysis' in period_results['race_point_correlation']):
                    corr_data = period_results['race_point_correlation']['correlation_analysis']
                    main_stat = f"r={corr_data['avg_point']['win_rate']['correlation']:.3f}"
                else:
                    main_stat = "データ不足"
                
                f.write(f"| {period_name} | {total_horses:,}頭 | {total_races:,}レース | {place_exp_horses:,}頭 | {main_stat} |\n")
            
            # 各期間の詳細
            for period_name, period_results in results.items():
                f.write(f"\n## 📈 期間: {period_name}\n\n")
                
                if ('race_point_correlation' in period_results and 
                    'correlation_analysis' in period_results['race_point_correlation']):
                    corr_analysis = period_results['race_point_correlation']['correlation_analysis']
                    place_exp_horses = period_results['race_point_correlation'].get('place_experienced_horses')
                    
                    if place_exp_horses is not None and len(place_exp_horses) >= 3:
                        f.write(f"### 複勝時重み付けポイント相関分析（複勝経験馬 {len(place_exp_horses)}頭）\n\n")
                        
                        # 平均ポイント分析結果
                        avg_win_corr = corr_analysis['avg_point']['win_rate']['correlation']
                        avg_win_p = corr_analysis['avg_point']['win_rate']['p_value']
                        avg_win_r2 = corr_analysis['avg_point']['win_rate']['r2']
                        
                        avg_place_corr = corr_analysis['avg_point']['place_rate']['correlation']
                        avg_place_p = corr_analysis['avg_point']['place_rate']['p_value']
                        avg_place_r2 = corr_analysis['avg_point']['place_rate']['r2']
                        
                        # 累積ポイント分析結果
                        cum_win_corr = corr_analysis['cumulative']['win_rate']['correlation']
                        cum_win_p = corr_analysis['cumulative']['win_rate']['p_value']
                        cum_win_r2 = corr_analysis['cumulative']['win_rate']['r2']
                        
                        cum_place_corr = corr_analysis['cumulative']['place_rate']['correlation']
                        cum_place_p = corr_analysis['cumulative']['place_rate']['p_value']
                        cum_place_r2 = corr_analysis['cumulative']['place_rate']['r2']
                        
                        f.write("#### 相関分析結果\n\n")
                        f.write("**平均重み付けポイント分析**\n")
                        f.write(f"- **複勝時平均ポイント vs 勝率**: r = {avg_win_corr:.3f}, p = {avg_win_p:.3f}, R² = {avg_win_r2:.3f}")
                        f.write(f" ({'有意' if avg_win_p < 0.05 else '非有意'})\n")
                        f.write(f"- **複勝時平均ポイント vs 複勝率**: r = {avg_place_corr:.3f}, p = {avg_place_p:.3f}, R² = {avg_place_r2:.3f}")
                        f.write(f" ({'有意' if avg_place_p < 0.05 else '非有意'})\n\n")
                        
                        f.write("**累積重み付けポイント（合計）分析**\n")
                        f.write(f"- **複勝時累積ポイント vs 勝率**: r = {cum_win_corr:.3f}, p = {cum_win_p:.3f}, R² = {cum_win_r2:.3f}")
                        f.write(f" ({'有意' if cum_win_p < 0.05 else '非有意'})\n")
                        f.write(f"- **複勝時累積ポイント vs 複勝率**: r = {cum_place_corr:.3f}, p = {cum_place_p:.3f}, R² = {cum_place_r2:.3f}")
                        f.write(f" ({'有意' if cum_place_p < 0.05 else '非有意'})\n\n")
                        
                        # 上位馬の分析（平均ポイント）
                        f.write("#### 複勝時平均重み付けポイント上位5頭\n")
                        f.write("| 順位 | 馬名 | 平均ポイント | 累積ポイント | 勝率 | 複勝率 | 複勝回数 | 出走数 |\n")
                        f.write("|------|------|-------------|-------------|------|--------|----------|--------|\n")
                        
                        top_horses_avg = place_exp_horses.nlargest(5, '平均重み付けポイント')
                        for i, (horse_name, row) in enumerate(top_horses_avg.iterrows(), 1):
                            f.write(f"| {i} | {horse_name} | {row['平均重み付けポイント']:.1f} | {row['累積重み付けポイント']:.1f} | {row['勝率']:.3f} | "
                                   f"{row['複勝率']:.3f} | {row['複勝回数']} | {row['出走数']} |\n")
                        
                        # 上位馬の分析（累積ポイント）
                        f.write("\n#### 複勝時累積重み付けポイント上位5頭\n")
                        f.write("| 順位 | 馬名 | 累積ポイント | 平均ポイント | 勝率 | 複勝率 | 複勝回数 | 出走数 |\n")
                        f.write("|------|------|-------------|-------------|------|--------|----------|--------|\n")
                        
                        top_horses_cumulative = place_exp_horses.nlargest(5, '累積重み付けポイント')
                        for i, (horse_name, row) in enumerate(top_horses_cumulative.iterrows(), 1):
                            f.write(f"| {i} | {horse_name} | {row['累積重み付けポイント']:.1f} | {row['平均重み付けポイント']:.1f} | {row['勝率']:.3f} | "
                                   f"{row['複勝率']:.3f} | {row['複勝回数']} | {row['出走数']} |\n")
                        
                        # 正規化分析結果（勝利数影響除去）
                        if 'normalized_point' in corr_analysis:
                            norm_win_corr = corr_analysis['normalized_point']['win_rate']['correlation']
                            norm_win_p = corr_analysis['normalized_point']['win_rate']['p_value']
                            norm_win_r2 = corr_analysis['normalized_point']['win_rate']['r2']
                            
                            norm_place_corr = corr_analysis['normalized_point']['place_rate']['correlation']
                            norm_place_p = corr_analysis['normalized_point']['place_rate']['p_value']
                            norm_place_r2 = corr_analysis['normalized_point']['place_rate']['r2']
                            
                            f.write("**出走数正規化重み付けポイント分析（勝利数影響除去）**\n")
                            f.write(f"- **正規化累積ポイント vs 勝率**: r = {norm_win_corr:.3f}, p = {norm_win_p:.3f}, R² = {norm_win_r2:.3f}")
                            f.write(f" ({'有意' if norm_win_p < 0.05 else '非有意'})\n")
                            f.write(f"- **正規化累積ポイント vs 複勝率**: r = {norm_place_corr:.3f}, p = {norm_place_p:.3f}, R² = {norm_place_r2:.3f}")
                            f.write(f" ({'有意' if norm_place_p < 0.05 else '非有意'})\n\n")
                        
                        # 部分相関分析結果
                        if 'partial_correlation' in corr_analysis:
                            partial_win_corr = corr_analysis['partial_correlation']['cumulative_vs_win_rate']
                            partial_place_corr = corr_analysis['partial_correlation']['cumulative_vs_place_rate']
                            
                            f.write("**部分相関分析（複勝回数の影響を統制）**\n")
                            f.write(f"- **累積ポイント vs 勝率（複勝回数統制）**: r = {partial_win_corr:.3f}\n")
                            f.write(f"- **累積ポイント vs 複勝率（複勝回数統制）**: r = {partial_place_corr:.3f}\n\n")
                            
                            f.write("⚠️ **重要な分析ポイント**\n")
                            f.write("- 累積ポイントと複勝率の高い相関は、複勝回数が多いほど累積ポイントも複勝率も高くなる構造的要因\n")
                            f.write("- 出走数正規化や部分相関により、真の競馬場格式度と成績の関係を評価\n")
                            f.write("- 正規化分析により勝利数の偏りの影響を除去した、より適切な評価が可能\n\n")
                    else:
                        f.write("この期間は複勝経験馬が3頭未満のため、相関分析を実行できませんでした。\n")
                else:
                    f.write("この期間はデータが不足しているため、統計分析を実行できませんでした。\n")
            
            # 全体的な傾向
            f.write("\n## 💡 全体的な傾向と知見\n\n")
            
            # 期間別の相関係数変化
            correlations_by_period = []
            for period_name, period_results in results.items():
                if ('race_point_correlation' in period_results and 
                    'correlation_analysis' in period_results['race_point_correlation']):
                    corr = period_results['race_point_correlation']['correlation_analysis']['avg_point']['win_rate']['correlation']
                    correlations_by_period.append((period_name, corr))
            
            if len(correlations_by_period) > 1:
                f.write("### 時系列変化\n")
                f.write("| 期間 | 相関係数（複勝時ポイント vs 勝率） | 傾向 |\n")
                f.write("|------|---------------------------|------|\n")
                
                for i, (period, corr) in enumerate(correlations_by_period):
                    if i > 0:
                        prev_corr = correlations_by_period[i-1][1]
                        trend = "上昇" if corr > prev_corr else "下降"
                    else:
                        trend = "基準"
                    f.write(f"| {period} | {corr:.3f} | {trend} |\n")
            
            f.write("\n### 複勝時重み付けポイントの特徴\n")
            f.write("- **複勝した場合のみポイント加算**により、実際の成績と競馬場格式の関係がより明確になります\n")
            f.write("- 複勝経験がない馬は重み付けポイント0となり、成績との相関がより純粋に評価できます\n")
            f.write("- 高格式競馬場での複勝経験は、その馬の実力の証明として機能します\n")
            f.write("- **平均ポイント**: 馬の質（どれだけ格式高い競馬場で勝っているか）を表す\n")
            f.write("- **累積ポイント**: 馬の量（どれだけ多くの実績を積んでいるか）を表す\n")
            
            f.write("\n### 実用的示唆\n")
            f.write("- 複勝時重み付けポイントは、馬の「実証済み格」を表す指標として有効です\n")
            f.write("- 高ポイントを持つ馬は、格式の高い競馬場で実際に結果を残した経験があります\n")
            f.write("- **平均ポイント**は新馬や経験の浅い馬の実力評価に有効\n")
            f.write("- **累積ポイント**は古馬や経験豊富な馬の総合力評価に有効\n")
            f.write("- この指標は馬券購入時の判断材料として活用できます\n")
        
        logger.info(f"メインレポート保存: {main_report_path}")

    def _plot_correlation_scatter_with_warning(self, horse_stats, title, x_col, y_col, correlation_data, output_path):
        """相関散布図の描画（警告付き）"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # データの準備
        x = horse_stats[x_col]
        y = horse_stats[y_col]
        
        # レース回数（出走数）に基づくサイズ設定（より明確に）
        min_size = 30
        max_size = 300
        race_counts = horse_stats['出走数']
        # レース回数を正規化してサイズに変換
        normalized_sizes = min_size + (race_counts - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
        
        # 散布図
        scatter = ax.scatter(x, y, c=horse_stats['出走数'], s=normalized_sizes, alpha=0.6, 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('出走数', fontsize=12)
        
        # 回帰直線
        regression = correlation_data['regression']
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred = regression.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r--', linewidth=2, 
               label=f'回帰直線 (R² = {correlation_data["r2"]:.3f})')
        
        # 統計情報の表示（警告付き）
        corr = correlation_data['correlation']
        p_val = correlation_data['p_value']
        
        stats_text = f'相関係数: {corr:.3f}\n'
        stats_text += f'p値: {p_val:.3f}\n' 
        stats_text += f'有意性: {"有意" if p_val < 0.05 else "非有意"}\n'
        stats_text += f'対象: 複勝経験馬のみ ({len(horse_stats)}頭)\n\n'
        stats_text += '⚠️ 警告:\n'
        stats_text += 'この指標は構造的に高い相関を\n'
        stats_text += '示すため、解釈に注意が必要です'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        # レース回数のサイズ凡例を追加
        sizes_for_legend = [race_counts.min(), race_counts.quantile(0.5), race_counts.max()]
        labels_for_legend = [f'{int(size)}回' for size in sizes_for_legend]
        
        # サイズ凡例用のマーカーサイズを計算
        legend_sizes = []
        for size in sizes_for_legend:
            normalized_size = min_size + (size - race_counts.min()) / (race_counts.max() - race_counts.min()) * (max_size - min_size)
            legend_sizes.append(normalized_size)
        
        # サイズ凡例の作成
        legend_elements = []
        for size, label, marker_size in zip(sizes_for_legend, labels_for_legend, legend_sizes):
            legend_elements.append(plt.scatter([], [], s=marker_size, c='gray', alpha=0.6, 
                                             edgecolors='black', linewidth=0.5, label=label))
        
        # 既存の凡例と組み合わせ
        legend1 = plt.legend(handles=legend_elements, title="出走数（点のサイズ）", 
                           loc='upper left', bbox_to_anchor=(0, 0.75), frameon=True, fancybox=True, shadow=True)
        plt.gca().add_artist(legend1)
        
        # 回帰直線の凡例
        legend2 = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # グラフ左下に出走数の範囲情報を追加
        info_text = f"出走数範囲: {int(race_counts.min())}～{int(race_counts.max())}回\n平均: {race_counts.mean():.1f}回"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom', fontsize=10)
        
        # ラベルとタイトル
        ax.set_xlabel(f'{x_col} ※複勝した場合のみポイント加算', fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{title}（複勝経験馬のみ）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.warning(f"⚠️  警告付き相関図を保存: {output_path}")

    def compare_random_vs_original_weights(self):
        """オリジナル重み付けとランダム重み付けの比較分析"""
        logger.info("🔄 オリジナル vs ランダム重み付け比較分析を開始...")
        
        # データの前処理
        self.df = self.preprocess_data()
        
        comparison_results = {}
        
        # 分析期間の設定
        if self.df['年'].notna().any():
            years = sorted(self.df['年'].dropna().unique())
            if len(years) >= 6:
                periods = [
                    ('2022-2024', years[-3:]),
                    ('2019-2021', years[-6:-3]),
                    ('全期間', years)
                ]
            else:
                periods = [('全期間', years)]
        else:
            periods = [('全期間', None)]
        
        for period_name, period_years in periods:
            logger.info(f"\n📊 期間: {period_name}")
            
            # 期間データの抽出
            if period_years:
                df_period = self.df[self.df['年'].isin(period_years)]
            else:
                df_period = self.df.copy()
            
            df_period = df_period[df_period.groupby('馬名')['馬名'].transform('count') >= self.config['min_races']]
            
            if len(df_period) == 0:
                logger.warning(f"期間 {period_name} にデータがありません")
                continue
            
            # 1. オリジナル重み付けでの分析
            logger.info("📊 オリジナル重み付けでの分析...")
            original_horse_stats = self._calculate_horse_race_point_stats(df_period)
            original_correlation = self._analyze_race_point_correlation(original_horse_stats)
            
            # 2. ランダム重み付けに変更
            self._apply_random_weights()
            
            # データを再計算（競馬場重み付けポイントを更新）
            df_period_random = df_period.copy()
            df_period_random['競馬場重み付けポイント'] = df_period_random['場名'].map(
                lambda x: self.track_hierarchy.get(x, {}).get('weight_points', 0)
            )
            
            # 3. ランダム重み付けでの分析
            logger.info("🎲 ランダム重み付けでの分析...")
            random_horse_stats = self._calculate_horse_race_point_stats(df_period_random)
            random_correlation = self._analyze_race_point_correlation(random_horse_stats)
            
            # 4. 結果の比較
            comparison_results[period_name] = {
                'original': {
                    'horse_stats': original_horse_stats,
                    'correlation': original_correlation
                },
                'random': {
                    'horse_stats': random_horse_stats,  
                    'correlation': random_correlation
                },
                'period_years': period_years,
                'total_horses': len(df_period['馬名'].unique()),
                'total_races': len(df_period)
            }
            
            # オリジナル重み付けに戻す
            self.track_hierarchy = copy.deepcopy(self.original_track_hierarchy)
            
        return comparison_results
    
    def visualize_comparison_results(self, comparison_results):
        """比較分析結果の可視化"""
        output_dir = Path(self.config['output_dir'])
        comparison_dir = output_dir / 'random_vs_original_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        for period_name, results in comparison_results.items():
            period_dir = comparison_dir / period_name.replace('/', '_')
            period_dir.mkdir(parents=True, exist_ok=True)
            
            # 比較チャートの作成
            self._plot_comparison_charts(results, period_dir, period_name)
            
            logger.info(f"📊 比較結果保存: {period_dir}")
    
    def _plot_comparison_charts(self, results, output_dir, period_name):
        """比較チャートの描画"""
        
        original_results = results['original']
        random_results = results['random']
        
        # 複勝経験馬のデータを取得
        original_experienced = original_results['correlation']['place_experienced_horses']
        random_experienced = random_results['correlation']['place_experienced_horses']
        
        # 相関結果を取得
        original_corr = original_results['correlation']
        random_corr = random_results['correlation']
        
        # 出走数正規化累積ポイントカラムが存在しない場合は作成
        if '出走数正規化累積ポイント' not in original_experienced.columns:
            original_experienced = original_experienced.copy()
            original_experienced['出走数正規化累積ポイント'] = original_experienced['累積重み付けポイント'] / original_experienced['出走数']
        
        if '出走数正規化累積ポイント' not in random_experienced.columns:
            random_experienced = random_experienced.copy()
            random_experienced['出走数正規化累積ポイント'] = random_experienced['累積重み付けポイント'] / random_experienced['出走数']
        
        # === 1. 平均重み付けポイント vs 複勝率の比較チャート ===
        self._create_single_comparison_chart(
            original_experienced, random_experienced,
            original_corr, random_corr,
            '平均重み付けポイント', '複勝率',
            'avg_point', 'place_rate',
            f'平均重み付けポイント vs 複勝率比較 ({period_name})',
            output_dir / f'平均重み付けポイントと複勝率の関係比較_{period_name}.png'
        )
        
        # === 2. 出走数正規化累積ポイント vs 複勝率の比較チャート ===
        self._create_single_comparison_chart(
            original_experienced, random_experienced,
            original_corr, random_corr,
            '出走数正規化累積ポイント', '複勝率',
            'normalized', 'place_rate',
            f'出走数正規化累積ポイント vs 複勝率比較 ({period_name})',
            output_dir / f'出走数正規化累積ポイントと複勝率の関係比較_{period_name}.png'
        )
        
        # === 3. 4象限総合比較チャート（従来版を維持） ===
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # オリジナル重み付け散布図
        x_orig = original_experienced['出走数正規化累積ポイント']
        y_orig = original_experienced['複勝率']
        
        # レース回数（出走数）に基づくサイズ設定
        min_size = 30
        max_size = 150
        race_counts_orig = original_experienced['出走数']
        normalized_sizes_orig = min_size + (race_counts_orig - race_counts_orig.min()) / (race_counts_orig.max() - race_counts_orig.min()) * (max_size - min_size)
        
        scatter_orig = ax1.scatter(x_orig, y_orig, alpha=0.6, c=race_counts_orig, s=normalized_sizes_orig, 
                                 cmap='Blues', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar1 = plt.colorbar(scatter_orig, ax=ax1)
        cbar1.set_label('出走数', fontsize=10)
        
        # 相関情報の取得
        if ('correlation_analysis' in original_corr and 
            'normalized' in original_corr['correlation_analysis'] and
            'place_rate' in original_corr['correlation_analysis']['normalized']):
            orig_corr_val = original_corr['correlation_analysis']['normalized']['place_rate']['correlation']
            orig_p_val = original_corr['correlation_analysis']['normalized']['place_rate']['p_value']
            orig_regression = original_corr['correlation_analysis']['normalized']['place_rate']['regression']
        else:
            from scipy.stats import pearsonr
            from sklearn.linear_model import LinearRegression
            orig_corr_val, orig_p_val = pearsonr(x_orig, y_orig)
            # 回帰直線を計算
            orig_regression = LinearRegression()
            orig_regression.fit(x_orig.values.reshape(-1, 1), y_orig.values)
        
        # オリジナルの回帰直線を描画
        x_range_orig = np.linspace(x_orig.min(), x_orig.max(), 100)
        y_pred_orig = orig_regression.predict(x_range_orig.reshape(-1, 1))
        ax1.plot(x_range_orig, y_pred_orig, 'r--', linewidth=2, 
               label=f'回帰直線 (r = {orig_corr_val:.3f})')
        
        ax1.set_title(f'オリジナル重み付け\nr = {orig_corr_val:.3f}, p = {orig_p_val:.3f}', fontsize=12)
        ax1.set_xlabel('出走数正規化累積ポイント')
        ax1.set_ylabel('複勝率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ランダム重み付け散布図
        x_rand = random_experienced['出走数正規化累積ポイント']
        y_rand = random_experienced['複勝率']
        
        # ランダム重み付け用のサイズ設定
        race_counts_rand = random_experienced['出走数']
        normalized_sizes_rand = min_size + (race_counts_rand - race_counts_rand.min()) / (race_counts_rand.max() - race_counts_rand.min()) * (max_size - min_size)
        
        scatter_rand = ax2.scatter(x_rand, y_rand, alpha=0.6, c=race_counts_rand, s=normalized_sizes_rand, 
                                 cmap='Reds', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar2 = plt.colorbar(scatter_rand, ax=ax2)
        cbar2.set_label('出走数', fontsize=10)
        
        if ('correlation_analysis' in random_corr and 
            'normalized' in random_corr['correlation_analysis'] and
            'place_rate' in random_corr['correlation_analysis']['normalized']):
            rand_corr_val = random_corr['correlation_analysis']['normalized']['place_rate']['correlation']
            rand_p_val = random_corr['correlation_analysis']['normalized']['place_rate']['p_value']
            rand_regression = random_corr['correlation_analysis']['normalized']['place_rate']['regression']
        else:
            from scipy.stats import pearsonr
            from sklearn.linear_model import LinearRegression
            rand_corr_val, rand_p_val = pearsonr(x_rand, y_rand)
            # 回帰直線を計算
            rand_regression = LinearRegression()
            rand_regression.fit(x_rand.values.reshape(-1, 1), y_rand.values)
        
        # ランダムの回帰直線を描画
        x_range_rand = np.linspace(x_rand.min(), x_rand.max(), 100)
        y_pred_rand = rand_regression.predict(x_range_rand.reshape(-1, 1))
        ax2.plot(x_range_rand, y_pred_rand, 'b--', linewidth=2, 
               label=f'回帰直線 (r = {rand_corr_val:.3f})')
        
        ax2.set_title(f'ランダム重み付け\nr = {rand_corr_val:.3f}, p = {rand_p_val:.3f}', fontsize=12)
        ax2.set_xlabel('出走数正規化累積ポイント')
        ax2.set_ylabel('複勝率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 相関係数比較棒グラフ
        categories = ['オリジナル', 'ランダム']
        correlations = [orig_corr_val, rand_corr_val]
        colors = ['blue', 'red']
        
        bars = ax3.bar(categories, correlations, color=colors, alpha=0.7)
        ax3.set_title('相関係数比較', fontsize=12)
        ax3.set_ylabel('相関係数')
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # 差の可視化
        diff = abs(orig_corr_val) - abs(rand_corr_val)
        
        # 出走数統計情報を追加
        orig_race_stats = f"出走数範囲: {int(race_counts_orig.min())}～{int(race_counts_orig.max())}回\n平均: {race_counts_orig.mean():.1f}回"
        rand_race_stats = f"出走数範囲: {int(race_counts_rand.min())}～{int(race_counts_rand.max())}回\n平均: {race_counts_rand.mean():.1f}回"
        
        ax4.text(0.5, 0.7, f'相関係数の差\n（絶対値）:\n{diff:.3f}', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        ax4.text(0.5, 0.4, f'オリジナル重み付け\n{orig_race_stats}', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax4.text(0.5, 0.1, f'ランダム重み付け\n{rand_race_stats}', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle(f'総合比較: 出走数正規化累積ポイント vs 複勝率 ({period_name})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'総合比較_出走数正規化累積ポイント_{period_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"総合比較チャート保存: {output_dir / f'総合比較_出走数正規化累積ポイント_{period_name}.png'}")

    def _create_single_comparison_chart(self, original_data, random_data, 
                                      original_corr, random_corr,
                                      x_col, y_col, corr_type, target_type,
                                      title, output_path):
        """単一の比較チャート作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # オリジナル重み付け散布図
        x_orig = original_data[x_col]
        y_orig = original_data[y_col]
        
        # レース回数（出走数）に基づくサイズ設定
        min_size = 30
        max_size = 150
        race_counts_orig = original_data['出走数']
        normalized_sizes_orig = min_size + (race_counts_orig - race_counts_orig.min()) / (race_counts_orig.max() - race_counts_orig.min()) * (max_size - min_size)
        
        scatter_orig = ax1.scatter(x_orig, y_orig, alpha=0.6, c=race_counts_orig, s=normalized_sizes_orig, 
                                 cmap='Blues', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar1 = plt.colorbar(scatter_orig, ax=ax1)
        cbar1.set_label('出走数', fontsize=10)
        
        # 相関情報の取得（オリジナル）
        if ('correlation_analysis' in original_corr and 
            corr_type in original_corr['correlation_analysis'] and
            target_type in original_corr['correlation_analysis'][corr_type]):
            orig_corr_val = original_corr['correlation_analysis'][corr_type][target_type]['correlation']
            orig_p_val = original_corr['correlation_analysis'][corr_type][target_type]['p_value']
            orig_regression = original_corr['correlation_analysis'][corr_type][target_type]['regression']
        else:
            from scipy.stats import pearsonr
            from sklearn.linear_model import LinearRegression
            orig_corr_val, orig_p_val = pearsonr(x_orig, y_orig)
            # 回帰直線を計算
            orig_regression = LinearRegression()
            orig_regression.fit(x_orig.values.reshape(-1, 1), y_orig.values)
        
        # オリジナルの回帰直線を描画
        x_range_orig = np.linspace(x_orig.min(), x_orig.max(), 100)
        y_pred_orig = orig_regression.predict(x_range_orig.reshape(-1, 1))
        ax1.plot(x_range_orig, y_pred_orig, 'r--', linewidth=2, 
               label=f'回帰直線 (r = {orig_corr_val:.3f})')
        
        ax1.set_title(f'オリジナル重み付け\nr = {orig_corr_val:.3f}, p = {orig_p_val:.3f}', fontsize=12)
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ランダム重み付け散布図
        x_rand = random_data[x_col]
        y_rand = random_data[y_col]
        
        # ランダム重み付け用のサイズ設定
        race_counts_rand = random_data['出走数']
        normalized_sizes_rand = min_size + (race_counts_rand - race_counts_rand.min()) / (race_counts_rand.max() - race_counts_rand.min()) * (max_size - min_size)
        
        scatter_rand = ax2.scatter(x_rand, y_rand, alpha=0.6, c=race_counts_rand, s=normalized_sizes_rand, 
                                 cmap='Reds', edgecolors='black', linewidth=0.5)
        
        # カラーバー
        cbar2 = plt.colorbar(scatter_rand, ax=ax2)
        cbar2.set_label('出走数', fontsize=10)
        
        # 相関情報の取得（ランダム）
        if ('correlation_analysis' in random_corr and 
            corr_type in random_corr['correlation_analysis'] and
            target_type in random_corr['correlation_analysis'][corr_type]):
            rand_corr_val = random_corr['correlation_analysis'][corr_type][target_type]['correlation']
            rand_p_val = random_corr['correlation_analysis'][corr_type][target_type]['p_value']
            rand_regression = random_corr['correlation_analysis'][corr_type][target_type]['regression']
        else:
            from scipy.stats import pearsonr
            from sklearn.linear_model import LinearRegression
            rand_corr_val, rand_p_val = pearsonr(x_rand, y_rand)
            # 回帰直線を計算
            rand_regression = LinearRegression()
            rand_regression.fit(x_rand.values.reshape(-1, 1), y_rand.values)
        
        # ランダムの回帰直線を描画
        x_range_rand = np.linspace(x_rand.min(), x_rand.max(), 100)
        y_pred_rand = rand_regression.predict(x_range_rand.reshape(-1, 1))
        ax2.plot(x_range_rand, y_pred_rand, 'b--', linewidth=2, 
               label=f'回帰直線 (r = {rand_corr_val:.3f})')
        
        ax2.set_title(f'ランダム重み付け\nr = {rand_corr_val:.3f}, p = {rand_p_val:.3f}', fontsize=12)
        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 相関係数比較棒グラフ
        categories = ['オリジナル', 'ランダム']
        correlations = [orig_corr_val, rand_corr_val]
        colors = ['blue', 'red']
        
        bars = ax3.bar(categories, correlations, color=colors, alpha=0.7)
        ax3.set_title('相関係数比較', fontsize=12)
        ax3.set_ylabel('相関係数')
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # 差の可視化と評価
        diff = abs(orig_corr_val) - abs(rand_corr_val)
        
        # 評価結果
        if diff > 0.1:
            verdict = "✅ オリジナル明らかに優秀"
            color = 'lightgreen'
        elif diff > 0:
            verdict = "➕ オリジナルがわずかに優秀"
            color = 'lightblue'
        elif diff > -0.1:
            verdict = "➖ ほぼ同等"
            color = 'lightyellow'
        else:
            verdict = "❌ ランダムが優秀"
            color = 'lightcoral'
        
        ax4.text(0.5, 0.7, f'相関係数の差\n（絶対値）:\n{diff:.3f}', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        ax4.text(0.5, 0.5, verdict, 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor=color))
        
        # 回帰係数情報も追加
        orig_slope = orig_regression.coef_[0]
        rand_slope = rand_regression.coef_[0]
        
        regression_text = f"回帰係数:\nオリジナル: {orig_slope:.4f}\nランダム: {rand_slope:.4f}"
        ax4.text(0.5, 0.3, regression_text, 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # 統計情報
        orig_race_stats = f"オリジナル出走数: {int(race_counts_orig.min())}～{int(race_counts_orig.max())}回"
        rand_race_stats = f"ランダム出走数: {int(race_counts_rand.min())}～{int(race_counts_rand.max())}回"
        
        ax4.text(0.5, 0.1, f'{orig_race_stats}\n{rand_race_stats}', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比較チャート保存: {output_path}")

    def generate_comparison_report(self, comparison_results):
        """比較分析レポートの生成"""
        output_dir = Path(self.config['output_dir'])
        comparison_dir = output_dir / 'random_vs_original_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = comparison_dir / '重み付け比較分析レポート.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 競馬場重み付け比較分析レポート\n\n")
            f.write("## 概要\n")
            f.write("競馬場の重み付けポイントを論理的な設定（オリジナル）とランダムな設定で比較し、\n")
            f.write("どちらが複勝率との相関が強いかを検証した結果です。\n\n")
            
            f.write("## 検証方法\n")
            f.write("1. **オリジナル重み付け**: 競馬場の格式・威信度・設備等を考慮した論理的な重み付け\n")
            f.write("2. **ランダム重み付け**: オリジナルの重み付け値をランダムに競馬場に再配分\n")
            f.write("3. 両方式で「出走数正規化累積ポイント」と「複勝率」の相関係数を計算\n")
            f.write("4. 相関の強さを比較して重み付けシステムの有効性を評価\n\n")
            
            f.write("## 期待される結果\n")
            f.write("- **オリジナル重み付けが優秀な場合**: 論理的な重み付けが競馬の実態を正しく反映\n")
            f.write("- **ランダム重み付けと同等の場合**: 重み付けシステムに改善の余地あり\n")
            f.write("- **ランダム重み付けが優秀な場合**: 論理的重み付けに根本的な問題あり\n\n")
            
            f.write("## 分析結果\n\n")
            
            summary_table = []
            overall_orig_better = 0
            overall_total = 0
            
            for period_name, results in comparison_results.items():
                f.write(f"### {period_name}\n\n")
                
                # 基本情報
                total_horses = results['total_horses']
                total_races = results['total_races']
                f.write(f"- **対象データ**: {total_horses:,}頭, {total_races:,}レース\n")
                
                # オリジナル結果
                orig_corr = results['original']['correlation']
                orig_experienced = results['original']['horse_stats'][
                    results['original']['horse_stats']['平均重み付けポイント'] > 0.0]
                
                # 出走数正規化累積ポイントカラムが存在しない場合は作成
                if '出走数正規化累積ポイント' not in orig_experienced.columns:
                    orig_experienced = orig_experienced.copy()
                    orig_experienced['出走数正規化累積ポイント'] = orig_experienced['累積重み付けポイント'] / orig_experienced['出走数']
                
                if ('correlation_analysis' in orig_corr and 
                    'normalized' in orig_corr['correlation_analysis'] and
                    'place_rate' in orig_corr['correlation_analysis']['normalized']):
                    orig_r = orig_corr['correlation_analysis']['normalized']['place_rate']['correlation']
                    orig_p = orig_corr['correlation_analysis']['normalized']['place_rate']['p_value']
                else:
                    from scipy.stats import pearsonr
                    x_orig = orig_experienced['出走数正規化累積ポイント']
                    y_orig = orig_experienced['複勝率']
                    orig_r, orig_p = pearsonr(x_orig, y_orig)
                
                # ランダム結果
                rand_corr = results['random']['correlation']
                rand_experienced = results['random']['horse_stats'][
                    results['random']['horse_stats']['平均重み付けポイント'] > 0.0]
                
                # 出走数正規化累積ポイントカラムが存在しない場合は作成
                if '出走数正規化累積ポイント' not in rand_experienced.columns:
                    rand_experienced = rand_experienced.copy()
                    rand_experienced['出走数正規化累積ポイント'] = rand_experienced['累積重み付けポイント'] / rand_experienced['出走数']
                
                if ('correlation_analysis' in rand_corr and 
                    'normalized' in rand_corr['correlation_analysis'] and
                    'place_rate' in rand_corr['correlation_analysis']['normalized']):
                    rand_r = rand_corr['correlation_analysis']['normalized']['place_rate']['correlation']
                    rand_p = rand_corr['correlation_analysis']['normalized']['place_rate']['p_value']
                else:
                    from scipy.stats import pearsonr
                    x_rand = rand_experienced['出走数正規化累積ポイント']
                    y_rand = rand_experienced['複勝率']
                    rand_r, rand_p = pearsonr(x_rand, y_rand)
                
                f.write(f"- **複勝経験馬**: オリジナル {len(orig_experienced)}頭, ランダム {len(rand_experienced)}頭\n\n")
                
                f.write("#### 相関分析結果\n\n")
                f.write("| 重み付け方式 | 相関係数 | p値 | 有意性 |\n")
                f.write("|-------------|---------|-----|--------|\n")
                f.write(f"| オリジナル | {orig_r:.3f} | {orig_p:.3f} | {'有意' if orig_p < 0.05 else '非有意'} |\n")
                f.write(f"| ランダム | {rand_r:.3f} | {rand_p:.3f} | {'有意' if rand_p < 0.05 else '非有意'} |\n\n")
                
                # 差の計算と評価
                diff = abs(orig_r) - abs(rand_r)
                f.write(f"#### 評価\n")
                f.write(f"- **相関係数の差（絶対値）**: {diff:.3f}\n")
                
                if diff > 0.1:
                    verdict = "✅ **オリジナル重み付けが明らかに優秀**"
                    explanation = "論理的重み付けが競馬の実態を正確に反映している"
                    orig_better = True
                elif diff > 0:
                    verdict = "➕ **オリジナル重み付けがわずかに優秀**"
                    explanation = "論理的重み付けに一定の効果がある"
                    orig_better = True
                elif diff > -0.1:
                    verdict = "➖ **ほぼ同等**"
                    explanation = "重み付けシステムの改善余地がある"
                    orig_better = False
                else:
                    verdict = "❌ **ランダム重み付けが優秀**"
                    explanation = "論理的重み付けに根本的な問題がある可能性"
                    orig_better = False
                
                f.write(f"- {verdict}\n")
                f.write(f"- **解釈**: {explanation}\n\n")
                
                # サマリー用データ
                summary_table.append({
                    'period': period_name,
                    'orig_r': orig_r,
                    'rand_r': rand_r,
                    'diff': diff,
                    'orig_better': orig_better
                })
                
                if orig_better:
                    overall_orig_better += 1
                overall_total += 1
            
            # 全体サマリー
            f.write("## 総合評価\n\n")
            f.write("### 期間別結果一覧\n\n")
            f.write("| 期間 | オリジナル相関 | ランダム相関 | 差 | 評価 |\n")
            f.write("|------|---------------|-------------|----|---------|\n")
            
            for result in summary_table:
                status = "🟢 優秀" if result['orig_better'] else "🔴 劣勢"
                f.write(f"| {result['period']} | {result['orig_r']:.3f} | {result['rand_r']:.3f} | {result['diff']:.3f} | {status} |\n")
            
            f.write(f"\n### 最終結論\n")
            success_rate = (overall_orig_better / overall_total) * 100 if overall_total > 0 else 0
            f.write(f"- **オリジナル重み付けが優秀な期間**: {overall_orig_better}/{overall_total} ({success_rate:.1f}%)\n")
            
            if success_rate >= 80:
                final_verdict = "🎉 **論理的重み付けシステムは非常に有効**"
                recommendation = "現在の重み付けシステムを継続使用することを推奨"
            elif success_rate >= 60:
                final_verdict = "👍 **論理的重み付けシステムは概ね有効**"
                recommendation = "細部の調整により、さらなる改善が期待できる"
            elif success_rate >= 40:
                final_verdict = "⚠️ **論理的重み付けシステムは部分的に有効**"
                recommendation = "重み付けロジックの見直しが必要"
            else:
                final_verdict = "❌ **論理的重み付けシステムに問題あり**"
                recommendation = "重み付けシステムの根本的な再設計が必要"
            
            f.write(f"- {final_verdict}\n")
            f.write(f"- **推奨**: {recommendation}\n\n")
            
            f.write("### 統計的意義\n")
            f.write("この比較検証により、以下が確認できます：\n")
            f.write("1. **重み付けシステムの有効性**: 論理的重み付けがランダムより優秀かどうか\n")
            f.write("2. **競馬場格式の妥当性**: 設定した競馬場格式が実態と合致しているか\n")
            f.write("3. **予測モデルの信頼性**: このシステムが実用的な予測に使えるか\n\n")
            
            f.write("### 注意事項\n")
            f.write("- この分析は複勝経験のある馬のみを対象としています\n")
            f.write("- ランダム重み付けは1回の実行結果であり、複数回の平均ではありません\n")
            f.write("- 相関係数が高くても因果関係を保証するものではありません\n")
            f.write("- 実際の馬券購入時は他の要因も総合的に検討してください\n")
        
        logger.info(f"📋 比較分析レポート保存: {report_path}")
        return report_path

def validate_args(args):
    """コマンドライン引数の検証"""
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"指定されたパスが存在しません: {input_path}")
    
    if args.min_races < 1:
        raise ValueError("最小レース数は1以上を指定してください")
    
    return args

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='馬ごと複勝時重み付けポイント分析を3年間隔で実行します',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python analyze_winning_track_level.py export/with_bias
  python analyze_winning_track_level.py export/with_bias --output-dir results/place_point_analysis
  python analyze_winning_track_level.py export/with_bias --compare-random  # ランダム重み付け比較
  
分析内容:
  - 馬ごとの複勝時重み付けポイント計算（複勝した場合のみポイント加算）
  - 3年間隔での時系列分析
  - 複勝経験馬のみでの相関分析
  - 実証済み格式度と勝率の関係分析
  - オプション: ランダム重み付けとの比較分析
        """
    )
    parser.add_argument('input_path', help='入力ファイルまたはディレクトリのパス')
    parser.add_argument('--output-dir', default='export/place_point_analysis', 
                       help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=5,  # デフォルトを5回に変更
                       help='分析対象とする最小レース数（デフォルト: 5）')
    parser.add_argument('--encoding', default='utf-8', 
                       help='入力ファイルのエンコーディング')
    parser.add_argument('--compare-random', action='store_true',
                       help='ランダム重み付けとの比較分析を実行')
    
    try:
        args = parser.parse_args()
        args = validate_args(args)
        
        # 設定の作成
        config = {
            'input_path': args.input_path,
            'output_dir': args.output_dir,
            'min_races': args.min_races,
            'encoding': args.encoding,
            'random_weights': False  # 通常は False
        }
        
        if args.compare_random:
            logger.info("🎲 ランダム重み付け比較分析モードで実行します...")
            logger.info(f"📁 入力パス: {args.input_path}")
            logger.info(f"📊 出力ディレクトリ: {args.output_dir}")
            logger.info(f"🎯 最小レース数: {args.min_races}")
            logger.info("🔄 オリジナル vs ランダム重み付けの比較を実行")
            
            # 比較分析の実行
            analyzer = TrackWinRateAnalyzer(config)
            
            logger.info("📖 データ読み込み中...")
            analyzer.df = analyzer.load_data()
            
            logger.info("🔄 比較分析実行中...")
            comparison_results = analyzer.compare_random_vs_original_weights()
            
            logger.info("📊 比較結果可視化中...")
            analyzer.visualize_comparison_results(comparison_results)
            
            logger.info("📝 比較分析レポート生成中...")
            analyzer.generate_comparison_report(comparison_results)
            
            # 比較結果のサマリー表示
            logger.info("\n" + "="*60)
            logger.info("🎉 比較分析完了！結果:")
            logger.info("="*60)
            
            for period_name, results in comparison_results.items():
                logger.info(f"📊 期間 {period_name}:")
                
                # オリジナル結果
                orig_corr = results['original']['correlation']
                if ('correlation_analysis' in orig_corr and 
                    'normalized' in orig_corr['correlation_analysis'] and
                    'place_rate' in orig_corr['correlation_analysis']['normalized']):
                    orig_r = orig_corr['correlation_analysis']['normalized']['place_rate']['correlation']
                    orig_p = orig_corr['correlation_analysis']['normalized']['place_rate']['p_value']
                    logger.info(f"   📈 オリジナル重み付け: r={orig_r:.3f}, p={orig_p:.3f}")
                
                # ランダム結果
                rand_corr = results['random']['correlation']
                if ('correlation_analysis' in rand_corr and 
                    'normalized' in rand_corr['correlation_analysis'] and
                    'place_rate' in rand_corr['correlation_analysis']['normalized']):
                    rand_r = rand_corr['correlation_analysis']['normalized']['place_rate']['correlation']
                    rand_p = rand_corr['correlation_analysis']['normalized']['place_rate']['p_value']
                    logger.info(f"   🎲 ランダム重み付け: r={rand_r:.3f}, p={rand_p:.3f}")
                    
                    # 差の計算と評価
                    diff = abs(orig_r) - abs(rand_r)
                    if diff > 0.1:
                        logger.info(f"   ✅ オリジナルが明らかに優秀 (差: {diff:.3f})")
                    elif diff > 0:
                        logger.info(f"   ➕ オリジナルがわずかに優秀 (差: {diff:.3f})")
                    elif diff > -0.1:
                        logger.info(f"   ➖ ほぼ同等 (差: {diff:.3f})")
                    else:
                        logger.info(f"   ❌ ランダムが優秀？ (差: {diff:.3f})")
                
                logger.info(f"   📊 対象: {results['total_horses']:,}頭, {results['total_races']:,}レース")
            
            logger.info("="*60)
            logger.info(f"✅ 比較結果は {args.output_dir}/random_vs_original_comparison に保存されました。")
            
        else:
            logger.info("🏇 馬ごと複勝時重み付けポイント分析を開始します...")
            logger.info(f"📁 入力パス: {args.input_path}")
            logger.info(f"📊 出力ディレクトリ: {args.output_dir}")
            logger.info(f"🎯 最小レース数: {args.min_races}")
            logger.info("✨ 特徴: 複勝した場合のみポイント加算")
            
            # 通常の分析の実行
            analyzer = TrackWinRateAnalyzer(config)
            
            logger.info("📖 データ読み込み中...")
            analyzer.df = analyzer.load_data()
            
            logger.info("🔧 データ前処理中...")
            analyzer.df = analyzer.preprocess_data()
            
            logger.info("📊 馬ごと複勝時重み付けポイント分析実行中...")
            results = analyzer.analyze_track_win_rates()
            
            logger.info("📊 可視化生成中...")
            analyzer.visualize_results(results)
            
            logger.info("📝 レポート生成中...")
            analyzer.generate_report(results)
            
            # 分析結果のサマリー表示
            logger.info("\n" + "="*60)
            logger.info("🎉 分析完了！主要な結果:")
            logger.info("="*60)
            
            for period_name, period_results in results.items():
                total_horses = period_results.get('total_horses', 0)
                total_races = period_results.get('total_races', 0)
                
                logger.info(f"📊 期間 {period_name}: {total_horses:,}頭, {total_races:,}レース")
                
                if ('race_point_correlation' in period_results and 
                    'correlation_analysis' in period_results['race_point_correlation']):
                    corr_analysis = period_results['race_point_correlation']['correlation_analysis']
                    place_exp_horses = period_results['race_point_correlation'].get('place_experienced_horses')
                    
                    if place_exp_horses is not None:
                        win_corr = corr_analysis['avg_point']['win_rate']['correlation']
                        win_p = corr_analysis['avg_point']['win_rate']['p_value']
                        win_r2 = corr_analysis['avg_point']['win_rate']['r2']
                        
                        logger.info(f"   📈 複勝時ポイント相関（複勝経験馬 {len(place_exp_horses)}頭）: r={win_corr:.3f}, p={win_p:.3f}")
                        logger.info(f"   📊 回帰R²: {win_r2:.3f}")
                    else:
                        logger.info(f"   ⚠️  複勝経験馬不足のため相関分析なし")
            
            logger.info("="*60)
            logger.info(f"✅ 全ての結果は {args.output_dir} に保存されました。")
            logger.info("📋 生成されたファイル:")
            logger.info("  - 馬ごと複勝時重み付けポイント分析レポート.md")
            logger.info("  - 各期間フォルダ内の分析結果PNG")
            logger.info("  ※ 複勝経験のない馬は重み付けポイント0として表示されます")
            logger.info("💡 ヒント: --compare-random オプションでランダム重み付けとの比較も可能です")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"❌ ファイルエラー: {str(e)}")
        return 1
    except ValueError as e:
        logger.error(f"❌ 入力値エラー: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"❌ 予期せぬエラーが発生しました: {str(e)}")
        logger.error("詳細なエラー情報:", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
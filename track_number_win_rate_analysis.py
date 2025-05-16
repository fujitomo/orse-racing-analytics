import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class TrackNumberWinRateAnalyzer:
    """
    競馬場別馬番と勝率の関係を分析するクラス
    """
    
    def __init__(self, data_folder="export/with_bias", output_folder="results/track_number_analysis"):
        """
        初期化
        
        Args:
            data_folder (str): データフォルダのパス
            output_folder (str): 結果出力先フォルダのパス
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.df = None
        
        # 日本語フォント設定
        self._setup_japanese_font()
        
    def _setup_japanese_font(self):
        """
        日本語フォントのセットアップ（改良版）
        """
        import matplotlib
        import matplotlib.font_manager as fm
        import platform
        import os
        
        try:
            print("日本語フォント設定を開始（改良版）...")
            
            if platform.system() == 'Windows':
                # Windowsフォントディレクトリ
                windows_fonts_dir = r'C:\Windows\Fonts'
                
                # 利用可能な日本語フォントファイルを直接指定
                font_candidates = [
                    (os.path.join(windows_fonts_dir, 'YuGothM.ttc'), 'Yu Gothic Medium'),
                    (os.path.join(windows_fonts_dir, 'YuGothB.ttc'), 'Yu Gothic Bold'),
                    (os.path.join(windows_fonts_dir, 'yugothm.ttf'), 'Yu Gothic Medium'),
                    (os.path.join(windows_fonts_dir, 'msgothic.ttc'), 'MS Gothic'),
                    (os.path.join(windows_fonts_dir, 'meiryo.ttc'), 'Meiryo'),
                    (os.path.join(windows_fonts_dir, 'msmincho.ttc'), 'MS Mincho'),
                ]
                
                font_found = False
                
                # フォントファイルの存在確認と設定
                for font_path, font_name in font_candidates:
                    if os.path.exists(font_path):
                        try:
                            # フォントプロパティを作成
                            prop = fm.FontProperties(fname=font_path)
                            
                            # フォントを登録
                            fm.fontManager.addfont(font_path)
                            
                            # matplotlibの設定を更新
                            matplotlib.rcParams['font.family'] = [font_name, 'DejaVu Sans', 'sans-serif']
                            
                            # テスト描画
                            fig, ax = plt.subplots(figsize=(2, 1))
                            ax.text(0.5, 0.5, '馬番勝率テスト', ha='center', va='center', fontproperties=prop)
                            plt.close(fig)
                            
                            print(f"フォント設定成功: {font_name} ({font_path})")
                            font_found = True
                            break
                            
                        except Exception as e:
                            print(f"フォント {font_name} の設定に失敗: {e}")
                            continue
                
                if not font_found:
                    print("Windowsフォントファイルからの設定に失敗しました。システムフォントを試行...")
                    # フォールバック: システム登録フォント
                    font_list = [f.name for f in fm.fontManager.ttflist]
                    japanese_fonts = ['Yu Gothic UI', 'Yu Gothic', 'MS Gothic', 'MS PGothic', 'Meiryo UI', 'Meiryo']
                    
                    for font in japanese_fonts:
                        if font in font_list:
                            matplotlib.rcParams['font.family'] = [font, 'DejaVu Sans']
                            print(f"システムフォント設定: {font}")
                            font_found = True
                            break
                    
                    if not font_found:
                        print("日本語フォントが見つかりません。英語表示になります。")
                        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
            else:
                # Windows以外のOS
                matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
                print("非Windows環境での日本語フォント設定")
            
            # 共通設定
            matplotlib.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['figure.figsize'] = (12, 8)
            matplotlib.rcParams['font.size'] = 10
            matplotlib.rcParams['axes.titlesize'] = 12
            matplotlib.rcParams['axes.labelsize'] = 10
            matplotlib.rcParams['legend.fontsize'] = 10
            
            # 設定確認のためのテスト
            current_font = matplotlib.rcParams['font.family']
            print(f"最終フォント設定: {current_font}")
            
        except Exception as e:
            print(f"フォント設定エラー: {e}")
            print("デフォルト設定を使用します。")
            # 最低限の設定
            try:
                matplotlib.rcParams['axes.unicode_minus'] = False
                matplotlib.rcParams['font.family'] = ['DejaVu Sans']
            except:
                pass
        
    def load_sed_data(self):
        """
        SEDファイル群からデータを読み込む
        
        Returns:
            bool: 読み込み成功時True、失敗時False
        """
        print("SEDデータを読み込んでいます...")
        
        # SEDファイルのパターンを検索
        sed_files = glob.glob(os.path.join(self.data_folder, "SED*_formatted_with_bias.csv"))
        
        if not sed_files:
            print(f"エラー: {self.data_folder} にSEDファイルが見つかりません。")
            return False
        
        print(f"見つかったSEDファイル: {len(sed_files)}個")
        
        data_list = []
        
        for file_path in sed_files:
            try:
                # エンコーディングを試行錯誤で読み込み
                for encoding in ['utf-8', 'shift-jis', 'cp932']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"読み込み成功: {os.path.basename(file_path)} ({len(df)}行)")
                        data_list.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"ファイル読み込みエラー: {file_path} - {e}")
                        break
            except Exception as e:
                print(f"ファイル {file_path} の処理中にエラーが発生: {e}")
        
        if not data_list:
            print("データの読み込みに失敗しました。")
            return False
        
        # データを結合
        self.df = pd.concat(data_list, ignore_index=True)
        print(f"総データ数: {len(self.df)}行")
        
        return True
    
    def preprocess_data(self):
        """
        データの前処理を実行
        
        Returns:
            bool: 前処理成功時True、失敗時False
        """
        print("データの前処理を実行中...")
        
        if self.df is None:
            print("エラー: データが読み込まれていません。")
            return False
        
        # 必要なカラムの存在確認
        required_columns = ['場コード', '年', '馬番', '着順']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"エラー: 必要なカラムが不足しています: {missing_columns}")
            return False
        
        # データ型の変換
        self.df['年'] = pd.to_numeric(self.df['年'], errors='coerce')
        self.df['馬番'] = pd.to_numeric(self.df['馬番'], errors='coerce')
        self.df['着順'] = pd.to_numeric(self.df['着順'], errors='coerce')
        
        # 勝利フラグの作成（1位なら1、それ以外は0）
        self.df['勝利'] = (self.df['着順'] == 1).astype(int)
        
        # 無効なデータを除去
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['年', '馬番', '着順'])
        after_count = len(self.df)
        
        print(f"データクリーニング: {before_count}行 → {after_count}行")
        
        # 基本統計情報の表示
        print("\n=== データ概要 ===")
        print(f"年の範囲: {self.df['年'].min()} - {self.df['年'].max()}")
        print(f"馬番の範囲: {self.df['馬番'].min()} - {self.df['馬番'].max()}")
        print(f"競馬場数: {self.df['場名'].nunique()}")
        print(f"競馬場: {list(self.df['場名'].unique())}")
        
        return True
    
    def get_period_ranges(self, period_years=3):
        """
        分析期間の範囲を取得
        
        Args:
            period_years (int): 1期間の年数
            
        Returns:
            list: 期間の範囲のリスト [(開始年, 終了年), ...]
        """
        min_year = self.df['年'].min()
        max_year = self.df['年'].max()
        
        if pd.isna(min_year) or pd.isna(max_year):
            print("エラー: 年データが不正です。")
            return []
        
        periods = []
        start_year = min_year
        
        while start_year <= max_year:
            end_year = min(start_year + period_years - 1, max_year)
            periods.append((int(start_year), int(end_year)))
            start_year = end_year + 1
        
        print(f"分析期間: {periods}")
        return periods
    
    def calculate_win_rate_by_number(self, data):
        """
        馬番別の勝率を計算
        
        Args:
            data (DataFrame): 分析対象データ
            
        Returns:
            DataFrame: 馬番別勝率データ
        """
        # 馬番別の勝率集計
        number_stats = data.groupby('馬番').agg({
            '勝利': ['count', 'sum', 'mean'],
            '着順': 'mean'
        }).round(4)
        
        # カラム名を整理
        number_stats.columns = ['総レース数', '勝利数', '勝率', '平均着順']
        number_stats = number_stats.reset_index()
        
        # 勝率をパーセンテージに変換
        number_stats['勝率_percent'] = number_stats['勝率'] * 100
        
        return number_stats
    
    def perform_statistical_analysis(self, data, track_name, period_name):
        """
        統計分析を実行
        
        Args:
            data (DataFrame): 分析対象データ
            track_name (str): 競馬場名
            period_name (str): 期間名
            
        Returns:
            dict: 分析結果
        """
        # 馬番別勝率を計算
        win_rate_data = self.calculate_win_rate_by_number(data)
        
        # 十分なデータがない場合はスキップ
        if len(win_rate_data) < 3:
            return None
        
        # 相関分析
        correlation_coef = win_rate_data['馬番'].corr(win_rate_data['勝率'])
        
        # ピアソン相関のp値を計算
        correlation_pvalue = stats.pearsonr(win_rate_data['馬番'], win_rate_data['勝率'])[1]
        
        # 線形回帰
        X = win_rate_data['馬番'].values.reshape(-1, 1)
        y = win_rate_data['勝率'].values
        
        # scikit-learnでの線形回帰
        from sklearn.linear_model import LinearRegression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        
        # R²値
        r2 = linear_model.score(X, y)
        
        # ロジスティック回帰（個別レース結果での分析）
        logistic_X = data['馬番'].values.reshape(-1, 1)
        logistic_y = data['勝利'].values
        
        logistic_model = LogisticRegression(max_iter=1000)
        try:
            logistic_model.fit(logistic_X, logistic_y)
            logistic_coef = logistic_model.coef_[0][0]
            logistic_intercept = logistic_model.intercept_[0]
        except:
            logistic_coef = np.nan
            logistic_intercept = np.nan
        
        # 結果をまとめる
        results = {
            'track_name': track_name,
            'period_name': period_name,
            'sample_size': len(data),
            'horse_numbers': len(win_rate_data),
            'correlation_coefficient': correlation_coef,
            'correlation_pvalue': correlation_pvalue,
            'linear_r2': r2,
            'linear_slope': linear_model.coef_[0],
            'linear_intercept': linear_model.intercept_,
            'logistic_coefficient': logistic_coef,
            'logistic_intercept': logistic_intercept,
            'win_rate_data': win_rate_data
        }
        
        return results
    
    def create_visualizations(self, results, output_dir):
        """
        可視化グラフを作成（確実なフォント設定版）
        
        Args:
            results (dict): 分析結果
            output_dir (str): 出力ディレクトリ
        """
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import platform
        import os
        
        # 日本語フォントプロパティを準備
        font_prop = None
        try:
            if platform.system() == 'Windows':
                # Windowsフォントファイルを直接指定
                windows_fonts_dir = r'C:\Windows\Fonts'
                font_candidates = [
                    os.path.join(windows_fonts_dir, 'YuGothM.ttc'),
                    os.path.join(windows_fonts_dir, 'msgothic.ttc'),
                    os.path.join(windows_fonts_dir, 'meiryo.ttc'),
                ]
                
                for font_path in font_candidates:
                    if os.path.exists(font_path):
                        font_prop = fm.FontProperties(fname=font_path)
                        print(f"可視化用フォント設定: {font_path}")
                        break
            
            if font_prop is None:
                # フォールバック
                font_prop = fm.FontProperties(family=['Yu Gothic', 'MS Gothic', 'Meiryo', 'DejaVu Sans'])
                
        except Exception as e:
            print(f"フォントプロパティ設定エラー: {e}")
            font_prop = fm.FontProperties()
        
        track_name = results['track_name']
        period_name = results['period_name']
        win_rate_data = results['win_rate_data']
        
        # 図のサイズを設定
        plt.style.use('default')
        
        # 1. 散布図と回帰直線
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # タイトル設定（フォントプロパティ指定）
        title_text = f'{track_name} - {period_name}期間 馬番と勝率の関係分析'
        fig.suptitle(title_text, fontsize=16, fontproperties=font_prop)
        
        # 散布図
        ax1 = axes[0, 0]
        ax1.scatter(win_rate_data['馬番'], win_rate_data['勝率_percent'], alpha=0.7, s=50)
        
        # 回帰直線
        x_range = np.linspace(win_rate_data['馬番'].min(), win_rate_data['馬番'].max(), 100)
        y_pred = results['linear_slope'] * x_range + results['linear_intercept']
        ax1.plot(x_range, y_pred * 100, 'r-', linewidth=2, label='回帰直線')
        
        ax1.set_xlabel('馬番', fontproperties=font_prop)
        ax1.set_ylabel('勝率 (%)', fontproperties=font_prop)
        ax1.set_title('散布図と回帰直線', fontproperties=font_prop)
        ax1.legend(prop=font_prop)
        ax1.grid(True, alpha=0.3)
        
        # 馬番別勝率棒グラフ
        ax2 = axes[0, 1]
        bars = ax2.bar(win_rate_data['馬番'], win_rate_data['勝率_percent'], alpha=0.7)
        ax2.set_xlabel('馬番', fontproperties=font_prop)
        ax2.set_ylabel('勝率 (%)', fontproperties=font_prop)
        ax2.set_title('馬番別勝率', fontproperties=font_prop)
        ax2.grid(True, alpha=0.3)
        
        # 統計情報テキスト
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # 統計情報（確実なフォント指定）
        stats_text = f"""統計分析結果:

相関係数: {results['correlation_coefficient']:.4f}
p値: {results['correlation_pvalue']:.4f}
決定係数(R²): {results['linear_r2']:.4f}

線形回帰:
  傾き: {results['linear_slope']:.6f}
  切片: {results['linear_intercept']:.6f}

ロジスティック回帰:
  係数: {results['logistic_coefficient']:.6f}
  切片: {results['logistic_intercept']:.6f}

サンプル数:
  総レース数: {results['sample_size']:,}
  馬番数: {results['horse_numbers']}
        """
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontproperties=font_prop)
        
        # 馬番別レース数
        ax4 = axes[1, 1]
        ax4.bar(win_rate_data['馬番'], win_rate_data['総レース数'], alpha=0.7, color='orange')
        ax4.set_xlabel('馬番', fontproperties=font_prop)
        ax4.set_ylabel('レース数', fontproperties=font_prop)
        ax4.set_title('馬番別レース数', fontproperties=font_prop)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル保存
        filename = f"{track_name}_{period_name}_馬番勝率分析_確実版.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"確実版グラフを保存しました: {filepath}")
    
    def analyze_track_number_winrate(self, period_years=3, min_races=100):
        """
        競馬場別馬番と勝率の関係を分析
        
        Args:
            period_years (int): 期間年数
            min_races (int): 分析に必要な最小レース数
            
        Returns:
            list: 分析結果のリスト
        """
        print(f"競馬場別馬番と勝率の関係分析を開始します（期間: {period_years}年）")
        
        # 出力ディレクトリの作成
        output_dir = os.path.join(self.output_folder, f"period_{period_years}years")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 期間範囲を取得
        periods = self.get_period_ranges(period_years)
        if not periods:
            return []
        
        # 分析結果を保存するリスト
        all_results = []
        
        # 競馬場別に分析
        tracks = self.df['場名'].unique()
        
        for track in tracks:
            print(f"\n=== {track}競馬場の分析 ===")
            track_data = self.df[self.df['場名'] == track]
            
            for period_start, period_end in periods:
                period_name = f"{period_start}-{period_end}"
                period_data = track_data[
                    (track_data['年'] >= period_start) & 
                    (track_data['年'] <= period_end)
                ]
                
                # データが十分にない場合はスキップ
                if len(period_data) < min_races:
                    print(f"  {period_name}: データ不足のためスキップ ({len(period_data)}レース < {min_races})")
                    continue
                
                print(f"  {period_name}: {len(period_data)}レースを分析中...")
                
                # 統計分析を実行
                results = self.perform_statistical_analysis(period_data, track, period_name)
                
                if results is None:
                    print(f"    分析失敗: 十分なデータがありません")
                    continue
                
                # 可視化
                self.create_visualizations(results, output_dir)
                
                # 結果を保存
                all_results.append(results)
                
                # 結果をコンソールに表示
                print(f"    相関係数: {results['correlation_coefficient']:.4f}")
                print(f"    p値: {results['correlation_pvalue']:.4f}")
                print(f"    決定係数: {results['linear_r2']:.4f}")
        
        return all_results
    
    def generate_summary_report(self, all_results, period_years):
        """
        総合レポートを生成
        
        Args:
            all_results (list): 全分析結果
            period_years (int): 期間年数
        """
        output_dir = os.path.join(self.output_folder, f"period_{period_years}years")
        report_path = os.path.join(output_dir, f"馬番勝率分析レポート_{period_years}年.md")
        
        # Windows環境での文字化けを防ぐためUTF-8 BOM付きで保存
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"# 競馬場別馬番と勝率の関係分析レポート（{period_years}年期間）\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 概要\n\n")
            f.write(f"各競馬場において、馬番と勝率の関係を{period_years}年期間ごとに分析しました。\n")
            f.write("統計的手法として相関分析、線形回帰、ロジスティック回帰を使用しています。\n\n")
            
            f.write("## 分析結果一覧\n\n")
            f.write("| 競馬場 | 期間 | 相関係数 | p値 | 決定係数(R²) | 線形回帰傾き | ロジスティック回帰係数 | レース数 |\n")
            f.write("|--------|------|----------|-----|-------------|-------------|---------------------|----------|\n")
            
            for result in all_results:
                f.write(f"| {result['track_name']} | {result['period_name']} | "
                       f"{result['correlation_coefficient']:.4f} | "
                       f"{result['correlation_pvalue']:.4f} | "
                       f"{result['linear_r2']:.4f} | "
                       f"{result['linear_slope']:.6f} | "
                       f"{result['logistic_coefficient']:.6f} | "
                       f"{result['sample_size']:,} |\n")
            
            f.write("\n## 統計的解釈\n\n")
            f.write("### 相関係数について\n")
            f.write("- -1 ≤ r ≤ 1 の範囲で、0に近いほど無相関\n")
            f.write("- |r| > 0.3 で弱い相関、|r| > 0.5 で中程度の相関\n\n")
            
            f.write("### p値について\n")
            f.write("- p < 0.05 で統計的に有意（95%信頼水準）\n")
            f.write("- p < 0.01 で高度に有意（99%信頼水準）\n\n")
            
            f.write("### 決定係数(R²)について\n")
            f.write("- 0 ≤ R² ≤ 1 の範囲で、1に近いほど回帰モデルの説明力が高い\n")
            f.write("- R² > 0.1 で実用的な説明力があると考えられる\n\n")
            
            # 有意な結果の要約
            significant_results = [r for r in all_results if r['correlation_pvalue'] < 0.05]
            
            f.write(f"### 統計的に有意な結果（p < 0.05）\n\n")
            if significant_results:
                f.write(f"全{len(all_results)}件中{len(significant_results)}件で統計的に有意な関係が検出されました。\n\n")
                
                for result in significant_results:
                    correlation_strength = "強い" if abs(result['correlation_coefficient']) > 0.5 else \
                                         "中程度" if abs(result['correlation_coefficient']) > 0.3 else "弱い"
                    direction = "正の" if result['correlation_coefficient'] > 0 else "負の"
                    
                    f.write(f"- **{result['track_name']}（{result['period_name']}）**: "
                           f"{direction}{correlation_strength}相関 (r={result['correlation_coefficient']:.4f}, "
                           f"p={result['correlation_pvalue']:.4f})\n")
                f.write("\n")
            else:
                f.write("統計的に有意な関係は検出されませんでした。\n\n")
        
        print(f"レポートを保存しました: {report_path}")
        
        # PowerShell用のShift-JIS版も作成
        report_path_sjis = os.path.join(output_dir, f"馬番勝率分析レポート_{period_years}年_SJIS.md")
        try:
            with open(report_path_sjis, 'w', encoding='shift-jis', errors='replace') as f:
                f.write(f"# 競馬場別馬番と勝率の関係分析レポート（{period_years}年期間）\n\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 概要\n\n")
                f.write(f"各競馬場において、馬番と勝率の関係を{period_years}年期間ごとに分析しました。\n")
                f.write("統計的手法として相関分析、線形回帰、ロジスティック回帰を使用しています。\n\n")
                
                f.write("## 分析結果一覧\n\n")
                f.write("| 競馬場 | 期間 | 相関係数 | p値 | 決定係数(R²) | 線形回帰傾き | ロジスティック回帰係数 | レース数 |\n")
                f.write("|--------|------|----------|-----|-------------|-------------|---------------------|----------|\n")
                
                for result in all_results:
                    f.write(f"| {result['track_name']} | {result['period_name']} | "
                           f"{result['correlation_coefficient']:.4f} | "
                           f"{result['correlation_pvalue']:.4f} | "
                           f"{result['linear_r2']:.4f} | "
                           f"{result['linear_slope']:.6f} | "
                           f"{result['logistic_coefficient']:.6f} | "
                           f"{result['sample_size']:,} |\n")
                
                f.write("\n## 統計的解釈\n\n")
                f.write("### 相関係数について\n")
                f.write("- -1 ≤ r ≤ 1 の範囲で、0に近いほど無相関\n")
                f.write("- |r| > 0.3 で弱い相関、|r| > 0.5 で中程度の相関\n\n")
                
                f.write("### p値について\n")
                f.write("- p < 0.05 で統計的に有意（95%信頼水準）\n")
                f.write("- p < 0.01 で高度に有意（99%信頼水準）\n\n")
                
                f.write("### 決定係数(R²)について\n")
                f.write("- 0 ≤ R² ≤ 1 の範囲で、1に近いほど回帰モデルの説明力が高い\n")
                f.write("- R² > 0.1 で実用的な説明力があると考えられる\n\n")
                
                f.write(f"### 統計的に有意な結果（p < 0.05）\n\n")
                if significant_results:
                    f.write(f"全{len(all_results)}件中{len(significant_results)}件で統計的に有意な関係が検出されました。\n\n")
                    
                    for result in significant_results:
                        correlation_strength = "強い" if abs(result['correlation_coefficient']) > 0.5 else \
                                             "中程度" if abs(result['correlation_coefficient']) > 0.3 else "弱い"
                        direction = "正の" if result['correlation_coefficient'] > 0 else "負の"
                        
                        f.write(f"- **{result['track_name']}（{result['period_name']}）**: "
                               f"{direction}{correlation_strength}相関 (r={result['correlation_coefficient']:.4f}, "
                               f"p={result['correlation_pvalue']:.4f})\n")
                    f.write("\n")
                else:
                    f.write("統計的に有意な関係は検出されませんでした。\n\n")
            
            print(f"PowerShell用レポートを保存しました: {report_path_sjis}")
        except Exception as e:
            print(f"Shift-JIS版の作成に失敗しました: {e}")

def main():
    """
    メイン実行関数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='競馬場別馬番と勝率の関係分析')
    parser.add_argument('--period', type=int, default=3, 
                       help='分析期間の年数（デフォルト: 3年）')
    parser.add_argument('--min-races', type=int, default=100,
                       help='分析に必要な最小レース数（デフォルト: 100）')
    parser.add_argument('--data-folder', type=str, default="export/with_bias",
                       help='データフォルダのパス')
    parser.add_argument('--output-folder', type=str, default="results/track_number_analysis",
                       help='結果出力先フォルダのパス')
    
    args = parser.parse_args()
    
    # 分析器を初期化
    analyzer = TrackNumberWinRateAnalyzer(
        data_folder=args.data_folder,
        output_folder=args.output_folder
    )
    
    # データ読み込み
    if not analyzer.load_sed_data():
        print("データ読み込みに失敗しました。")
        return
    
    # データ前処理
    if not analyzer.preprocess_data():
        print("データ前処理に失敗しました。")
        return
    
    # 分析実行
    all_results = analyzer.analyze_track_number_winrate(
        period_years=args.period,
        min_races=args.min_races
    )
    
    if not all_results:
        print("分析結果がありません。")
        return
    
    # レポート生成
    analyzer.generate_summary_report(all_results, args.period)
    
    print(f"\n=== 分析完了 ===")
    print(f"分析件数: {len(all_results)}件")
    print(f"結果保存先: {args.output_folder}")

if __name__ == "__main__":
    main() 
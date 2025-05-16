"""
競馬場バイアス×期間×能力指標分析システム

このプログラムは以下の分析を実行します：

1. 期間別分析（デフォルト5年ごと）
   - 各競馬場のバイアス傾向の時系列変化
   - 期間ごとの勝率・複勝率の推移
   - 馬の能力指標（IDM、素点）との関係性

2. コース種別（芝・ダート）別分析
   - コース種別ごとのバイアス影響度
   - 能力指標との相関関係
   - 勝率・複勝率への影響

3. コーナー別分析
   - 各コーナー（1-4角）のバイアス傾向
   - 内・中・外の位置による影響
   - 能力指標との組み合わせ分析

4. 出力内容
   - 期間別分析レポート（勝率・複勝率）
   - コーナー別バイアス分析グラフ
   - 能力指標との関係性可視化
   - 統計的検定結果

使用方法：
python track_bias_by_period_ability.py [--period 年数]
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib

def load_data(formatted_folder, bias_folder):
    """
    データの読み込みと結合
    
    Args:
        formatted_folder (str): 通常フォーマットのデータフォルダ
        bias_folder (str): バイアス情報のデータフォルダ
    
    Returns:
        DataFrame: 結合したDataFrame
    """
    print("データを読み込んでいます...")
    
    # 通常フォーマットのデータ読み込み
    formatted_files = glob.glob(os.path.join(formatted_folder, "*.csv"))
    formatted_data = []
    
    for file_path in formatted_files:
        try:
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    
                    # レースIDを作成
                    if '場コード' in df.columns and '年' in df.columns and '回' in df.columns and '日' in df.columns and 'R' in df.columns:
                        df['レースID'] = df['場コード'].astype(str) + df['年'].astype(str) + df['回'].astype(str) + df['日'].astype(str) + df['R'].astype(str)
                    
                    formatted_data.append(df)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"ファイル読み込みエラー: {file_path} - {e}")
                    break
        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生: {e}")
    
    # バイアス情報のデータ読み込み
    bias_files = glob.glob(os.path.join(bias_folder, "*_formatted_with_bias.csv"))
    bias_data = []
    
    for file_path in bias_files:
        try:
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    
                    # レースIDを作成
                    if '場コード' in df.columns and '年' in df.columns and '回' in df.columns and '日' in df.columns and 'R' in df.columns:
                        df['レースID'] = df['場コード'].astype(str) + df['年'].astype(str) + df['回'].astype(str) + df['日'].astype(str) + df['R'].astype(str)
                    
                    bias_data.append(df)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"ファイル読み込みエラー: {file_path} - {e}")
                    break
        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生: {e}")
    
    # データが読み込めなかった場合
    if not formatted_data or not bias_data:
        print("データの読み込みに失敗しました。")
        return None
    
    # データ結合
    formatted_df = pd.concat(formatted_data, ignore_index=True)
    bias_df = pd.concat(bias_data, ignore_index=True)
    
    print(f"通常データ: {len(formatted_df)} 行")
    print(f"バイアスデータ: {len(bias_df)} 行")
    
    # バイアス情報のカラムを特定
    bias_cols = [col for col in bias_df.columns if 'バイアス' in col]
    bias_cols.extend(['レースID', '馬番'])
    
    # データを結合
    merged_df = pd.merge(
        formatted_df,
        bias_df[bias_cols],
        on=['レースID', '馬番'],
        how='left'
    )
    
    print(f"結合後のデータ: {len(merged_df)} 行")
    
    # データ前処理
    merged_df = preprocess_data(merged_df)
    
    return merged_df

def preprocess_data(df):
    """
    データの前処理
    
    Args:
        df (DataFrame): 前処理するデータフレーム
    
    Returns:
        DataFrame: 前処理後のデータフレーム
    """
    print("データの前処理を行っています...")
    
    # 年を確実に数値型に変換
    if '年' in df.columns:
        df['年'] = pd.to_numeric(df['年'], errors='coerce')
        print(f"年の範囲: {df['年'].min()} - {df['年'].max()}")
    else:
        print("警告: '年'カラムが見つかりません。期間分析ができません。")
    
    # 着順を数値に変換
    if '着順' in df.columns:
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    
    # 勝利フラグを追加（1位なら1、それ以外は0）
    df['勝利'] = (df['着順'] == 1).astype(int)
    
    # 複勝フラグを追加（3位以内なら1、それ以外は0）
    df['複勝'] = (df['着順'] <= 3).astype(int)
    
    # 能力値（IDM、素点）を数値に変換
    for col in ['IDM', '素点']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"{col}の範囲: {df[col].min()} - {df[col].max()}")
    
    # コーナーバイアスの種類をカウント
    for corner in ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']:
        if corner in df.columns:
            df[f'{corner}_内側'] = df[corner].str.contains(r'\([*]?1,|\(1[,)]|\([*]?2,|\(2[,)]|\([*]?3,|\(3[,)]', regex=True).fillna(False).astype(int)
            df[f'{corner}_中間'] = df[corner].str.contains(r'\([*]?[4-9],|\([4-9][,)]|\([*]?1[0-2],|\(1[0-2][,)]', regex=True).fillna(False).astype(int)
            df[f'{corner}_外側'] = df[corner].str.contains(r'\([*]?1[3-9],|\(1[3-9][,)]|\([*]?2[0-9],|\(2[0-9][,)]', regex=True).fillna(False).astype(int)
    
    # 芝・ダート情報の変換
    if '芝ダ障害コード' in df.columns:
        if pd.api.types.is_numeric_dtype(df['芝ダ障害コード']):
            df['コース種別'] = df['芝ダ障害コード'].map({1: '芝', 2: 'ダート', 3: '障害'})
        else:
            df['コース種別'] = df['芝ダ障害コード']
    
    # 脚質の変換
    if 'レース脚質' in df.columns:
        if pd.api.types.is_numeric_dtype(df['レース脚質']):
            df['脚質'] = df['レース脚質'].map({1: '逃げ', 2: '先行', 3: '差し', 4: '追込'})
        else:
            df['脚質'] = df['レース脚質']
    
    return df

def get_period_ranges(df, period_years=5):
    """
    分析期間の範囲を取得する
    
    Args:
        df (DataFrame): 分析データ
        period_years (int): 1期間の年数（デフォルト: 5年）
    
    Returns:
        list: 期間の範囲のリスト [(開始年, 終了年), ...]
    """
    if '年' not in df.columns:
        print("年カラムがないため、期間分析ができません。")
        return []
    
    min_year = df['年'].min()
    max_year = df['年'].max()
    
    if pd.isna(min_year) or pd.isna(max_year):
        print("年データが不正です。")
        return []
    
    # 期間の範囲を作成
    periods = []
    start_year = min_year
    
    while start_year <= max_year:
        end_year = min(start_year + period_years - 1, max_year)
        periods.append((int(start_year), int(end_year)))
        start_year = end_year + 1
    
    print(f"分析期間: {periods}")
    return periods

def analyze_location_period_ability(df, output_dir, period_years=5, ability_columns=['IDM', '素点']):
    """
    場所×期間×能力指標の分析
    
    Args:
        df (DataFrame): 分析対象のデータフレーム
        output_dir (str): 結果出力先ディレクトリ
        period_years (int): 1期間の年数（デフォルト: 5年）
        ability_columns (list): 分析する能力指標のカラム名
    
    Returns:
        dict: 分析結果
    """
    print(f"場所×期間({period_years}年)×能力指標の分析を実行中...")
    
    # 年カラムがあるか確認
    if '年' not in df.columns:
        print("年カラムがないため、期間分析ができません。")
        return None
    
    # 分析する能力指標が存在するか確認
    available_ability_cols = [col for col in ability_columns if col in df.columns]
    if not available_ability_cols:
        print(f"分析対象の能力指標({', '.join(ability_columns)})が見つかりません。")
        return None
    
    # 出力フォルダ作成
    analysis_dir = os.path.join(output_dir, f"location_period{period_years}_ability_analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # 複勝分析用の出力フォルダ
    show_analysis_dir = os.path.join(output_dir, f"location_period{period_years}_ability_show_analysis")
    if not os.path.exists(show_analysis_dir):
        os.makedirs(show_analysis_dir)
    
    # 期間範囲を取得
    periods = get_period_ranges(df, period_years)
    if not periods:
        return None
    
    # 結果保存用の辞書
    results = {}
    show_results = {}  # 複勝結果用
    
    # 競馬場リスト
    tracks = df['場名'].unique()
    
    # コーナーリスト
    corners = ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス']
    
    # 各競馬場、各期間、各能力指標、各コーナーバイアスの組み合わせを分析
    for track in tracks:
        track_data = df[df['場名'] == track]
        track_results = {}
        track_show_results = {}  # 複勝結果用
        
        for period_start, period_end in periods:
            period_key = f"{period_start}-{period_end}"
            period_data = track_data[(track_data['年'] >= period_start) & (track_data['年'] <= period_end)]
            
            # データが少なすぎる場合はスキップ
            if len(period_data) < 100:
                print(f"{track}競馬場の{period_key}期間はデータが少ないのでスキップします（{len(period_data)}件）")
                continue
            
            print(f"{track}競馬場の{period_key}期間のデータ: {len(period_data)}件")
            
            # 各コース種別ごとに分析
            surface_results = {}
            surface_show_results = {}  # 複勝結果用
            
            for surface in ['芝', 'ダート']:
                surface_data = period_data[period_data['コース種別'] == surface]
                
                if len(surface_data) < 50:
                    print(f"{track}競馬場（{surface}）の{period_key}期間はデータが少ないのでスキップします（{len(surface_data)}件）")
                    continue
                
                # 各能力指標ごとに分析
                ability_results = {}
                ability_show_results = {}  # 複勝結果用
                
                for ability_col in available_ability_cols:
                    # 欠損値を除外
                    valid_data = surface_data.dropna(subset=[ability_col, '勝利', '複勝'])
                    
                    if len(valid_data) < 50:
                        continue
                    
                    # 能力値を5分位に分割
                    valid_data[f'{ability_col}_分位'] = pd.qcut(valid_data[ability_col], 5, labels=False)
                    
                    # 各コーナーごとに分析（勝率）
                    corner_results = {}
                    for corner in corners:
                        inner_col = f'{corner}_内側'
                        middle_col = f'{corner}_中間'
                        outer_col = f'{corner}_外側'
                        
                        if inner_col not in valid_data.columns or middle_col not in valid_data.columns or outer_col not in valid_data.columns:
                            continue
                        
                        # 各分位ごとの、内側・中間・外側の勝率を計算
                        position_results = {
                            '内側': [],
                            '中間': [],
                            '外側': []
                        }
                        
                        for position_idx, position_col in enumerate([inner_col, middle_col, outer_col]):
                            position_key = ['内側', '中間', '外側'][position_idx]
                            position_data = valid_data[valid_data[position_col] == 1]
                            
                            if len(position_data) < 20:
                                continue
                            
                            # 分位ごとの勝率を計算
                            win_rates = []
                            for quintile in range(5):
                                quintile_data = position_data[position_data[f'{ability_col}_分位'] == quintile]
                                if len(quintile_data) > 0:
                                    win_rate = quintile_data['勝利'].mean() * 100
                                    sample_size = len(quintile_data)
                                else:
                                    win_rate = 0
                                    sample_size = 0
                                
                                win_rates.append({
                                    '分位': quintile,
                                    '勝率': win_rate,
                                    'サンプル数': sample_size
                                })
                            
                            position_results[position_key] = win_rates
                        
                        # 各コーナーの結果を保存
                        corner_results[corner] = position_results
                    
                    # 各コーナーごとに分析（複勝率）
                    corner_show_results = {}
                    for corner in corners:
                        inner_col = f'{corner}_内側'
                        middle_col = f'{corner}_中間'
                        outer_col = f'{corner}_外側'
                        
                        if inner_col not in valid_data.columns or middle_col not in valid_data.columns or outer_col not in valid_data.columns:
                            continue
                        
                        # 各分位ごとの、内側・中間・外側の複勝率を計算
                        position_show_results = {
                            '内側': [],
                            '中間': [],
                            '外側': []
                        }
                        
                        for position_idx, position_col in enumerate([inner_col, middle_col, outer_col]):
                            position_key = ['内側', '中間', '外側'][position_idx]
                            position_data = valid_data[valid_data[position_col] == 1]
                            
                            if len(position_data) < 20:
                                continue
                            
                            # 分位ごとの複勝率を計算
                            show_rates = []
                            for quintile in range(5):
                                quintile_data = position_data[position_data[f'{ability_col}_分位'] == quintile]
                                if len(quintile_data) > 0:
                                    show_rate = quintile_data['複勝'].mean() * 100
                                    sample_size = len(quintile_data)
                                else:
                                    show_rate = 0
                                    sample_size = 0
                                
                                show_rates.append({
                                    '分位': quintile,
                                    '複勝率': show_rate,
                                    'サンプル数': sample_size
                                })
                            
                            position_show_results[position_key] = show_rates
                        
                        # 各コーナーの複勝結果を保存
                        corner_show_results[corner] = position_show_results
                    
                    # グラフ作成（勝率）
                    for corner, positions in corner_results.items():
                        plt.figure(figsize=(12, 8))
                        
                        for position, win_rates in positions.items():
                            if not win_rates:
                                continue
                            
                            x_vals = [item['分位'] for item in win_rates]
                            y_vals = [item['勝率'] for item in win_rates]
                            
                            plt.plot(x_vals, y_vals, marker='o', label=position)
                        
                        plt.title(f'{track}（{surface}）{period_key}期間の{corner.replace("バイアス", "")}と{ability_col}の関係')
                        plt.xlabel(f'{ability_col}分位')
                        plt.ylabel('勝率 (%)')
                        plt.xticks(range(5))
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        
                        # 保存
                        plt.savefig(os.path.join(analysis_dir, f'{track}_{surface}_{period_key}_{corner}_{ability_col}_関係.png'), dpi=300)
                        plt.close()
                    
                    # グラフ作成（複勝率）
                    for corner, positions in corner_show_results.items():
                        plt.figure(figsize=(12, 8))
                        
                        for position, show_rates in positions.items():
                            if not show_rates:
                                continue
                            
                            x_vals = [item['分位'] for item in show_rates]
                            y_vals = [item['複勝率'] for item in show_rates]
                            
                            plt.plot(x_vals, y_vals, marker='o', label=position)
                        
                        plt.title(f'{track}（{surface}）{period_key}期間の{corner.replace("バイアス", "")}と{ability_col}の関係（複勝）')
                        plt.xlabel(f'{ability_col}分位')
                        plt.ylabel('複勝率 (%)')
                        plt.xticks(range(5))
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        
                        # 保存
                        plt.savefig(os.path.join(show_analysis_dir, f'{track}_{surface}_{period_key}_{corner}_{ability_col}_複勝関係.png'), dpi=300)
                        plt.close()
                    
                    # 能力指標ごとの結果を保存
                    ability_results[ability_col] = corner_results
                    ability_show_results[ability_col] = corner_show_results
                
                # コース種別ごとの結果を保存
                if ability_results:
                    surface_results[surface] = ability_results
                if ability_show_results:
                    surface_show_results[surface] = ability_show_results
            
            # 期間ごとの結果を保存
            if surface_results:
                track_results[period_key] = surface_results
            if surface_show_results:
                track_show_results[period_key] = surface_show_results
        
        # 競馬場ごとの結果を保存
        if track_results:
            results[track] = track_results
        if track_show_results:
            show_results[track] = track_show_results
    
    # レポート作成
    generate_report(results, analysis_dir, period_years, '勝率')
    generate_report(show_results, show_analysis_dir, period_years, '複勝率')
    
    return results, show_results

def generate_report(results, output_dir, period_years, rate_type='勝率'):
    """
    分析結果のレポートを生成
    
    Args:
        results (dict): 分析結果
        output_dir (str): 出力先ディレクトリ
        period_years (int): 期間年数
        rate_type (str): 勝率または複勝率
    """
    report_suffix = "_show" if rate_type == '複勝率' else ""
    report_path = os.path.join(output_dir, f"location_period{period_years}_ability{report_suffix}_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬場×期間({period_years}年)×能力指標分析レポート（{rate_type}）\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for track, periods in results.items():
            f.write(f"## {track}競馬場\n\n")
            
            for period, surfaces in periods.items():
                f.write(f"### {period}期間\n\n")
                
                for surface, abilities in surfaces.items():
                    f.write(f"#### {surface}コース\n\n")
                    
                    for ability, corners in abilities.items():
                        f.write(f"##### {ability}と各コーナーバイアスの関係\n\n")
                        
                        for corner, positions in corners.items():
                            f.write(f"###### {corner.replace('バイアス', '')}\n\n")
                            
                            if rate_type == '勝率':
                                f.write("| 分位 | 内側勝率(%) | 内側サンプル数 | 中間勝率(%) | 中間サンプル数 | 外側勝率(%) | 外側サンプル数 |\n")
                            else:  # 複勝率
                                f.write("| 分位 | 内側複勝率(%) | 内側サンプル数 | 中間複勝率(%) | 中間サンプル数 | 外側複勝率(%) | 外側サンプル数 |\n")
                                
                            f.write("|------|------------|--------------|------------|--------------|------------|---------------|\n")
                            
                            # 分位ごとにデータを整理
                            quintile_data = {}
                            for position, rates in positions.items():
                                for item in rates:
                                    quintile = item['分位']
                                    if quintile not in quintile_data:
                                        quintile_data[quintile] = {'内側': {}, '中間': {}, '外側': {}}
                                    
                                    rate_key = '勝率' if rate_type == '勝率' else '複勝率'
                                    quintile_data[quintile][position] = {
                                        rate_key: item.get(rate_key, item.get('勝率', 0)),  # 複勝率がなければ勝率を使用
                                        'サンプル数': item['サンプル数']
                                    }
                            
                            # 分位順に出力
                            for quintile in sorted(quintile_data.keys()):
                                data = quintile_data[quintile]
                                
                                rate_key = '勝率' if rate_type == '勝率' else '複勝率'
                                inner_rate = data['内側'].get(rate_key, '-')
                                inner_samples = data['内側'].get('サンプル数', '-')
                                middle_rate = data['中間'].get(rate_key, '-')
                                middle_samples = data['中間'].get('サンプル数', '-')
                                outer_rate = data['外側'].get(rate_key, '-')
                                outer_samples = data['外側'].get('サンプル数', '-')
                                
                                # 数値データの場合のみ小数点以下2桁を表示
                                inner_rate_str = f"{inner_rate:.2f}" if isinstance(inner_rate, (int, float)) and inner_rate != '-' else inner_rate
                                middle_rate_str = f"{middle_rate:.2f}" if isinstance(middle_rate, (int, float)) and middle_rate != '-' else middle_rate
                                outer_rate_str = f"{outer_rate:.2f}" if isinstance(outer_rate, (int, float)) and outer_rate != '-' else outer_rate
                                
                                f.write(f"| {quintile} | {inner_rate_str} | {inner_samples} | {middle_rate_str} | {middle_samples} | {outer_rate_str} | {outer_samples} |\n")
                            
                            f.write("\n")
    
    print(f"レポートを {report_path} に保存しました。")

def main(period_years=5):
    """
    メイン関数
    
    Args:
        period_years (int): 期間年数（デフォルト: 5年）
    """
    # データフォルダとデータ出力用フォルダを指定
    formatted_folder = "export/SED/formatted"
    bias_folder = "export/with_bias"
    output_folder = "results/period_ability_analysis"
    
    # フォルダが存在するか確認
    if not os.path.exists(formatted_folder):
        print(f"エラー: フォルダ {formatted_folder} が見つかりません。")
        return
    
    if not os.path.exists(bias_folder):
        print(f"エラー: フォルダ {bias_folder} が見つかりません。")
        return
    
    # 出力フォルダを作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # データ読み込み
    df = load_data(formatted_folder, bias_folder)
    
    if df is None:
        print("データの読み込みに失敗しました。")
        return
    
    # 場所×期間×能力指標の分析（勝率と複勝率）
    results, show_results = analyze_location_period_ability(df, output_folder, period_years)
    
    if results and show_results:
        print("分析が完了しました。")
        print(f"結果は {output_folder} に保存されています。")
        print(f"- 勝率分析: location_period{period_years}_ability_analysis")
        print(f"- 複勝率分析: location_period{period_years}_ability_show_analysis")
    else:
        print("分析に失敗しました。")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='競馬場×期間×能力指標の分析')
    parser.add_argument('--period', type=int, default=5, help='分析期間の年数（デフォルト: 5年）')
    
    args = parser.parse_args()
    
    main(period_years=args.period) 
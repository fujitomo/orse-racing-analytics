import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import japanize_matplotlib

def load_formatted_files(folder_path):
    """
    formattedフォルダからCSVファイルを読み込む
    
    Args:
        folder_path (str): CSVファイルが格納されているパス
    
    Returns:
        DataFrame: 読み込んだデータを結合したDataFrame
    """
    print("通常フォーマットのファイルを読み込んでいます...")
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    for file_path in csv_files:
        try:
            # エンコーディングを試行
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"ファイル読み込みエラー: {file_path} - {e}")
                    break
            
            # ファイルが正常に読み込まれた場合のみ追加
            if 'df' in locals() and df is not None:
                # レースIDを作成（後でデータ結合に使用）
                file_name = os.path.basename(file_path)
                date_str = file_name.split('_')[0].replace('SED', '')
                if '場コード' in df.columns and '年' in df.columns and '回' in df.columns and '日' in df.columns and 'R' in df.columns:
                    df['レースID'] = df['場コード'].astype(str) + df['年'].astype(str) + df['回'].astype(str) + df['日'].astype(str) + df['R'].astype(str)
                
                all_data.append(df)
        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生: {e}")
    
    if not all_data:
        print("有効なデータが見つかりませんでした。")
        return None
    
    # 全データを結合
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"合計 {len(combined_df)} 行のデータを読み込みました。")
    return combined_df

def load_bias_files(folder_path):
    """
    with_biasフォルダからCSVファイルを読み込む
    
    Args:
        folder_path (str): CSVファイルが格納されているパス
    
    Returns:
        DataFrame: 読み込んだデータを結合したDataFrame
    """
    print("バイアス情報ファイルを読み込んでいます...")
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*_formatted_with_bias.csv"))
    
    for file_path in csv_files:
        try:
            # エンコーディングを試行
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"ファイル読み込みエラー: {file_path} - {e}")
                    break
            
            # ファイルが正常に読み込まれた場合のみ追加
            if 'df' in locals() and df is not None:
                # レースIDを作成（後でデータ結合に使用）
                file_name = os.path.basename(file_path)
                date_str = file_name.split('_')[0].replace('SED', '')
                if '場コード' in df.columns and '年' in df.columns and '回' in df.columns and '日' in df.columns and 'R' in df.columns:
                    df['レースID'] = df['場コード'].astype(str) + df['年'].astype(str) + df['回'].astype(str) + df['日'].astype(str) + df['R'].astype(str)
                
                all_data.append(df)
        except Exception as e:
            print(f"ファイル {file_path} の処理中にエラーが発生: {e}")
    
    if not all_data:
        print("有効なデータが見つかりませんでした。")
        return None
    
    # 全データを結合
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"合計 {len(combined_df)} 行のデータを読み込みました。")
    return combined_df

def preprocess_data(formatted_df, bias_df):
    """
    データの前処理
    
    Args:
        formatted_df (DataFrame): 通常フォーマットのデータ
        bias_df (DataFrame): バイアス情報のデータ
    
    Returns:
        DataFrame: 結合・前処理後のDataFrame
    """
    print("データの前処理を行っています...")
    
    # カラム名を確認
    print(f"formatted_df カラム: {formatted_df.columns[:10]}...")
    print(f"bias_df カラム: {bias_df.columns[:10]}...")
    
    # レースIDで結合
    if 'レースID' in formatted_df.columns and 'レースID' in bias_df.columns:
        # 必要なカラムを選択（バイアス情報のみ）
        bias_cols = [col for col in bias_df.columns if 'バイアス' in col]
        bias_cols.append('レースID')
        bias_cols.append('馬番')  # 馬番も追加して一意に識別
        
        # 重複するカラムを削除しないように結合（バイアス情報のみ取得）
        merged_df = pd.merge(
            formatted_df,
            bias_df[bias_cols],
            on=['レースID', '馬番'],
            how='left'
        )
        
        print(f"結合後のデータ: {len(merged_df)} 行")
    else:
        print("結合に必要なカラムが見つかりません。")
        merged_df = formatted_df
    
    # データ型変換
    # IDMと素点を数値に変換
    for col in ['IDM', '素点']:
        if col in merged_df.columns:
            try:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            except Exception as e:
                print(f"{col} の変換中にエラーが発生: {e}")
    
    # 着順を数値に変換
    if '着順' in merged_df.columns:
        try:
            merged_df['着順'] = pd.to_numeric(merged_df['着順'], errors='coerce')
        except Exception as e:
            print(f"着順の変換中にエラーが発生: {e}")
    
    # 勝利フラグを追加（1位なら1、それ以外は0）
    merged_df['勝利'] = (merged_df['着順'] == 1).astype(int)
    
    # コーナーバイアスの種類をカウント
    corner_bias_types = {}
    for corner in ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']:
        if corner in merged_df.columns:
            # 括弧で囲まれたバイアス表記（例: (1,3)）からデータを抽出
            merged_df[f'{corner}_内側'] = merged_df[corner].str.contains(r'\([*]?1,|\(1[,)]|\([*]?2,|\(2[,)]|\([*]?3,|\(3[,)]', regex=True).fillna(False).astype(int)
            merged_df[f'{corner}_中間'] = merged_df[corner].str.contains(r'\([*]?[4-9],|\([4-9][,)]|\([*]?1[0-2],|\(1[0-2][,)]', regex=True).fillna(False).astype(int)
            merged_df[f'{corner}_外側'] = merged_df[corner].str.contains(r'\([*]?1[3-9],|\(1[3-9][,)]|\([*]?2[0-9],|\(2[0-9][,)]', regex=True).fillna(False).astype(int)
    
    # 芝・ダート情報の変換
    if pd.api.types.is_numeric_dtype(merged_df['芝ダ障害コード']):
        merged_df['コース種別'] = merged_df['芝ダ障害コード'].map({1: '芝', 2: 'ダート', 3: '障害'})
    else:
        merged_df['コース種別'] = merged_df['芝ダ障害コード']
    
    # レース脚質の変換
    if pd.api.types.is_numeric_dtype(merged_df['レース脚質']):
        merged_df['脚質'] = merged_df['レース脚質'].map({1: '逃げ', 2: '先行', 3: '差し', 4: '追込'})
    else:
        merged_df['脚質'] = merged_df['レース脚質']
    
    return merged_df

def analyze_ability_bias_win_rate(data):
    """
    馬の能力と馬場バイアスと勝率の関連性を分析
    
    Args:
        data (DataFrame): 分析対象のDataFrame
    
    Returns:
        dict: 分析結果
    """
    print("馬の能力とトラックバイアスの勝率への影響を分析中...")
    
    # 必要なカラムが存在するか確認
    ability_columns = ['IDM', '素点']
    available_ability_cols = [col for col in ability_columns if col in data.columns]
    
    if not available_ability_cols:
        print("馬の能力データ（IDM、素点など）が見つかりません。")
        return None
    
    results = {
        '能力別勝率': {},
        'バイアスと能力の関連': {},
        '脚質×能力×バイアス': {}
    }
    
    # 1. 能力指標別の勝率
    for col in available_ability_cols:
        # 欠損値を除外
        valid_data = data.dropna(subset=[col, '勝利'])
        
        if len(valid_data) == 0:
            continue
        
        # 能力値を10分位に分割
        valid_data[f'{col}_分位'] = pd.qcut(valid_data[col], 10, labels=False)
        
        # 分位ごとの勝率を計算
        win_rate_by_ability = valid_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
        win_rate_by_ability['勝率'] = (win_rate_by_ability['mean'] * 100).round(2)
        
        # 能力値の範囲を追加
        quantiles = pd.qcut(valid_data[col], 10, retbins=True)[1]
        win_rate_by_ability['下限'] = [round(quantiles[i], 2) for i in win_rate_by_ability[f'{col}_分位']]
        win_rate_by_ability['上限'] = [round(quantiles[i+1], 2) for i in win_rate_by_ability[f'{col}_分位']]
        win_rate_by_ability['範囲'] = win_rate_by_ability.apply(lambda x: f"{x['下限']}～{x['上限']}", axis=1)
        
        results['能力別勝率'][col] = win_rate_by_ability.to_dict('records')
    
    # 2. コーナー位置とバイアスの関連
    corner_columns = [col for col in data.columns if '_内側' in col or '_中間' in col or '_外側' in col]
    
    if corner_columns:
        for col in available_ability_cols:
            # 欠損値を除外
            valid_data = data.dropna(subset=[col, '勝利'])
            
            # コーナーごとのバイアスと能力の関連
            corner_bias_results = {}
            
            for corner_base in ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']:
                inner_col = f'{corner_base}_内側'
                middle_col = f'{corner_base}_中間'
                outer_col = f'{corner_base}_外側'
                
                if inner_col in data.columns and middle_col in data.columns and outer_col in data.columns:
                    # 内側・中間・外側別のデータを抽出
                    inner_data = valid_data[valid_data[inner_col] == 1]
                    middle_data = valid_data[valid_data[middle_col] == 1]
                    outer_data = valid_data[valid_data[outer_col] == 1]
                    
                    # データがある場合のみ分析
                    corner_results = {}
                    
                    if len(inner_data) > 0:
                        # 能力値を5分位に分割（データ量が少ないため分位数を減らす）
                        inner_data[f'{col}_分位'] = pd.qcut(inner_data[col], 5, labels=False)
                        inner_win_rate = inner_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
                        inner_win_rate['勝率'] = (inner_win_rate['mean'] * 100).round(2)
                        corner_results['内側'] = inner_win_rate.to_dict('records')
                    
                    if len(middle_data) > 0:
                        middle_data[f'{col}_分位'] = pd.qcut(middle_data[col], 5, labels=False)
                        middle_win_rate = middle_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
                        middle_win_rate['勝率'] = (middle_win_rate['mean'] * 100).round(2)
                        corner_results['中間'] = middle_win_rate.to_dict('records')
                    
                    if len(outer_data) > 0:
                        outer_data[f'{col}_分位'] = pd.qcut(outer_data[col], 5, labels=False)
                        outer_win_rate = outer_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
                        outer_win_rate['勝率'] = (outer_win_rate['mean'] * 100).round(2)
                        corner_results['外側'] = outer_win_rate.to_dict('records')
                    
                    if corner_results:
                        corner_bias_results[corner_base] = corner_results
            
            if corner_bias_results:
                results['バイアスと能力の関連'][col] = corner_bias_results
    
    # 3. 脚質×能力×バイアスの分析
    if '脚質' in data.columns:
        for col in available_ability_cols:
            # 欠損値を除外
            valid_data = data.dropna(subset=[col, '勝利', '脚質'])
            
            # 脚質ごとの能力と勝率の関係
            style_ability_results = {}
            
            for style in ['逃げ', '先行', '差し', '追込']:
                style_data = valid_data[valid_data['脚質'] == style]
                
                if len(style_data) > 20:  # 十分なデータがある場合のみ
                    # 能力値を5分位に分割
                    style_data[f'{col}_分位'] = pd.qcut(style_data[col], 5, labels=False, duplicates='drop')
                    style_win_rate = style_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
                    style_win_rate['勝率'] = (style_win_rate['mean'] * 100).round(2)
                    style_ability_results[style] = style_win_rate.to_dict('records')
            
            if style_ability_results:
                results['脚質×能力×バイアス'][col] = style_ability_results
    
    # 4. 競馬場・コース種別ごとの能力とバイアスの関係
    results['競馬場別分析'] = {}
    
    for track in data['場名'].unique():
        track_data = data[data['場名'] == track]
        track_results = {}
        
        for surface in ['芝', 'ダート']:
            surface_data = track_data[track_data['コース種別'] == surface]
            
            if len(surface_data) < 50:  # データが少なすぎる場合はスキップ
                continue
            
            surface_results = {}
            
            # 能力指標ごとに分析
            for col in available_ability_cols:
                valid_surface_data = surface_data.dropna(subset=[col, '勝利'])
                
                if len(valid_surface_data) < 50:
                    continue
                
                # 能力値を5分位に分割
                valid_surface_data[f'{col}_分位'] = pd.qcut(valid_surface_data[col], 5, labels=False, duplicates='drop')
                ability_win_rate = valid_surface_data.groupby(f'{col}_分位')['勝利'].agg(['mean', 'count']).reset_index()
                ability_win_rate['勝率'] = (ability_win_rate['mean'] * 100).round(2)
                
                surface_results[col] = ability_win_rate.to_dict('records')
            
            if surface_results:
                track_results[surface] = surface_results
        
        if track_results:
            results['競馬場別分析'][track] = track_results
    
    return results

def visualize_results(results, output_dir):
    """
    分析結果を可視化
    
    Args:
        results (dict): 分析結果
        output_dir (str): 出力ディレクトリ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 能力別勝率のグラフ
    for ability, data in results['能力別勝率'].items():
        plt.figure(figsize=(10, 6))
        
        # データ準備
        df = pd.DataFrame(data)
        x = df['範囲']
        y = df['勝率']
        
        # プロット
        bars = plt.bar(x, y, color='skyblue')
        
        # 数値を表示
        for bar, rate in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{rate:.2f}%', ha='center', va='bottom')
        
        plt.title(f'{ability}別勝率')
        plt.xlabel(ability)
        plt.ylabel('勝率 (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{ability}別勝率.png'), dpi=300)
        plt.close()
    
    # 2. コーナーポジションと能力の関係
    for ability, corner_data in results['バイアスと能力の関連'].items():
        for corner, positions in corner_data.items():
            plt.figure(figsize=(12, 7))
            
            # 位置ごとのデータを準備
            for position, data in positions.items():
                df = pd.DataFrame(data)
                plt.plot(df[f'{ability}_分位'], df['勝率'], marker='o', label=f'{position}')
            
            plt.title(f'{corner}と{ability}の関係')
            plt.xlabel(f'{ability}分位')
            plt.ylabel('勝率 (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{corner}_{ability}_関係.png'), dpi=300)
            plt.close()
    
    # 3. 脚質×能力×バイアスの関係
    for ability, style_data in results['脚質×能力×バイアス'].items():
        plt.figure(figsize=(12, 7))
        
        # 脚質ごとのデータを準備
        for style, data in style_data.items():
            df = pd.DataFrame(data)
            plt.plot(df[f'{ability}_分位'], df['勝率'], marker='o', label=style)
        
        plt.title(f'脚質別の{ability}と勝率の関係')
        plt.xlabel(f'{ability}分位')
        plt.ylabel('勝率 (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'脚質_{ability}_勝率.png'), dpi=300)
        plt.close()
    
    # 4. 競馬場別の分析
    for track, track_data in results['競馬場別分析'].items():
        for surface, surface_data in track_data.items():
            for ability, ability_data in surface_data.items():
                plt.figure(figsize=(10, 6))
                
                # データ準備
                df = pd.DataFrame(ability_data)
                
                # プロット
                bars = plt.bar(df[f'{ability}_分位'], df['勝率'], color='lightgreen')
                
                # 数値を表示
                for bar, rate in zip(bars, df['勝率']):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                            f'{rate:.2f}%', ha='center', va='bottom')
                
                plt.title(f'{track}競馬場({surface})の{ability}別勝率')
                plt.xlabel(f'{ability}分位')
                plt.ylabel('勝率 (%)')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{track}_{surface}_{ability}_勝率.png'), dpi=300)
                plt.close()

def generate_report(results, output_dir):
    """
    分析結果のレポートを生成
    
    Args:
        results (dict): 分析結果
        output_dir (str): 出力ディレクトリ
    """
    report_path = os.path.join(output_dir, "ability_bias_win_rate_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 馬の能力とトラックバイアスの勝率への影響分析レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 能力別勝率
        f.write("## 1. 能力別勝率\n\n")
        for ability, data in results['能力別勝率'].items():
            f.write(f"### {ability}別勝率\n\n")
            f.write("| 分位 | 範囲 | データ数 | 勝率(%) |\n")
            f.write("|------|------|----------|--------|\n")
            
            df = pd.DataFrame(data)
            for _, row in df.iterrows():
                f.write(f"| {row[f'{ability}_分位']} | {row['範囲']} | {row['count']} | {row['勝率']:.2f} |\n")
            
            f.write("\n")
        
        # 2. コーナーポジションと能力の関係
        f.write("## 2. コーナーポジションと能力の関係\n\n")
        for ability, corner_data in results['バイアスと能力の関連'].items():
            f.write(f"### {ability}とコーナーポジションの関係\n\n")
            
            for corner, positions in corner_data.items():
                f.write(f"#### {corner}\n\n")
                f.write("| 分位 | 内側勝率(%) | 中間勝率(%) | 外側勝率(%) |\n")
                f.write("|------|------------|------------|------------|\n")
                
                # 位置ごとのデータを統合
                position_data = {}
                for position, data in positions.items():
                    df = pd.DataFrame(data)
                    for _, row in df.iterrows():
                        position_data.setdefault(row[f'{ability}_分位'], {})
                        position_data[row[f'{ability}_分位']][position] = row['勝率']
                
                # 分位ごとに出力
                for quintile, rates in sorted(position_data.items()):
                    inner_rate = rates.get('内側', '-')
                    middle_rate = rates.get('中間', '-')
                    outer_rate = rates.get('外側', '-')
                    f.write(f"| {quintile} | {inner_rate:.2f} | {middle_rate:.2f} | {outer_rate:.2f} |\n")
                
                f.write("\n")
        
        # 3. 脚質×能力×バイアスの関係
        f.write("## 3. 脚質×能力×バイアスの関係\n\n")
        for ability, style_data in results['脚質×能力×バイアス'].items():
            f.write(f"### {ability}と脚質の関係\n\n")
            f.write("| 分位 | 逃げ勝率(%) | 先行勝率(%) | 差し勝率(%) | 追込勝率(%) |\n")
            f.write("|------|------------|------------|------------|------------|\n")
            
            # 脚質ごとのデータを統合
            style_rates = {}
            for style, data in style_data.items():
                df = pd.DataFrame(data)
                for _, row in df.iterrows():
                    style_rates.setdefault(row[f'{ability}_分位'], {})
                    style_rates[row[f'{ability}_分位']][style] = row['勝率']
            
            # 分位ごとに出力
            for quintile, rates in sorted(style_rates.items()):
                nige_rate = rates.get('逃げ', '-')
                senko_rate = rates.get('先行', '-')
                sashi_rate = rates.get('差し', '-')
                oikomi_rate = rates.get('追込', '-')
                
                # 数値データの場合のみ小数点以下2桁を表示
                nige_str = f"{nige_rate:.2f}" if isinstance(nige_rate, (int, float)) else nige_rate
                senko_str = f"{senko_rate:.2f}" if isinstance(senko_rate, (int, float)) else senko_rate
                sashi_str = f"{sashi_rate:.2f}" if isinstance(sashi_rate, (int, float)) else sashi_rate
                oikomi_str = f"{oikomi_rate:.2f}" if isinstance(oikomi_rate, (int, float)) else oikomi_rate
                
                f.write(f"| {quintile} | {nige_str} | {senko_str} | {sashi_str} | {oikomi_str} |\n")
            
            f.write("\n")
        
        # 4. 競馬場別の分析
        f.write("## 4. 競馬場別の能力と勝率の関係\n\n")
        for track, track_data in results['競馬場別分析'].items():
            f.write(f"### {track}競馬場\n\n")
            
            for surface, surface_data in track_data.items():
                f.write(f"#### {surface}コース\n\n")
                
                for ability, ability_data in surface_data.items():
                    f.write(f"##### {ability}別勝率\n\n")
                    f.write("| 分位 | データ数 | 勝率(%) |\n")
                    f.write("|------|----------|--------|\n")
                    
                    df = pd.DataFrame(ability_data)
                    for _, row in df.iterrows():
                        f.write(f"| {row[f'{ability}_分位']} | {row['count']} | {row['勝率']:.2f} |\n")
                    
                    f.write("\n")
    
    print(f"分析レポートを {report_path} に保存しました。")

def main():
    # データフォルダとデータ出力用フォルダを指定
    formatted_folder = "export/SED/formatted"
    bias_folder = "export/with_bias"
    output_folder = "results/ability_bias_analysis"
    
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
    formatted_df = load_formatted_files(formatted_folder)
    bias_df = load_bias_files(bias_folder)
    
    if formatted_df is None or bias_df is None:
        print("データの読み込みに失敗しました。")
        return
    
    # データ前処理
    merged_df = preprocess_data(formatted_df, bias_df)
    
    if merged_df is None:
        print("データの前処理に失敗しました。")
        return
    
    # 分析実行
    results = analyze_ability_bias_win_rate(merged_df)
    
    if results:
        # 結果の可視化
        visualize_results(results, output_folder)
        
        # レポート生成
        generate_report(results, output_folder)
        
        print("分析が完了しました。")
    else:
        print("分析中にエラーが発生しました。")

if __name__ == "__main__":
    main() 
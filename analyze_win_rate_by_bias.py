import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import japanize_matplotlib

def load_bias_files(folder_path):
    """
    指定フォルダ内のバイアス情報CSVファイルを読み込む
    
    Args:
        folder_path (str): with_biasフォルダのパス
    
    Returns:
        DataFrame: 複数ファイルを結合したDataFrame
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

def analyze_win_rate_by_bias(data):
    """
    バイアス情報と勝率の関係を分析
    
    Args:
        data (DataFrame): 分析対象のDataFrame
    
    Returns:
        dict: 分析結果
    """
    # 必要なカラムが存在することを確認
    required_columns = ['場名', '芝ダ障害コード', 'レース脚質', '着順', '１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"必要なカラムがありません: {missing_columns}")
        return None
    
    # データの前処理
    # 芝・ダート情報の変換
    if pd.api.types.is_numeric_dtype(data['芝ダ障害コード']):
        data['コース種別'] = data['芝ダ障害コード'].map({1: '芝', 2: 'ダート', 3: '障害'})
    else:
        data['コース種別'] = data['芝ダ障害コード']
    
    # レース脚質の変換
    if pd.api.types.is_numeric_dtype(data['レース脚質']):
        data['脚質'] = data['レース脚質'].map({1: '逃げ', 2: '先行', 3: '差し', 4: '追込'})
    else:
        data['脚質'] = data['レース脚質']
    
    # 勝利を1位とする
    data['勝利'] = (data['着順'] == 1).astype(int)
    
    # コーナーバイアスの種類をカウント
    corner_bias_types = {}
    for corner in ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']:
        if corner in data.columns:
            # 括弧で囲まれたバイアス表記（例: (1,3)）からデータを抽出
            data[f'{corner}_内側'] = data[corner].str.contains(r'\([*]?1,|\(1[,)]|\([*]?2,|\(2[,)]|\([*]?3,|\(3[,)]', regex=True).fillna(False).astype(int)
            data[f'{corner}_中間'] = data[corner].str.contains(r'\([*]?[4-9],|\([4-9][,)]|\([*]?1[0-2],|\(1[0-2][,)]', regex=True).fillna(False).astype(int)
            data[f'{corner}_外側'] = data[corner].str.contains(r'\([*]?1[3-9],|\(1[3-9][,)]|\([*]?2[0-9],|\(2[0-9][,)]', regex=True).fillna(False).astype(int)
            
            unique_values = data[corner].dropna().unique()
            corner_bias_types[corner] = unique_values
    
    # 分析結果
    results = {
        '脚質別勝率': {},
        'コーナー位置別勝率': {},
        '競馬場別バイアス勝率': {}
    }
    
    # 1. 脚質別の勝率
    running_style_win = data.groupby('脚質')['勝利'].agg(['sum', 'count']).reset_index()
    running_style_win['勝率'] = (running_style_win['sum'] / running_style_win['count'] * 100).round(2)
    results['脚質別勝率'] = running_style_win.to_dict('records')
    
    # 2. コーナーポジション別勝率（内・中・外）
    for corner in ['１角バイアス', '２角バイアス', '３角バイアス', '４角バイアス', '直線バイアス']:
        if f'{corner}_内側' in data.columns:
            # 内側にいる場合の勝率
            inner_win = data[data[f'{corner}_内側'] == 1]['勝利'].mean() * 100
            # 中間にいる場合の勝率
            middle_win = data[data[f'{corner}_中間'] == 1]['勝利'].mean() * 100
            # 外側にいる場合の勝率
            outer_win = data[data[f'{corner}_外側'] == 1]['勝利'].mean() * 100
            
            results['コーナー位置別勝率'][corner] = {
                '内側': round(inner_win, 2),
                '中間': round(middle_win, 2),
                '外側': round(outer_win, 2)
            }
    
    # 3. 競馬場・コース種別・脚質別の勝率
    for track in data['場名'].unique():
        track_data = data[data['場名'] == track]
        for surface in track_data['コース種別'].unique():
            surface_data = track_data[track_data['コース種別'] == surface]
            style_win = surface_data.groupby('脚質')['勝利'].agg(['sum', 'count']).reset_index()
            style_win['勝率'] = (style_win['sum'] / style_win['count'] * 100).round(2)
            
            if track not in results['競馬場別バイアス勝率']:
                results['競馬場別バイアス勝率'][track] = {}
                
            results['競馬場別バイアス勝率'][track][surface] = style_win.to_dict('records')
    
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
    
    # 脚質別勝率のグラフ
    plt.figure(figsize=(10, 6))
    
    styles = []
    win_rates = []
    for style_data in results['脚質別勝率']:
        styles.append(style_data['脚質'])
        win_rates.append(style_data['勝率'])
    
    bars = plt.bar(styles, win_rates, color=['#FF9999', '#FFCC99', '#99CCFF', '#9999FF'])
    
    # 数値を表示
    for bar, rate in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{rate:.2f}%', ha='center', va='bottom')
    
    plt.title('脚質別勝率')
    plt.ylabel('勝率 (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '脚質別勝率.png'), dpi=300)
    plt.close()
    
    # コーナー位置別勝率のグラフ
    plt.figure(figsize=(12, 8))
    
    corners = list(results['コーナー位置別勝率'].keys())
    inner_rates = [results['コーナー位置別勝率'][corner]['内側'] for corner in corners]
    middle_rates = [results['コーナー位置別勝率'][corner]['中間'] for corner in corners]
    outer_rates = [results['コーナー位置別勝率'][corner]['外側'] for corner in corners]
    
    x = np.arange(len(corners))
    width = 0.25
    
    plt.bar(x - width, inner_rates, width, label='内側', color='#FF9999')
    plt.bar(x, middle_rates, width, label='中間', color='#FFCC99')
    plt.bar(x + width, outer_rates, width, label='外側', color='#99CCFF')
    
    plt.xlabel('コーナー')
    plt.ylabel('勝率 (%)')
    plt.title('コーナー位置別勝率')
    plt.xticks(x, corners)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'コーナー位置別勝率.png'), dpi=300)
    plt.close()
    
    # 競馬場・コース種別・脚質別の勝率ヒートマップ
    for track, surfaces in results['競馬場別バイアス勝率'].items():
        for surface, style_data in surfaces.items():
            # データフレームに変換
            df = pd.DataFrame(style_data)
            
            # ピボットテーブル作成
            pivot_df = df.pivot_table(index=['脚質'], values=['勝率'], aggfunc=np.mean)
            
            plt.figure(figsize=(8, 4))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': '勝率 (%)'})
            plt.title(f'{track}競馬場 {surface} - 脚質別勝率')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{track}_{surface}_脚質別勝率.png'), dpi=300)
            plt.close()

def generate_report(results, output_dir):
    """
    分析結果のレポートを生成
    
    Args:
        results (dict): 分析結果
        output_dir (str): 出力ディレクトリ
    """
    report_path = os.path.join(output_dir, "bias_win_rate_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# バイアス傾向と勝率の関係分析レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 脚質別勝率
        f.write("## 脚質別勝率\n\n")
        f.write("| 脚質 | 勝利数 | 出走数 | 勝率(%) |\n")
        f.write("|------|--------|--------|--------|\n")
        
        for style_data in results['脚質別勝率']:
            f.write(f"| {style_data['脚質']} | {style_data['sum']} | {style_data['count']} | {style_data['勝率']} |\n")
        
        # コーナー位置別勝率
        f.write("\n## コーナー位置別勝率\n\n")
        f.write("| コーナー | 内側(%) | 中間(%) | 外側(%) |\n")
        f.write("|----------|--------|--------|--------|\n")
        
        for corner, rates in results['コーナー位置別勝率'].items():
            f.write(f"| {corner} | {rates['内側']} | {rates['中間']} | {rates['外側']} |\n")
        
        # 競馬場別バイアス勝率
        f.write("\n## 競馬場・コース種別・脚質別の勝率\n\n")
        
        for track, surfaces in results['競馬場別バイアス勝率'].items():
            f.write(f"\n### {track}競馬場\n\n")
            
            for surface, style_data in surfaces.items():
                f.write(f"\n#### {surface}コース\n\n")
                f.write("| 脚質 | 勝利数 | 出走数 | 勝率(%) |\n")
                f.write("|------|--------|--------|--------|\n")
                
                for data in style_data:
                    if '脚質' in data:  # 有効なデータのみ処理
                        f.write(f"| {data['脚質']} | {data['sum']} | {data['count']} | {data['勝率']} |\n")
    
    print(f"分析レポートを {report_path} に保存しました。")

if __name__ == "__main__":
    # データ読み込み
    input_folder = "export/with_bias"
    output_folder = "results/bias_win_rate_analysis"
    
    data = load_bias_files(input_folder)
    
    if data is not None:
        # バイアスと勝率の関係を分析
        results = analyze_win_rate_by_bias(data)
        
        if results:
            # 分析結果を可視化
            visualize_results(results, output_folder)
            
            # レポート生成
            generate_report(results, output_folder)
            
            print("分析が完了しました。")
        else:
            print("分析中にエラーが発生しました。")
    else:
        print("データの読み込みに失敗しました。") 
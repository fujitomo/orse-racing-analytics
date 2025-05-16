import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from track_bias_analyzer import TrackBiasAnalyzer

def analyze_all_files_in_folder(folder_path, output_dir, quiet=False):
    """
    指定したフォルダ内のすべてのCSVファイルを分析する
    
    Args:
        folder_path (str): 分析対象のCSVファイルが格納されているフォルダパス
        output_dir (str): 分析結果を出力するフォルダパス
        quiet (bool): True の場合、出力メッセージを抑制する
    """
    # 出力用フォルダを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力フォルダ {output_dir} を作成しました。")
    
    # 分析対象のファイル一覧を取得
    csv_files = glob.glob(os.path.join(folder_path, "*_formatted_with_bias.csv"))
    csv_files.sort()  # 日付順にソート
    
    print(f"{len(csv_files)}件のファイルが見つかりました。")
    
    # 分析用のアナライザインスタンスを作成
    analyzer = TrackBiasAnalyzer()
    
    # 分析結果のサマリーを保存するDataFrame
    summary_data = []
    
    # 各ファイルを処理
    for i, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        print(f"\n処理中: {file_name} ({i+1}/{len(csv_files)})")
        
        # 日付情報を取得
        date_str = file_name.split('_')[0].replace('SED', '')
        
        # ファイルを読み込む
        analyzer.load_csv(file_path)
        analyzer.preprocess_data()
        
        # 競馬場ごとの分析
        track_names = analyzer.df['場名'].unique() if '場名' in analyzer.df.columns else []
        
        for track_name in track_names:
            if not quiet:
                print(f"  {track_name}の分析開始")
            
            # 芝・ダート別に分析
            for track_type in ['芝', 'ダート']:
                # トラックバイアスの分析
                bias_result = analyzer.analyze_track_bias(
                    track_name=track_name,
                    track_type=track_type
                )
                
                if bias_result is None:
                    continue
                
                # コーナーバイアスの分析
                corner_bias_result = analyzer.analyze_corner_bias(
                    track_name=track_name, 
                    track_type=track_type
                )
                
                # 分析結果をサマリーに追加
                summary_entry = {
                    '日付': date_str,
                    '競馬場': track_name,
                    'コース種別': track_type,
                    'バイアス': bias_result['バイアス'],
                    '強度': bias_result['強度'],
                    'データ件数': bias_result['データ件数'],
                    '上位入着馬数': bias_result['上位入着馬数'],
                    '逃げ(%)': bias_result['脚質分布'].get('逃げ', 0),
                    '先行(%)': bias_result['脚質分布'].get('先行', 0),
                    '差し(%)': bias_result['脚質分布'].get('差し', 0),
                    '追込(%)': bias_result['脚質分布'].get('追込', 0),
                }
                
                # 内外バイアスがあれば追加
                if '内外バイアス' in bias_result:
                    summary_entry['内外バイアス'] = bias_result['内外バイアス']
                
                # コーナーバイアスの主要情報を追加
                if corner_bias_result:
                    for corner, data in corner_bias_result['コーナーバイアス'].items():
                        if '主要バイアス' in data:
                            summary_entry[f'{corner}_主要バイアス'] = data['主要バイアス']
                            summary_entry[f'{corner}_主要バイアス割合'] = data['主要バイアス割合']
                
                summary_data.append(summary_entry)
                
                # 可視化とファイル出力
                output_base = os.path.join(output_dir, f"{date_str}_{track_name}_{track_type}")
                
                # トラックバイアスの可視化 - 画面出力を抑制
                bias_output_file = f"{output_base}_track_bias.png"
                plt_show_orig = plt.show
                if quiet:
                    plt.show = lambda: None  # 画面表示を抑制
                
                # 描画と保存（出力メッセージを抑制）
                try:
                    analyzer.visualize_bias(
                        bias_result,
                        f"{track_name}競馬場({track_type})のトラックバイアス - {date_str}",
                        bias_output_file,
                        quiet=quiet
                    )
                finally:
                    plt.show = plt_show_orig  # 元に戻す
                
                # コーナーバイアスの可視化 - 画面出力を抑制
                if corner_bias_result:
                    corner_output_file = f"{output_base}_corner_bias.png"
                    plt_show_orig = plt.show
                    if quiet:
                        plt.show = lambda: None  # 画面表示を抑制
                    
                    try:
                        analyzer.visualize_corner_bias(
                            corner_bias_result,
                            f"{track_name}競馬場({track_type})のコーナー別バイアス - {date_str}",
                            corner_output_file,
                            quiet=quiet
                        )
                    finally:
                        plt.show = plt_show_orig  # 元に戻す
                
                # JSON出力
                json_output_file = f"{output_base}_analysis.json"
                result_data = {
                    'track_bias': bias_result,
                    'corner_bias': corner_bias_result,
                    'date': date_str,
                    'track_name': track_name,
                    'track_type': track_type
                }
                analyzer.export_results_to_json(result_data, json_output_file, quiet=quiet)
    
    # サマリーDataFrameを作成して出力
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, "track_bias_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nサマリー情報を {summary_csv_path} に保存しました。")
        
        # 月別や競馬場別の集計も行う
        summary_df['年月'] = summary_df['日付'].str[:6]
        
        # 月別・競馬場別の分析
        monthly_track_summary = summary_df.groupby(['年月', '競馬場', 'コース種別']).agg({
            'バイアス': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A',
            '強度': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A',
            'データ件数': 'sum',
            '上位入着馬数': 'sum',
            '逃げ(%)': 'mean',
            '先行(%)': 'mean',
            '差し(%)': 'mean',
            '追込(%)': 'mean'
        }).reset_index()
        
        monthly_summary_path = os.path.join(output_dir, "track_bias_monthly_summary.csv")
        monthly_track_summary.to_csv(monthly_summary_path, index=False, encoding='utf-8-sig')
        print(f"月別・競馬場別サマリーを {monthly_summary_path} に保存しました。")
        
        return summary_df
    
    return None

def generate_track_bias_report(summary_df, output_dir):
    """
    トラックバイアスの分析レポートを生成する
    
    Args:
        summary_df (DataFrame): 分析結果のサマリー情報
        output_dir (str): 出力ディレクトリ
    """
    # 競馬場ごとの傾向分析
    track_summary = summary_df.groupby(['競馬場', 'コース種別']).agg({
        'バイアス': lambda x: x.value_counts().index[0],
        '強度': lambda x: x.value_counts().index[0],
        'データ件数': 'sum',
        '上位入着馬数': 'sum',
        '逃げ(%)': 'mean',
        '先行(%)': 'mean',
        '差し(%)': 'mean',
        '追込(%)': 'mean'
    }).reset_index()
    
    # 傾向レポートのMarkdownファイル生成
    report_path = os.path.join(output_dir, "track_bias_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# トラックバイアス分析レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"分析対象レース数: {summary_df['データ件数'].sum()}\n\n")
        
        f.write("## 競馬場別バイアス傾向\n\n")
        f.write("| 競馬場 | コース種別 | 主要バイアス | 強度 | 平均逃げ(%) | 平均先行(%) | 平均差し(%) | 平均追込(%) |\n")
        f.write("|--------|------------|------------|------|-----------|-----------|-----------|------------|\n")
        
        for _, row in track_summary.iterrows():
            f.write(f"| {row['競馬場']} | {row['コース種別']} | {row['バイアス']} | {row['強度']} | {row['逃げ(%)']:.1f} | {row['先行(%)']:.1f} | {row['差し(%)']:.1f} | {row['追込(%)']:.1f} |\n")
        
        # コーナーバイアスについての分析（可能であれば）
        corner_columns = [col for col in summary_df.columns if '_主要バイアス' in col]
        if corner_columns:
            f.write("\n## コーナー別バイアス傾向\n\n")
            
            corners = [col.split('_')[0] for col in corner_columns if '主要バイアス' in col]
            
            for corner in sorted(set(corners)):
                f.write(f"\n### {corner}のバイアス\n\n")
                f.write("| 競馬場 | コース種別 | 主要バイアス | 出現率(%) |\n")
                f.write("|--------|------------|------------|----------|\n")
                
                corner_data = summary_df.groupby(['競馬場', 'コース種別']).agg({
                    f'{corner}_主要バイアス': lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else 'N/A',
                    f'{corner}_主要バイアス割合': 'mean'
                }).reset_index()
                
                for _, row in corner_data.iterrows():
                    bias = row.get(f'{corner}_主要バイアス', 'N/A')
                    if bias == 'N/A':
                        continue
                    percentage = row.get(f'{corner}_主要バイアス割合', 0)
                    f.write(f"| {row['競馬場']} | {row['コース種別']} | {bias} | {percentage:.1f} |\n")
    
    print(f"分析レポートを {report_path} に保存しました。")

if __name__ == "__main__":
    # 分析対象のフォルダとデータ出力用フォルダを指定
    input_folder = "export/with_bias"
    output_folder = "results/track_bias_analysis"
    
    # すべてのファイルを分析（メッセージを抑制）
    summary_df = analyze_all_files_in_folder(input_folder, output_folder, quiet=True)
    
    if summary_df is not None:
        # レポート生成
        generate_track_bias_report(summary_df, output_folder) 
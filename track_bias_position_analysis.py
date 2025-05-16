"""
競馬場バイアス×位置分析システム

このプログラムは以下の分析を実行します：

1. バイアスカテゴリ分析
   - 各競馬場のバイアス傾向をカテゴリ化（内・中・外）
   - カテゴリ別の複勝率分析
   - 統計的有意性の検定
   - 競馬場間の比較

2. コーナー通過位置分析
   - 各コーナーでの通過位置とバイアスの関係
   - 位置別の複勝率推移
   - 平均通過位置との相関
   - 競馬場別の特徴抽出

3. 可視化
   - バイアスカテゴリ別複勝率グラフ
   - コーナー通過位置分析グラフ
   - 競馬場別ヒートマップ
   - 総合比較チャート

4. 出力内容
   - 詳細分析レポート（Markdown形式）
   - 統計的検定結果
   - 注目すべき発見のCSV
   - 各種分析グラフ

使用方法：
python track_bias_position_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import japanize_matplotlib
import os
from collections import defaultdict

# 結果保存ディレクトリの確認
results_dir = 'results/track_bias_position_analysis'
os.makedirs(results_dir, exist_ok=True)

def load_data(file_path):
    print(f"データファイル {file_path} を読み込んでいます...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"データ読み込み完了: {len(df)} 行")
    return df

def parse_track_bias(bias_str, is_corner_45=False):
    """トラックバイアスの文字列をパースしてカテゴリに変換する
    
    Args:
        bias_str: バイアス文字列 (例: "(*10,14,11)")
        is_corner_45: ４角と直線かどうか（5段階の場合True）
    
    Returns:
        カテゴリ文字列: 'inside', 'middle', 'outside', 'far_inside', 'far_outside' のいずれか
    """
    if pd.isna(bias_str) or bias_str == 'nan':
        return np.nan
    
    # カテゴリの定義（1-3は内中外、1-5は最内～大外）
    categories_3 = {1: 'inside', 2: 'middle', 3: 'outside'}
    categories_5 = {1: 'far_inside', 2: 'inside', 3: 'middle', 4: 'outside', 5: 'far_outside'}
    
    categories = categories_5 if is_corner_45 else categories_3
    
    # バイアス値を抽出
    try:
        # (*10,14) のような形式から最初の値を取得
        if '(' in bias_str and ')' in bias_str:
            content = bias_str.split('(')[1].split(')')[0]
            # アスタリスクを削除
            if '*' in content:
                content = content.replace('*', '')
            # カンマで区切られた最初の値を取得
            if ',' in content:
                content = content.split(',')[0]
            # 数値化
            try:
                value = int(content)
                return categories.get(value, np.nan)
            except:
                return np.nan
        # ハイフンを含む形式（例: "7-4,3-1"）
        elif '-' in bias_str:
            parts = bias_str.split('-')[0]
            if ',' in parts:
                parts = parts.split(',')[0]
            try:
                value = int(parts)
                return categories.get(value, np.nan)
            except:
                return np.nan
        # 数字列（例: "91511--"）
        else:
            # 最初の数字を取得
            for char in bias_str:
                if char.isdigit():
                    value = int(char)
                    return categories.get(value, np.nan)
            return np.nan
    except:
        return np.nan

def analyze_bias_categories(df):
    """トラックバイアスをカテゴリとして分析"""
    # 複勝率の計算（着順が3位以内なら複勝）
    df['is_show'] = df['着順'] <= 3
    
    # 場コードのマッピング
    race_course_codes = {
        1: '札幌', 2: '函館', 3: '福島', 4: '新潟', 
        5: '東京', 6: '中山', 7: '中京', 8: '京都', 
        9: '阪神', 10: '小倉'
    }
    
    # バイアスカラムとその解釈
    bias_columns = [
        ('１角バイアス', False),  # False=3段階
        ('２角バイアス', False),
        ('向正バイアス', False),
        ('３角バイアス', False),
        ('４角バイアス', True),   # True=5段階
        ('直線バイアス', True)
    ]
    
    # バイアスをカテゴリに変換
    for col, is_corner_45 in bias_columns:
        df[f'{col}_cat'] = df[col].apply(lambda x: parse_track_bias(x, is_corner_45))
        # カテゴリの数をカウント
        counts = df[f'{col}_cat'].value_counts().to_dict()
        print(f"{col}のカテゴリ分布: {counts}")
    
    # 結果格納用
    results = {}
    plots_data = []
    
    # 各競馬場ごとに分析
    for course_code, course_name in race_course_codes.items():
        course_df = df[df['場コード'] == course_code].copy()
        
        if len(course_df) == 0:
            continue
            
        print(f"\n{course_name}（コード: {course_code}）の分析 - データ数: {len(course_df)}")
        course_results = {}
        
        # 各バイアスごとに分析
        for col, is_corner_45 in bias_columns:
            cat_col = f'{col}_cat'
            valid_data = course_df.dropna(subset=[cat_col, 'is_show'])
            
            if len(valid_data) < 5:  # データが少なすぎる場合はスキップ
                print(f"  - {col}: 有効なデータが少なすぎるためスキップします（{len(valid_data)}件）")
                continue
            
            # カテゴリごとの複勝率計算
            category_stats = valid_data.groupby(cat_col)['is_show'].agg(['mean', 'count']).reset_index()
            category_stats.columns = ['カテゴリ', '複勝率', '件数']
            
            # 全体の複勝率
            overall_show_rate = valid_data['is_show'].mean()
            
            course_results[col] = {
                'category_stats': category_stats.to_dict('records'),
                'overall_show_rate': overall_show_rate,
                'data_count': len(valid_data)
            }
            
            print(f"  - {col}: 各カテゴリの複勝率")
            print(category_stats)
            
            # グラフ用データの追加
            for _, row in category_stats.iterrows():
                plots_data.append({
                    '競馬場': course_name,
                    'バイアス': col.replace('バイアス', ''),
                    'カテゴリ': row['カテゴリ'],
                    '複勝率': row['複勝率'],
                    '件数': row['件数'],
                    '全体平均との差': row['複勝率'] - overall_show_rate
                })
            
            # グラフ作成
            if len(category_stats) > 1:  # 複数カテゴリがある場合のみグラフ作成
                plt.figure(figsize=(10, 6))
                
                # 複勝率のバーチャート
                ax = sns.barplot(x='カテゴリ', y='複勝率', data=category_stats)
                
                # 全体平均の水平線
                plt.axhline(y=overall_show_rate, color='r', linestyle='--', label=f'全体平均: {overall_show_rate:.2f}')
                
                # 各バーに値を表示
                for i, row in enumerate(category_stats.itertuples()):
                    ax.text(i, row.複勝率 + 0.02, f'{row.複勝率:.2f}\n(n={row.件数})', 
                            ha='center', va='bottom')
                
                # グラフの装飾
                plt.title(f"{course_name} - {col.replace('バイアス', '')}バイアスと複勝率の関係")
                plt.ylabel('複勝率')
                plt.ylim(0, 1.1)  # y軸の範囲を0-1に設定
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                
                # ファイル保存
                file_path = os.path.join(results_dir, f"{course_code}_{course_name}_{col}_category.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        results[course_code] = course_results
    
    # 総合グラフの作成（全競馬場のデータを集約）
    if plots_data:
        plot_df = pd.DataFrame(plots_data)
        
        # バイアスごとに全競馬場のカテゴリ別複勝率を比較
        for bias_name in plot_df['バイアス'].unique():
            bias_data = plot_df[plot_df['バイアス'] == bias_name].copy()
            
            if len(bias_data) >= 5:  # 十分なデータがある場合
                plt.figure(figsize=(12, 8))
                
                # 競馬場ごとにカテゴリ別の複勝率を表示
                sns.barplot(x='カテゴリ', y='複勝率', hue='競馬場', data=bias_data)
                
                plt.title(f"各競馬場における{bias_name}バイアスと複勝率の関係")
                plt.ylabel('複勝率')
                plt.ylim(0, 1.1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend(title='競馬場')
                
                # ファイル保存
                file_path = os.path.join(results_dir, f"all_courses_{bias_name}_category.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 全体平均との差のヒートマップ
                if len(bias_data) >= 10:
                    plt.figure(figsize=(14, 8))
                    
                    # カテゴリと競馬場でピボットテーブル作成
                    pivot_data = bias_data.pivot_table(
                        values='全体平均との差', 
                        index='競馬場', 
                        columns='カテゴリ',
                        aggfunc='mean'
                    ).fillna(0)
                    
                    # ヒートマップ描画
                    sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
                    
                    plt.title(f"{bias_name}バイアス - 競馬場別の全体平均との差")
                    plt.tight_layout()
                    
                    # ファイル保存
                    file_path = os.path.join(results_dir, f"heatmap_{bias_name}_vs_average.png")
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close()
    
    return results

def analyze_corner_positions(df):
    """コーナー通過位置とバイアスの関係を分析"""
    # 複勝率の計算
    df['is_show'] = df['着順'] <= 3
    
    # 場コードのマッピング
    race_course_codes = {
        1: '札幌', 2: '函館', 3: '福島', 4: '新潟', 
        5: '東京', 6: '中山', 7: '中京', 8: '京都', 
        9: '阪神', 10: '小倉'
    }
    
    # コーナー通過位置カラム
    corner_columns = ['コーナー順位1', 'コーナー順位2', 'コーナー順位3', 'コーナー順位4']
    
    # バイアスカラムとマッチするコーナー
    bias_corner_pairs = [
        ('１角バイアス', 'コーナー順位1', False),
        ('２角バイアス', 'コーナー順位2', False),
        ('３角バイアス', 'コーナー順位3', False),
        ('４角バイアス', 'コーナー順位4', True)
    ]
    
    # 各コーナーのデータ分布を確認
    for corner in corner_columns:
        valid_count = df[corner].notna().sum()
        print(f"{corner}の有効データ数: {valid_count}")
    
    # バイアスを位置カテゴリに変換
    for bias_col, _, is_corner_45 in bias_corner_pairs:
        df[f'{bias_col}_cat'] = df[bias_col].apply(lambda x: parse_track_bias(x, is_corner_45))
    
    # 結果格納用
    results = {}
    
    # 各競馬場ごとに分析
    for course_code, course_name in race_course_codes.items():
        course_df = df[df['場コード'] == course_code].copy()
        
        if len(course_df) == 0:
            continue
            
        print(f"\n{course_name}（コード: {course_code}）のコーナー位置分析 - データ数: {len(course_df)}")
        corner_results = {}
        
        # 各コーナーとバイアスの関係を分析
        for bias_col, corner_col, _ in bias_corner_pairs:
            bias_cat_col = f'{bias_col}_cat'
            valid_data = course_df.dropna(subset=[bias_cat_col, corner_col, 'is_show'])
            
            if len(valid_data) < 5:
                print(f"  - {bias_col}と{corner_col}: 有効なデータが少なすぎるためスキップします（{len(valid_data)}件）")
                continue
            
            # バイアスカテゴリごとの平均通過位置と複勝率を算出
            bias_stats = valid_data.groupby(bias_cat_col).agg({
                corner_col: 'mean',
                'is_show': ['mean', 'count']
            }).reset_index()
            bias_stats.columns = [bias_cat_col, '平均通過位置', '複勝率', 'データ数']
            
            corner_results[f"{bias_col}_{corner_col}"] = bias_stats.to_dict('records')
            
            print(f"  - {bias_col}と{corner_col}の関係:")
            print(bias_stats)
            
            # グラフ作成（複勝率とコーナー通過位置の2軸グラフ）
            if len(bias_stats) > 1:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # 複勝率のバー
                bar_positions = np.arange(len(bias_stats))
                bars = ax1.bar(bar_positions, bias_stats['複勝率'], color='skyblue', alpha=0.7)
                ax1.set_ylabel('複勝率', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_ylim(0, 1)
                
                # 各バーに値を表示
                for i, (_, row) in enumerate(bias_stats.iterrows()):
                    ax1.text(i, row['複勝率'] + 0.05, f"{row['複勝率']:.2f}", 
                             ha='center', va='bottom', color='blue')
                    ax1.text(i, row['複勝率'] / 2, f"n={row['データ数']}", 
                             ha='center', va='center', color='white', fontweight='bold')
                
                # 平均通過位置の折れ線
                ax2 = ax1.twinx()
                ax2.plot(bar_positions, bias_stats['平均通過位置'], 'ro-', linewidth=2)
                ax2.set_ylabel('平均通過位置', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # 各ポイントに値を表示
                for i, (_, row) in enumerate(bias_stats.iterrows()):
                    ax2.text(i, row['平均通過位置'] + 0.2, f"{row['平均通過位置']:.1f}", 
                             ha='center', va='bottom', color='red')
                
                # グラフの装飾
                plt.title(f"{course_name} - {bias_col.replace('バイアス', '')}のバイアスと{corner_col}の関係")
                ax1.set_xticks(bar_positions)
                ax1.set_xticklabels(bias_stats[bias_cat_col])
                ax1.set_xlabel('バイアスカテゴリ')
                ax1.grid(axis='y', linestyle='--', alpha=0.3)
                
                # ファイル保存
                file_path = os.path.join(results_dir, f"{course_code}_{course_name}_{bias_col}_{corner_col}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        results[course_code] = corner_results
    
    return results

def create_summary_report(bias_results, corner_results):
    """分析結果のサマリーレポートを作成"""
    # サマリーテキストを作成
    summary_text = "# トラックバイアスと複勝率の分析レポート\n\n"
    summary_text += "## 1. バイアスカテゴリごとの複勝率分析\n\n"
    
    # 特に差が大きいカテゴリを抽出
    significant_findings = []
    
    for course_code, course_results in bias_results.items():
        course_name = race_course_codes.get(course_code, str(course_code))
        summary_text += f"### {course_name}（コード: {course_code}）\n\n"
        
        for bias_name, bias_data in course_results.items():
            summary_text += f"#### {bias_name}\n"
            
            if 'category_stats' in bias_data:
                overall_rate = bias_data['overall_show_rate']
                summary_text += f"全体の複勝率: {overall_rate:.2f}\n\n"
                
                summary_text += "| カテゴリ | 複勝率 | 件数 | 全体との差 |\n"
                summary_text += "|---------|--------|------|------------|\n"
                
                for stat in bias_data['category_stats']:
                    diff = stat['複勝率'] - overall_rate
                    summary_text += f"| {stat['カテゴリ']} | {stat['複勝率']:.2f} | {stat['件数']} | {diff:+.2f} |\n"
                    
                    # 顕著な差がある場合（差が0.1以上かつデータ数が5以上）
                    if abs(diff) >= 0.1 and stat['件数'] >= 5:
                        direction = "高い" if diff > 0 else "低い"
                        significant_findings.append({
                            '競馬場': course_name,
                            'バイアス': bias_name,
                            'カテゴリ': stat['カテゴリ'],
                            '全体との差': diff,
                            '複勝率': stat['複勝率'],
                            '件数': stat['件数'],
                            '説明': f"{course_name}の{bias_name}で{stat['カテゴリ']}の複勝率が全体より{abs(diff):.2f}ポイント{direction}"
                        })
            
            summary_text += "\n\n"
    
    # コーナー通過位置との関係
    summary_text += "## 2. バイアスとコーナー通過位置の関係\n\n"
    
    for course_code, course_results in corner_results.items():
        course_name = race_course_codes.get(course_code, str(course_code))
        summary_text += f"### {course_name}（コード: {course_code}）\n\n"
        
        for analysis_key, stats in course_results.items():
            bias_name, corner_name = analysis_key.split('_コーナー')
            summary_text += f"#### {bias_name}と{corner_name}の関係\n\n"
            
            summary_text += "| バイアスカテゴリ | 平均通過位置 | 複勝率 | データ数 |\n"
            summary_text += "|-----------------|--------------|--------|----------|\n"
            
            for stat in stats:
                cat_col = f"{bias_name}_cat"
                summary_text += f"| {stat[cat_col]} | {stat['平均通過位置']:.1f} | {stat['複勝率']:.2f} | {stat['データ数']} |\n"
            
            summary_text += "\n\n"
    
    # 特に注目すべき結果
    if significant_findings:
        summary_text += "## 3. 特に注目すべき結果\n\n"
        
        # 差の大きい順にソート
        significant_findings.sort(key=lambda x: abs(x['全体との差']), reverse=True)
        
        for i, finding in enumerate(significant_findings, 1):
            summary_text += f"{i}. {finding['説明']} (複勝率: {finding['複勝率']:.2f}, データ数: {finding['件数']})\n"
    
    # サマリーファイルに保存
    summary_path = os.path.join(results_dir, 'analysis_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\nサマリーレポートを作成しました: {summary_path}")
    
    # 注目すべき結果をデータフレームとしても保存
    if significant_findings:
        significant_df = pd.DataFrame(significant_findings)
        significant_df = significant_df.sort_values(by='全体との差', key=abs, ascending=False)
        significant_csv_path = os.path.join(results_dir, 'significant_findings.csv')
        significant_df.to_csv(significant_csv_path, index=False, encoding='utf-8')
        print(f"注目すべき結果をCSVに保存しました: {significant_csv_path}")

# 場コードのマッピング（グローバル変数として定義）
race_course_codes = {
    1: '札幌', 2: '函館', 3: '福島', 4: '新潟', 
    5: '東京', 6: '中山', 7: '中京', 8: '京都', 
    9: '阪神', 10: '小倉'
}

def main():
    # データ読み込み
    file_path = 'export/with_bias/SED250427_formatted_with_bias.csv'
    df = load_data(file_path)
    
    print("\n=== 1. バイアスカテゴリごとの複勝率分析 ===")
    bias_results = analyze_bias_categories(df)
    
    print("\n=== 2. コーナー通過位置との関係分析 ===")
    corner_results = analyze_corner_positions(df)
    
    print("\n=== 3. サマリーレポート作成 ===")
    create_summary_report(bias_results, corner_results)
    
    print("\n===== 分析完了 =====")
    print(f"結果は {results_dir} ディレクトリに保存されました")

if __name__ == "__main__":
    main() 
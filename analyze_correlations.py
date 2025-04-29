import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import argparse
from tqdm import tqdm

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def load_and_process_data(input_path):
    """
    CSVファイルを読み込んで処理します。
    """
    input_path = Path(input_path)
    print("データの読み込みを開始します...")
    
    # 必要な列のみを読み込む
    usecols = ['距離', '着順', '馬番', '人気', '斤量', '単勝オッズ', 
               '複勝オッズ', '上り', '芝ダ障害コード', '天候', 
               '馬場状態', '性別']
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path, encoding="utf-8-sig", usecols=lambda x: x in usecols)
            print(f"ファイル {input_path.name} を読み込みました。レコード数: {len(df)}")
    else:
        csv_files = list(input_path.glob("SEC*.csv"))
        print(f"SECファイル数: {len(csv_files)}")
        
        # データフレームのリストを作成
        dfs = []
        for file in tqdm(csv_files, desc="ファイル処理中"):
            try:
                temp_df = pd.read_csv(file, encoding="utf-8-sig", usecols=lambda x: x in usecols)
                dfs.append(temp_df)
            except Exception as e:
                print(f"警告: {file.name} の処理中にエラー: {str(e)}")
        
        if not dfs:
            raise ValueError("有効なデータを含むファイルが見つかりませんでした。")
        
        print("データの結合を開始します...")
        df = pd.concat(dfs, ignore_index=True)
        print(f"合計レコード数: {len(df)}")
    
    print("データの前処理を開始します...")
    
    # 数値データの変換（高速化のため一括で処理）
    numeric_columns = ['距離', '着順', '馬番', '人気', '斤量', '単勝オッズ', '複勝オッズ', '上り']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # 勝利フラグの作成
    df['is_win'] = (df['着順'] == 1).astype(int)
    
    # メモリ使用量の最適化
    df = df.astype({
        'is_win': 'int8',
        '馬番': 'int16',
        '人気': 'int16',
        '距離': 'int16',
        '斤量': 'float32'
    })
    
    print("データの前処理が完了しました。")
    return df

def create_correlation_heatmap(df, output_dir):
    """
    相関分析とヒートマップを作成します。
    """
    print("相関分析を開始します...")
    
    # 分析対象の列を選択
    target_columns = [
        '距離', '着順', '馬番', '人気', '斤量', '単勝オッズ', 
        '複勝オッズ', '上り', 'is_win'
    ]
    
    # 利用可能な列のみを使用
    available_columns = [col for col in target_columns if col in df.columns]
    print(f"分析対象の列: {available_columns}")
    
    # 相関行列の計算
    correlation_matrix = df[available_columns].corr()
    
    print("ヒートマップを作成中...")
    # ヒートマップの作成
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': '相関係数'})
    
    plt.title('レース要素間の相関係数ヒートマップ')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'is_win' in available_columns:
        print("勝率との相関を分析中...")
        win_correlations = correlation_matrix['is_win'].sort_values(ascending=False)
        
        plt.figure(figsize=(15, 8))
        win_correlations.plot(kind='bar')
        plt.title('各要素と勝率の相関')
        plt.xlabel('レース要素')
        plt.ylabel('勝率との相関係数')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'win_correlation_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        win_correlations.to_frame('相関係数').to_csv(
            output_dir / 'win_correlations.csv',
            encoding='utf-8-sig'
        )

def analyze_categorical_correlations(df, output_dir):
    """
    カテゴリカル変数と勝率の関係を分析します。
    """
    print("カテゴリカル変数の分析を開始します...")
    categorical_columns = ['芝ダ障害コード', '天候', '馬場状態', '性別']
    
    for col in tqdm(categorical_columns, desc="カテゴリ分析中"):
        if col in df.columns:
            # カテゴリ別の勝率を計算
            win_rates = df.groupby(col)['is_win'].agg(['mean', 'count']).reset_index()
            win_rates.columns = [col, '勝率', 'レース数']
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=win_rates, x=col, y='勝率')
            plt.title(f'{col}別の勝率')
            plt.xlabel(col)
            plt.ylabel('勝率')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'{col}_win_rates.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            win_rates.to_csv(
                output_dir / f'{col}_win_rates.csv',
                encoding='utf-8-sig',
                index=False
            )

def main():
    parser = argparse.ArgumentParser(description='レース要素間の相関分析を行います。')
    parser.add_argument('input_path', help='入力CSVファイルのパス、またはSECファイルを含むディレクトリのパス')
    parser.add_argument('--output-dir', '-o', default='export/correlations',
                      help='出力ディレクトリのパス（デフォルト: export/correlations）')
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # データの読み込みと処理
        df = load_and_process_data(args.input_path)
        
        # 相関分析とヒートマップの作成
        create_correlation_heatmap(df, output_dir)
        
        # カテゴリカル変数の分析
        analyze_categorical_correlations(df, output_dir)
        
        print(f"\n分析が完了しました。結果は {output_dir} に保存されています。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
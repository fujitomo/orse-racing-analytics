import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='馬ごとに出現した脚質コード一覧を出力')
    parser.add_argument('input_path', help='入力CSVファイルまたはディレクトリのパス')
    parser.add_argument('--output', default='export/analysis/trackbias/horse_running_style_list.csv', help='出力CSVファイルのパス')
    args = parser.parse_args()

    # データ読み込み
    if os.path.isdir(args.input_path):
        csv_files = sorted([f for f in os.listdir(args.input_path) if f.endswith('.csv')])
        if not csv_files:
            raise ValueError(f"{args.input_path} にCSVファイルが見つかりません")
        df_list = []
        for file in csv_files:
            file_path = os.path.join(args.input_path, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df_list.append(df)
            except Exception as e:
                print(f"警告: {file} の読み込みに失敗: {e}")
        if not df_list:
            raise ValueError("有効なCSVファイルが見つかりませんでした。")
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(args.input_path, encoding='utf-8')

    # レース脚質カラムの特定
    if 'レース脚質' not in df.columns:
        raise ValueError('レース脚質カラムが見つかりません')

    # 馬ごとに出現した脚質コード一覧を集計（すべてstr型に変換してからset・sort）
    horse_style_list = (
        df.groupby('馬名')['レース脚質']
          .apply(lambda x: sorted(set(str(v) for v in x)))
          .reset_index()
    )
    horse_style_list = horse_style_list.rename(columns={'レース脚質': '脚質コード一覧'})

    # 出力
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    horse_style_list.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'出力完了: {args.output}')

if __name__ == '__main__':
    main() 
import pandas as pd
from pathlib import Path

def check_columns():
    # SECデータの最初のファイルを読み込む
    sec_files = list(Path("export/SEC").glob("*.csv"))
    if not sec_files:
        print("SECファイルが見つかりません。")
        return
    
    first_file = sec_files[0]
    print(f"ファイル: {first_file}")
    
    # 列名を表示
    df = pd.read_csv(first_file, encoding="utf-8")
    print("\n列名一覧:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    check_columns() 
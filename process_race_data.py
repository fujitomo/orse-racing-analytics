"""
レースデータ処理のコマンドラインエントリーポイント
"""

import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from horse_racing.data.processors.bac_processor import process_all_bac_files
from horse_racing.data.processors.sed_processor import process_all_sed_files

def process_race_data():
    """
    レースデータを以下の順序で処理します：
    1. BACデータ（レース基本情報）
    2. SEDデータ（競走成績）
    """
    print("レースデータの処理を開始します。")
    
    # 1. BACデータの処理
    print("\n=== BACデータ（レース基本情報）の処理を開始 ===")
    process_all_bac_files()
    print("=== BACデータの処理が完了しました ===")
    
    # 2. SEDデータの処理
    print("\n=== SEDデータ（競走成績）の処理を開始 ===")
    process_all_sed_files()
    print("=== SEDデータの処理が完了しました ===")
    
    print("\nすべてのデータ処理が完了しました。")

if __name__ == "__main__":
    process_race_data() 
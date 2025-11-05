"""
レースデータ（BAC、SED）の一括処理スクリプト
"""

from .processors.bac_processor import process_all_bac_files
from .processors.sed_processor import process_all_sed_files

def process_race_data():
    """レースデータの処理フローを実行します。

    Steps:
        1. BAC データ（レース基本情報）の処理。
        2. SED データ（競走成績）の処理。
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
"""
レースデータ処理のコマンドラインエントリーポイント
"""
from horse_racing.data.processors.bac_processor import process_all_bac_files
from horse_racing.data.processors.sed_processor import process_all_sed_files
from horse_racing.data.processors.srb_processor import process_all_srb_files, merge_srb_with_sed
import os
import argparse
from output_utils import OutputUtils

def ensure_export_dirs():
    """
    出力用ディレクトリの存在確認と作成
    """
    dirs = ['export/BAC', 'export/SRB', 'export/SED', 'export/SED_SRB']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"ディレクトリを作成しました: {dir_path}")

def process_race_data(exclude_turf=False, turf_only=False):
    """
    レースデータを以下の順序で処理します：
    1. BACデータ（レース基本情報）
    2. SRBデータ（レース詳細情報）
    3. SEDデータ（競走成績）とSRB・BACデータの紐づけ
    4. SEDデータとSRBデータの紐づけ（すべてのバイアス情報が揃ったデータのみ保持）
    
    Args:
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
    """
    print("レースデータの処理を開始します。")
    if exclude_turf and turf_only:
        print("エラー: 芝コースを除外するオプションと芝コースのみを処理するオプションは同時に指定できません。")
        return
    
    if exclude_turf:
        print("※ 芝コースのデータは除外されます")
    elif turf_only:
        print("※ 芝コースのデータのみが処理されます")
    
    # 出力用ディレクトリの確認
    ensure_export_dirs()
    
    # 1. BACデータの処理
    print("\n=== BACデータ（レース基本情報）の処理を開始 ===")
    process_all_bac_files(exclude_turf=exclude_turf, turf_only=turf_only)
    print("=== BACデータの処理が完了しました ===")
    
    # 2. SRBデータの処理
    print("\n=== SRBデータ（レース詳細情報）の処理を開始 ===")
    process_all_srb_files(exclude_turf=exclude_turf, turf_only=turf_only)
    print("=== SRBデータの処理が完了しました ===")
    
    # 3. SEDデータの処理とSRB・BACデータとの紐づけ
    print("\n=== SEDデータ（競走成績）の処理と紐づけを開始 ===")
    process_all_sed_files(exclude_turf=exclude_turf, turf_only=turf_only)
    
    # SEDデータとSRBデータの紐づけ（別々のフォルダに出力）
    print("\n=== SEDデータとSRBデータの紐づけを開始 ===")
    print("※ １角バイアス、２角バイアス、向正バイアス、３角バイアス、４角バイアス、直線バイアス、レースコメントがすべて揃ったデータのみを保持します")
    merge_result = merge_srb_with_sed(separate_output=True, exclude_turf=exclude_turf, turf_only=turf_only)  # separate_output=Trueを追加
    if merge_result:
        print("=== SEDデータとSRBデータの紐づけが完了しました ===")
        print("SEDデータは export/SED/ に出力されました")
        print("SRBデータは export/SRB/ に出力されました")
        print("バイアス情報付きのSEDデータは export/SED_SRB/ に出力されました")
        if exclude_turf:
            print("※ 芝コースのデータは除外されています")
        elif turf_only:
            print("※ 芝コースのデータのみが処理されています")
    else:
        print("=== SEDデータとSRBデータの紐づけに失敗しました ===")
    
    print("\nすべてのデータ処理が完了しました。")

if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='レースデータを処理します')
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', action='store_true', help='芝コースのデータを除外する')
    track_group.add_argument('--turf-only', action='store_true', help='芝コースのデータのみを処理する')
    args = parser.parse_args()
    
    # レースデータ処理の実行
    process_race_data(exclude_turf=args.exclude_turf, turf_only=args.turf_only) 
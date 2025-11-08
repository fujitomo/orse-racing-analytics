"""
バージョンアップ後に使用予定
SRBデータ（レース詳細情報）の処理モジュール
"""

import csv
import os
import glob
import pandas as pd
from pathlib import Path
from ..constants.jra_masters import JRA_MASTERS
from .utils import convert_year_to_4digits
import numpy as np

def add_grade_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    数値グレードから「グレード名」列をグレード列の直後に追加する
    
    Args:
        df: 処理対象DataFrame
        
    Returns:
        グレード名列が追加されたDataFrame
    """
    # グレード変換マッピング
    grade_mapping = {
        1: 'Ｇ１',
        2: 'Ｇ２', 
        3: 'Ｇ３',
        4: '重賞',
        5: '特別',
        6: 'Ｌ　（リステッド競走）'
    }
    
    # グレード列が存在するかチェック
    if 'グレード' not in df.columns:
        return df
    
    # グレード列を数値型として保持
    df['グレード'] = pd.to_numeric(df['グレード'], errors='coerce')
    
    # NaN値がある場合はデフォルト値（5: 特別）を設定
    df['グレード'] = df['グレード'].fillna(5)
    
    # グレード名データを作成
    grade_names = df['グレード'].map(grade_mapping).fillna('特別')
    
    # グレード名列が既に存在するかチェック
    if 'グレード名' in df.columns:
        # 既存の列を更新
        df['グレード名'] = grade_names
    else:
        # グレード列の直後に「グレード名」列を挿入
        grade_col_index = df.columns.get_loc('グレード')
        df.insert(grade_col_index + 1, 'グレード名', grade_names)
    
    return df

def process_srb_record(record, index):
    """
    SRBレコードを処理します
    
    Args:
        record (bytes): バイナリレコード
        index (int): レコードのインデックス
        
    Returns:
        dict: 処理されたフィールドの辞書
    """
    try:
        # 信頼性の高いフィールドから抽出（エラー時は例外を発生させる）
        場コード = record[0:2].decode("shift_jis", errors="strict").strip()
        year_2digit = record[2:4].decode("shift_jis", errors="strict").strip()
        year_4digit = convert_year_to_4digits(year_2digit, index)
        回 = record[4:5].decode("shift_jis", errors="strict").strip()
        日 = record[5:6].decode("shift_jis", errors="strict").strip()
        Ｒ = record[6:8].decode("shift_jis", errors="strict").strip()

        # 数字でない場合はデフォルト値を使用
        if not 回.isdigit(): 回 = "0"
        if not 日.isdigit(): 日 = "0"
        if not Ｒ.isdigit(): Ｒ = "00"

        # レースIDを作成
        レースID = 場コード + year_4digit + 回 + 日 + Ｒ
        
        # 場名を取得
        場名 = JRA_MASTERS["場コード"].get(場コード, "")
        
        # 結果辞書を初期化
        result = {
            "場コード": 場コード, "場名": 場名, "年": year_4digit, "回": 回,
            "日": 日, "Ｒ": Ｒ, "レースID": レースID,
            "１角バイアス": "", "２角バイアス": "", "向正バイアス": "", "３角バイアス": "",
            "４角バイアス": "", "直線バイアス": "", "レースコメント": ""
        }
        
        import re
        # ヘッダー以降のテキストをデコード
        full_text = record[8:].decode("shift_jis", errors="replace").strip()

        # バイアス情報とレースコメントを抽出
        # バイアス情報は括弧で始まり、その後ろにコメントが続く場合がある
        match = re.search(r'^(?P<bias>\(.*\)).*?(?P<comment>\s{2,}.*)?$', full_text)
        
        bias_info = ""
        race_comment = ""

        if match:
            bias_info = match.group('bias').strip()
            if match.group('comment'):
                race_comment = match.group('comment').strip()

        if bias_info:
            # 各コーナーのバイアス情報を抽出
            bracket_parts = re.findall(r'\(([^)]*)\)', bias_info)
            
            result["１角バイアス"] = bracket_parts[0] if len(bracket_parts) > 0 else ""
            result["２角バイアス"] = bracket_parts[1] if len(bracket_parts) > 1 else ""
            result["３角バイアス"] = bracket_parts[2] if len(bracket_parts) > 2 else ""
            result["４角バイアス"] = bracket_parts[3] if len(bracket_parts) > 3 else ""
            
            # 直線バイアス（括弧の外側）
            result["直線バイアス"] = re.sub(r'\([^)]*\)', '', bias_info).strip()
            result["レースコメント"] = race_comment

        return result

    except (UnicodeDecodeError, IndexError, ValueError) as e:
        # デコードエラーやインデックスエラーは不正なレコードとして処理
        # print(f"レコード処理中にエラーが発生: {str(e)}") # ログが冗長なためコメントアウト
        return {
            "場コード": "", "場名": "", "年": "", "回": "", "日": "", "Ｒ": "", "レースID": "",
            "１角バイアス": "", "２角バイアス": "", "向正バイアス": "", "３角バイアス": "",
            "４角バイアス": "", "直線バイアス": "", "レースコメント": ""
        }

def format_srb_file(input_file, output_file):
    """
    SRBファイルを整形してCSVに変換します。
    
    Args:
        input_file (str): 入力ファイルのパス
        output_file (str): 出力ファイルのパス
    """
    record_length = 852  # 固定長レコードのバイト数

    # バイナリモードで読み込む（改行コードを無視）
    with open(input_file, "rb") as infile:
        content = infile.read()

    # 852バイトごとに分割
    records = [content[i:i + record_length] for i in range(0, len(content), record_length)]

    # 仕様書に従い、各レコードを正しく分割
    processed_records = []
    for i, record in enumerate(records):
        if len(record) != record_length:
            continue  # 852バイトに満たないレコードは無視
            
        processed_record = process_srb_record(record, i)
        if processed_record:
            processed_records.append(processed_record)

    # CSVに書き出し（UTF-8で出力）
    if processed_records:
        # 必須の基本フィールドを定義
        base_fields = [
            "レースID", "場コード", "場名", "年", "回", "日", "Ｒ", 
            "１角バイアス", "２角バイアス", "向正バイアス", "３角バイアス", 
            "４角バイアス", "直線バイアス", "レースコメント"
        ]
        
        # 全レコードから使用されている全フィールドを収集
        all_fields = set()
        for record in processed_records:
            all_fields.update(record.keys())
        
        # ベースフィールドを優先し、残りをアルファベット順
        other_fields = sorted(list(all_fields - set(base_fields)))
        field_names = base_fields + other_fields
        
        # 各レコードに不足フィールドを追加
        for record in processed_records:
            for field in field_names:
                if field not in record:
                    record[field] = ""
        
        # CSVに書き出し
        with open(output_file, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(processed_records)

        print(f"✅ 整形されたファイルが {output_file} に保存されました。")
        return processed_records
    else:
        print(f"⚠️ {input_file} からの有効なレコードがありません。")
        return []

def process_all_srb_files(exclude_turf=False, turf_only=False):
    """
    すべてのSRBファイルを処理します。
    
    Args:
        exclude_turf (bool): 芝コースを除外するかどうか。
            SRBデータ自体には芝・ダートの情報がないため、
            BACデータ・SEDデータとの紐づけ時に除外します。
        turf_only (bool): 芝コースのみを処理するかどうか。
            SRBデータ自体には芝・ダートの情報がないため、
            BACデータ・SEDデータとの紐づけ時に絞り込みます。
    """
    # importフォルダとexportフォルダのパスを設定
    import_dir = Path("import/SRB")
    export_dir = Path("export/SRB")
    formatted_dir = Path("export/SRB/formatted")  # formattedファイル用のディレクトリ
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(exist_ok=True, parents=True)
    formatted_dir.mkdir(exist_ok=True, parents=True)  # formattedディレクトリを作成
    
    # importフォルダ内のすべてのSRBファイルを処理
    all_processed = []
    for srb_file in import_dir.glob("SRB*.txt"):
        # formattedファイルは別のフォルダに出力
        output_file = formatted_dir / f"{srb_file.stem}_formatted.csv"
        print(f"処理中: {srb_file}")
        processed = format_srb_file(str(srb_file), str(output_file))
        all_processed.extend(processed)
    
    # 全レコードを1つのCSVにまとめる（formattedディレクトリに出力）
    if all_processed:
        # 必須の基本フィールドを定義
        base_fields = [
            "レースID", "場コード", "場名", "年", "回", "日", "Ｒ", 
            "１角バイアス", "２角バイアス", "向正バイアス", "３角バイアス", 
            "４角バイアス", "直線バイアス", "レースコメント"
        ]
        
        # 全レコードから使用されている全フィールドを収集
        all_fields = set()
        for record in all_processed:
            all_fields.update(record.keys())
        
        # ベースフィールドを優先し、残りをアルファベット順
        other_fields = sorted(list(all_fields - set(base_fields)))
        field_names = base_fields + other_fields
        
        # 各レコードに不足フィールドを追加
        for record in all_processed:
            for field in field_names:
                if field not in record:
                    record[field] = ""
        
        # CSVに書き出し
        with open(formatted_dir / "SRB_ALL_formatted.csv", "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(all_processed)
        print(f"✅ すべてのSRBデータが {formatted_dir}/SRB_ALL_formatted.csv に統合されました。")
    
    return all_processed

def merge_srb_with_sed(separate_output=False, exclude_turf=False, turf_only=False):
    """
    SEDデータとBACデータを紐づけます。(SRBデータは結合しません)
    指定されたキー（場コード, 回, 年月日, R）で紐づけを行い、
    BACの賞金情報等をSEDデータに追加します。
    
    Args:
        separate_output (bool): この引数は現在使用されていません。
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
    """
    # エクスポートディレクトリのパス
    sed_formatted_dir = Path("export/SED/formatted")
    dataset_dir = Path("export/dataset")
    bac_formatted_dir = Path("export/BAC/formatted")

    dataset_dir.mkdir(exist_ok=True, parents=True)

    # --- BACデータの読み込みと集約 ---
    try:
        bac_files = list(bac_formatted_dir.glob("*.csv"))
        if not bac_files:
            print("⚠️ BACフォーマット済みファイルが見つかりません。")
            return False
        
        # 結合に必要な列のみを読み込む
        bac_cols = [
            '場コード', '回', '年月日', 'R', '1着賞金', '2着賞金', '3着賞金', '4着賞金', '5着賞金', 'グレード', 'レース名', '頭数',
            '1着賞金(1着算入賞金込み)', '2着賞金(2着算入賞金込み)', '平均賞金'
        ]
        bac_df_list = []
        for f in bac_files:
            try:
                # `usecols`で列が存在しない場合のエラーをハンドリング
                df = pd.read_csv(f, dtype=str)
                cols_to_use = [col for col in bac_cols if col in df.columns]
                bac_df_list.append(df[cols_to_use])
            except Exception as e:
                print(f"⚠️ BACファイル {f.name} の読み込み中にエラー: {e}")
                continue

        if not bac_df_list:
            print("⚠️ 有効なBACデータが読み込めませんでした。")
            return False

        bac_df = pd.concat(bac_df_list, ignore_index=True)
        
        # データ型を適切に変換
        bac_df['1着賞金'] = pd.to_numeric(bac_df['1着賞金'], errors='coerce').fillna(0)
        bac_df['2着賞金'] = pd.to_numeric(bac_df['2着賞金'], errors='coerce').fillna(0)
        bac_df['3着賞金'] = pd.to_numeric(bac_df['3着賞金'], errors='coerce').fillna(0)
        bac_df['4着賞金'] = pd.to_numeric(bac_df['4着賞金'], errors='coerce').fillna(0)
        bac_df['5着賞金'] = pd.to_numeric(bac_df['5着賞金'], errors='coerce').fillna(0)
        bac_df['1着賞金(1着算入賞金込み)'] = pd.to_numeric(bac_df['1着賞金(1着算入賞金込み)'], errors='coerce').fillna(0)
        bac_df['2着賞金(2着算入賞金込み)'] = pd.to_numeric(bac_df['2着賞金(2着算入賞金込み)'], errors='coerce').fillna(0)
        bac_df['平均賞金'] = pd.to_numeric(bac_df['平均賞金'], errors='coerce').fillna(0)
        
        # 結合キー
        join_keys = ['場コード', '回', '年月日', 'R']

        # 結合キーがすべて存在することを確認
        if not all(key in bac_df.columns for key in join_keys):
            print(f"⚠️ BACデータに結合キー {join_keys} が揃っていません。")
            return False

        # キーで重複を排除（1着賞金が最大のレコードを残す）
        bac_agg_df = bac_df.sort_values('1着賞金', ascending=False).drop_duplicates(subset=join_keys)
        
        print(f"✅ BACデータを読み込み、集約しました。（{len(bac_agg_df)}ユニークレース）")
    except Exception as e:
        print(f"⚠️ BACデータの読み込みまたは集約に失敗しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # --- SEDフォーマット済みファイルの処理 ---
    processed_count = 0
    sed_files = list(sed_formatted_dir.glob("SED*_formatted.csv"))
    print(f"\n処理対象のSEDファイル数: {len(sed_files)}")
    
    for sed_file in sed_files:
        try:
            sed_df = pd.read_csv(sed_file, encoding="utf-8", dtype=str)
            print(f"\n処理中: {sed_file.name}（{len(sed_df)}レコード）")
            
            # BAC由来の賞金と重複するため、SED側の賞金カラムを削除
            prize_cols_to_drop = ['本賞金', '収得賞金', '1着賞金']
            sed_df = sed_df.drop(columns=[col for col in prize_cols_to_drop if col in sed_df.columns])
            
            # (芝・ダートのフィルタリング処理は変更なし)
            if exclude_turf and '芝ダ障害コード' in sed_df.columns:
                sed_df = sed_df[sed_df['芝ダ障害コード'] != '芝']
            if turf_only and '芝ダ障害コード' in sed_df.columns:
                sed_df = sed_df[sed_df['芝ダ障害コード'] == '芝']

            # SED側の結合キーの存在確認と作成
            # '年'が2桁の場合のみ4桁に変換
            if sed_df['年'].str.len().max() == 2:
                sed_df['年'] = sed_df['年'].apply(lambda x: convert_year_to_4digits(x, sed_file.name))
            
            # 'R'を2桁ゼロ埋め
            if 'R' in sed_df.columns:
                sed_df['R'] = sed_df['R'].str.zfill(2)

            if not all(key in sed_df.columns for key in join_keys):
                print(f"  ⚠️ SEDファイル {sed_file.name} に結合キー {join_keys} が揃っていません。スキップします。")
                continue

            # データ結合: SEDデータに集約済みBACデータを紐づける
            merged_df = pd.merge(
                sed_df,
                bac_agg_df,
                on=join_keys,
                how='left'
            )
            
            # 欠損値処理の実行（グレード推定処理含む）
            from .missing_value_handler import MissingValueHandler
            missing_handler = MissingValueHandler()
            final_df = missing_handler.handle_missing_values(merged_df)

            # 結果の保存
            output_file = dataset_dir / f"{sed_file.stem}_dataset.csv"
            final_df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"  ✓ データを {output_file} に保存しました。")
            
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ {sed_file.name} の処理中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # --- 処理結果の報告 ---
    if processed_count > 0:
        print(f"\n✅ 合計 {processed_count} 件のSEDファイルにBAC情報を追加しました。")
        return True
    else:
        print("\n⚠️ SEDファイルの処理に失敗しました。")
        return False
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SRBファイルを処理します')
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', action='store_true', help='芝コースを除外する')
    track_group.add_argument('--turf-only', action='store_true', help='芝コースのみを処理する')
    args = parser.parse_args()
    
    # process_all_srb_filesはturf_onlyを使わないが、一貫性のために引数として渡す
    process_all_srb_files(exclude_turf=args.exclude_turf, turf_only=args.turf_only)
    
    # SEDデータとSRBデータの紐づけ
    merge_srb_with_sed()

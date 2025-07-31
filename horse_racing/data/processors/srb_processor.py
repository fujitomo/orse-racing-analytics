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
    SRBデータとSEDデータを紐づけます。
    レースIDをキーにして紐づけを行い、SRBのトラックバイアス情報をSEDデータに追加します。
    すべてのバイアス情報（１角バイアス、２角バイアス、向正バイアス、３角バイアス、４角バイアス、直線バイアス、レースコメント）に
    値がある場合のみデータを保持します。
    
    Args:
        separate_output (bool): Trueの場合、SEDデータとSRBデータを別々のフォルダに出力します。
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
    """
    # エクスポートディレクトリのパス
    srb_export_dir = Path("export/SRB")
    sed_export_dir = Path("export/SED")
    sed_formatted_dir = Path("export/SED/formatted")  # formattedファイル用のディレクトリ
    merged_dir = Path("export/SED_SRB") if not separate_output else None
    with_bias_dir = Path("export/with_bias")  # _with_biasファイル用のディレクトリ
    
    # マージディレクトリが存在しない場合は作成
    if not separate_output and merged_dir:
        merged_dir.mkdir(exist_ok=True, parents=True)
    with_bias_dir.mkdir(exist_ok=True, parents=True)  # with_biasディレクトリを作成
    
    # SRB統合ファイルの読み込み
    srb_all_file = srb_export_dir / "formatted/SRB_ALL_formatted.csv"
    if not srb_all_file.exists():
        print("⚠️ SRB統合ファイルが見つかりません。先にSRBファイルの処理を実行してください。")
        return False
        
    try:
        srb_df = pd.read_csv(srb_all_file, encoding="utf-8", dtype=str)
        print(f"✅ SRB統合ファイルを読み込みました。（{len(srb_df)}レコード）")
        
        # SRBデータの全カラムを文字列型に変換
        for col in srb_df.columns:
            srb_df[col] = srb_df[col].astype(str)
        
        # SRBデータのレースIDを使用する
        print(f"SRBデータのレースIDサンプル: {srb_df['レースID'].head(5).tolist()}")
    except Exception as e:
        print(f"⚠️ SRB統合ファイルの読み込みに失敗しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 必要なカラムのみ抽出
    bias_columns = ['レースID', '１角バイアス', '２角バイアス', '向正バイアス', 
                    '３角バイアス', '４角バイアス', '直線バイアス', 'レースコメント']
    srb_bias_df = srb_df[bias_columns].copy()
    
    # SRBのバイアス情報を確認（デバッグ用）
    srb_sample = srb_bias_df.head(5)
    print("SRBバイアス情報サンプル:")
    for _, row in srb_sample.iterrows():
        print(f"レースID: {row['レースID']}, １角: {row['１角バイアス']}, ２角: {row['２角バイアス']}")
    
    # 重複を排除（同じレースIDのデータが複数ある場合）
    srb_bias_df = srb_bias_df.drop_duplicates(subset=['レースID'])
    print(f"重複排除後のSRBデータ数: {len(srb_bias_df)}")
    
    # SEDフォーマット済みファイルの処理
    processed_count = 0
    matched_race_count = 0
    sed_files = list(sed_formatted_dir.glob("SED*_formatted.csv"))  # formattedディレクトリから検索
    print(f"\n処理対象のSEDファイル数: {len(sed_files)}")
    print("処理対象ファイル:")
    for file in sed_files:
        print(f"- {file.name}")
    
    for sed_file in sed_files:
        try:
            # SEDファイルの読み込み
            sed_df = pd.read_csv(sed_file, encoding="utf-8", dtype=str)
            print(f"\n処理中: {sed_file.name}（{len(sed_df)}レコード）")
            
            # 芝コースを除外する場合
            if exclude_turf and '芝ダ障害コード' in sed_df.columns:
                original_len = len(sed_df)
                sed_df = sed_df[sed_df['芝ダ障害コード'] != '芝']
                excluded = original_len - len(sed_df)
                print(f"芝コースを除外: {excluded}件のレコードを除外しました（{len(sed_df)}件残り）")
                
            # 芝コースのみを処理する場合
            if turf_only and '芝ダ障害コード' in sed_df.columns:
                original_len = len(sed_df)
                sed_df = sed_df[sed_df['芝ダ障害コード'] == '芝']
                excluded = original_len - len(sed_df)
                print(f"芝コース以外を除外: {excluded}件のレコードを除外しました（{len(sed_df)}件残り）")
            
            # デバッグ情報：SEDデータのカラムを表示
            print("\nSEDデータのカラム:")
            print(sed_df.columns.tolist())
            
            # 全カラムを文字列型に変換
            for col in sed_df.columns:
                sed_df[col] = sed_df[col].astype(str)
            
            # SEDデータにレースIDを作成（紐づけ用）
            # 場コード + 年 + 回 + 日 + R の形式
            sed_df['レースID'] = sed_df['場コード'] + sed_df['年'] + sed_df['回'] + sed_df['日'] + sed_df['R']
            
            # デバッグ情報：SEDデータのサンプルを表示
            print("\nSEDデータのサンプル（最初の5レコード）:")
            print(sed_df[['レースID', '馬名', '着順', '場コード', '年', '回', '日', 'R']].head())
            
            # デバッグ情報：SRBデータのサンプルを表示
            print("\nSRBデータのサンプル（最初の5レコード）:")
            print(srb_bias_df[['レースID', '１角バイアス', '２角バイアス']].head())
            
            # データ結合: 左外部結合（SEDデータにSRBデータを紐づける）
            merged_df = pd.merge(
                sed_df,
                srb_bias_df,
                on='レースID',
                how='left'
            )
            
            # バイアス情報がすべて揃っているデータのみを抽出
            bias_columns = ['１角バイアス', '２角バイアス', '向正バイアス', '３角バイアス', '４角バイアス', '直線バイアス', 'レースコメント']
            
            # 各カラムについて、NaN値または空文字列でないかをチェック
            # すべての列がNaNでないAND空文字列でないレコードを選択する
            has_all_bias_info = merged_df[bias_columns].apply(lambda x: ~x.isna() & (x.str.strip() != ''), axis=0).all(axis=1)
            
            # バイアス情報詳細のデバッグ出力
            print("\n詳細なバイアス情報チェック:")
            sample_indices = merged_df.sample(min(5, len(merged_df))).index
            for idx in sample_indices:
                row = merged_df.loc[idx]
                print(f"レースID: {row['レースID']}, レース名: {row.get('レース名', 'N/A')}")
                for col in bias_columns:
                    value = row[col]
                    is_na = pd.isna(value)
                    is_empty = False if is_na else value.strip() == ''
                    status = "❌ NaN" if is_na else ("❌ 空文字" if is_empty else "✓ 有効")
                    print(f"  {col}: [{status}] {value if not is_na else 'NaN'}")
                print(f"  結果: {'✓ 保持' if not is_na and not is_empty else '❌ 除外'}")
                print("  " + "-" * 50)
            
            # バイアス情報がすべて揃っているデータのみを抽出
            filtered_df = merged_df[has_all_bias_info]
            
            if len(filtered_df) == 0:
                print(f"  ⚠️ {sed_file.name} のすべてのレコードでバイアス情報のすべての項目が揃っていないため、出力をスキップします。")
                continue
                
            print(f"  ✓ すべてのバイアス情報が揃っているデータ: {len(filtered_df)} レコード（{len(merged_df) - len(filtered_df)} レコードを除外）")
            
            # デバッグ情報：バイアス情報の状態を表示
            print("\nバイアス情報の状態:")
            for col in bias_columns:
                non_empty = filtered_df[col].notna().sum()
                non_empty_str = (filtered_df[col].str.strip() != '').sum()
                print(f"  {col}: 非NaN値={non_empty}, 非空文字列={non_empty_str}")
            
            # マッチしたレース数をカウント
            matched_races = len(filtered_df) > 0
            if matched_races:
                matched_count = len(filtered_df)
                matched_race_count += len(filtered_df['レースID'].unique())
                print(f"  ✓ {matched_count} レコードのトラックバイアス情報が完全に紐づきました。")
            else:
                print("  ⚠️ すべてのバイアス情報が揃ったレコードがありませんでした。")
                continue  # すべてのバイアス情報が揃っていない場合はスキップ
            
            # 欠損値処理の実行（グレード推定処理含む）
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from process_race_data import MissingValueHandler
            missing_handler = MissingValueHandler()
            filtered_df = missing_handler.handle_missing_values(filtered_df)
            
            # グレード名列の追加（既に追加済みの場合はスキップ）
            if 'グレード名' not in filtered_df.columns:
                filtered_df = add_grade_name_column(filtered_df)
            
            # 結果の保存
            if separate_output:
                # すべてのファイルをwith_biasディレクトリに保存
                output_file = with_bias_dir / f"{sed_file.stem}_with_bias.csv"
                
                # 結合したデータを保存（フィルタリング済みのデータを使用）
                filtered_df.to_csv(output_file, index=False, encoding="utf-8")
                print(f"  ✓ データを {output_file} に保存しました。")
                
            else:
                # 統合データを保存
                output_file = with_bias_dir / f"{sed_file.stem}_with_bias.csv"
                filtered_df.to_csv(output_file, index=False, encoding="utf-8")
                print(f"  ✓ 結果を {output_file} に保存しました。")
            
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ {sed_file.name} の処理中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 処理結果の報告
    if processed_count > 0:
        print(f"\n✅ 合計 {processed_count} 件のSEDファイルにトラックバイアス情報を追加しました。")
        print(f"✅ 合計 {matched_race_count} レースのバイアス情報をマッチングしました。")
        if separate_output:
            print("✅ SEDデータとSRBデータを別々のフォルダに出力しました。")
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
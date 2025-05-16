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
        # 信頼性の高いフィールドから抽出
        場コード = record[0:2].decode("shift_jis", errors="replace").strip()
        try:
            year_2digit = record[2:4].decode("shift_jis", errors="replace").strip()
            year_4digit = convert_year_to_4digits(year_2digit, index)
        except:
            year_4digit = "2000"  # デフォルト値
        
        try:
            回 = record[4:5].decode("shift_jis", errors="replace").strip()
            if not 回.isdigit():
                回 = "0"  # 解析不能な場合はデフォルト値
        except:
            回 = "0"
            
        try:
            日 = record[5:6].decode("shift_jis", errors="replace").strip()
            if not 日.isdigit():
                日 = "0"  # 解析不能な場合はデフォルト値
        except:
            日 = "0"
            
        try:
            Ｒ = record[6:8].decode("shift_jis", errors="replace").strip()
            if not Ｒ.isdigit():
                Ｒ = "00"  # 解析不能な場合はデフォルト値
        except:
            Ｒ = "00"
        
        # レースIDを作成
        レースID = 場コード + year_4digit + 回 + 日 + Ｒ
        
        # 場名を取得
        場名 = JRA_MASTERS["場コード"].get(場コード, "")
        
        # 結果辞書を初期化
        result = {
            "場コード": 場コード,
            "場名": 場名,
            "年": year_4digit,
            "回": 回,
            "日": 日,
            "Ｒ": Ｒ,
            "レースID": レースID
        }
        
        # ファイルの実際のフォーマットに合わせたデータ抽出
        # バイアス情報は括弧付きの数字の並び（例：(*15,9)(3,8,12)13(10,14)(2,11)(4,5)1-7,6-16）
        try:
            # 実際のフォーマットからバイアス情報を抽出
            # トラックバイアスを抽出するために、レコードのテキスト全体を取得
            full_text = record.decode("shift_jis", errors="replace").strip()
            
            # レコードを空白で分割してバイアス情報を含む部分を抽出
            # 実際のレコードではバイアス情報は括弧を含む文字列として存在
            parts = full_text.split()
            bias_info = None
            race_comment = ""
            
            # 括弧を含む要素を探してバイアス情報として抽出
            for i, part in enumerate(parts):
                if "(" in part and ")" in part:
                    # 複数の括弧がある場合は、それをバイアス情報として取得
                    bias_info = part
                    
                    # レースコメントがあれば取得（バイアス情報の後にあるテキスト）
                    if i + 1 < len(parts):
                        race_comment = " ".join(parts[i+1:])
                    break
            
            # バイアス情報が見つかった場合は、各コーナーのバイアス情報を抽出
            if bias_info:
                print(f"抽出されたバイアス情報: {bias_info}")  # デバッグ出力
                
                # バイアス情報からコーナー毎に抽出
                # 記述フォーマットが "(数字,数字)(数字,数字)..." のような形式
                first_corner_bias = ""
                second_corner_bias = ""
                homestr_bias = ""
                third_corner_bias = ""
                fourth_corner_bias = ""
                straight_bias = ""
                
                # コーナー情報の抽出位置を特定
                # 例：(*3,9)(6,10,16)(1,2,11)(4,15)13,5-12,14,7,8 の場合
                # 1コーナー: (*3,9)
                # 2コーナー: (6,10,16)
                # 3コーナー: (1,2,11) 
                # 4コーナー: (4,15)
                # ホームストレッチ: 13,5-12,14,7,8
                
                # バイアス情報の分割（括弧で区切られた部分を抽出）
                import re
                bracket_parts = re.findall(r'\([^)]*\)', bias_info)
                print(f"抽出された括弧部分: {bracket_parts}")  # デバッグ出力
                
                # 各コーナーのバイアス情報を設定
                if len(bracket_parts) >= 1:
                    first_corner_bias = bracket_parts[0]
                if len(bracket_parts) >= 2:
                    second_corner_bias = bracket_parts[1]
                if len(bracket_parts) >= 3:
                    third_corner_bias = bracket_parts[2]
                if len(bracket_parts) >= 4:
                    fourth_corner_bias = bracket_parts[3]
                
                # 直線・向正のバイアス（残りの部分）
                remaining = bias_info
                for part in bracket_parts:
                    remaining = remaining.replace(part, '')
                
                # 残りの部分を直線バイアスとして使用
                straight_bias = remaining.strip()
                
                # バイアス情報をセット
                result.update({
                    "１角バイアス": first_corner_bias,
                    "２角バイアス": second_corner_bias,
                    "向正バイアス": homestr_bias,  # 向正バイアス情報は別途抽出が必要
                    "３角バイアス": third_corner_bias,
                    "４角バイアス": fourth_corner_bias,
                    "直線バイアス": straight_bias,
                    "レースコメント": race_comment
                })
            else:
                # バイアス情報が見つからない場合
                result.update({
                    "１角バイアス": "",
                    "２角バイアス": "",
                    "向正バイアス": "",
                    "３角バイアス": "",
                    "４角バイアス": "",
                    "直線バイアス": "",
                    "レースコメント": ""
                })
        except Exception as e:
            # エラーが発生した場合は空の値を設定
            print(f"バイアス情報の抽出に失敗しました: {str(e)}")
            result.update({
                "１角バイアス": "",
                "２角バイアス": "",
                "向正バイアス": "",
                "３角バイアス": "",
                "４角バイアス": "",
                "直線バイアス": "",
                "レースコメント": ""
            })
        
        return result
    except Exception as e:
        print(f"レコード処理中にエラーが発生: {str(e)}")
        # 最低限の情報を持つ辞書を返す
        return {
            "場コード": "",
            "場名": "",
            "年": "",
            "回": "",
            "日": "",
            "Ｒ": "",
            "レースID": "",
            "１角バイアス": "",
            "２角バイアス": "",
            "向正バイアス": "",
            "３角バイアス": "",
            "４角バイアス": "",
            "直線バイアス": "",
            "レースコメント": ""
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
    sed100105_dir = Path("export/SED100105")  # SED100105用のディレクトリ
    with_bias_dir = Path("export/with_bias")  # _with_biasファイル用のディレクトリ
    
    # マージディレクトリが存在しない場合は作成
    if not separate_output and merged_dir:
        merged_dir.mkdir(exist_ok=True, parents=True)
    sed100105_dir.mkdir(exist_ok=True, parents=True)  # SED100105用のディレクトリを作成
    with_bias_dir.mkdir(exist_ok=True, parents=True)  # with_biasディレクトリを作成
    
    # SRB統合ファイルの読み込み
    srb_all_file = srb_export_dir / "formatted/SRB_ALL_formatted.csv"
    if not srb_all_file.exists():
        print("⚠️ SRB統合ファイルが見つかりません。先にSRBファイルの処理を実行してください。")
        return False
        
    try:
        srb_df = pd.read_csv(srb_all_file, encoding="utf-8")
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
            sed_df = pd.read_csv(sed_file, encoding="utf-8")
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
            
            # 結果の保存
            if separate_output:
                # SED100105の場合は特別なフォルダに保存
                if "SED100105" in sed_file.name:
                    output_file = sed100105_dir / f"{sed_file.stem}_with_bias.csv"
                else:
                    # 通常の場合はwith_biasディレクトリに保存
                    output_file = with_bias_dir / f"{sed_file.stem}_with_bias.csv"
                
                # 結合したデータを保存（フィルタリング済みのデータを使用）
                filtered_df.to_csv(output_file, index=False, encoding="utf-8")
                print(f"  ✓ データを {output_file} に保存しました。")
                
            else:
                # 統合データを保存
                output_file = merged_dir / f"{sed_file.stem}_with_bias.csv"
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
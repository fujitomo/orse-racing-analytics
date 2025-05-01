"""
データ処理のための共通ユーティリティ関数
"""

from pathlib import Path
import csv

def clean_text(text):
    """
    テキストから全角スペースを除去し、文字列を整形します
    
    Args:
        text (str): 整形する文字列
        
    Returns:
        str: 整形された文字列
    """
    # 全角スペース（\u3000）を除去
    text = text.replace('\u3000', '')
    # 前後の空白を除去
    text = text.strip()
    return text

def convert_year_to_4digits(year_2digit, record_index=None):
    """
    2桁の年を4桁に変換します
    
    Args:
        year_2digit (str): 2桁の年
        record_index (int, optional): レコードのインデックス（エラーメッセージ用）
        
    Returns:
        str: 4桁の年
        
    Raises:
        ValueError: 年の変換に失敗した場合
    """
    error_context = f" レコード {record_index} -" if record_index is not None else ""
    
    if not year_2digit.isdigit():
        raise ValueError(f"{error_context} 年の値に数字以外の文字が含まれています: '{year_2digit}'")
        
    year_int = int(year_2digit)
    if year_int >= 0 and year_int <= 99:
        return f"20{year_2digit.zfill(2)}"
    else:
        raise ValueError(f"{error_context} 不正な年の値です（範囲外）: {year_2digit}")

def process_fixed_length_file(input_file, output_file, record_length, headers, process_record_func):
    """
    固定長レコードファイルを処理し、CSVに変換します
    
    Args:
        input_file (str): 入力ファイルのパス
        output_file (str): 出力ファイルのパス
        record_length (int): レコードの長さ（バイト、改行文字を含まない）
        headers (list): CSVヘッダー
        process_record_func (callable): レコード処理関数
        
    Returns:
        list: フォーマット済みのレコードのリスト
    """
    try:
        with open(input_file, "rb") as infile:
            content = infile.read()
    except Exception as e:
        print(f"⚠️ ファイルの読み込みに失敗しました: {str(e)}")
        return []

    records = []
    pos = 0
    content_length = len(content)
    
    while pos < content_length:
        # 残りのコンテンツが1レコード分未満の場合は終了
        if pos + record_length > content_length:
            break
            
        # レコードを取得
        record = content[pos:pos + record_length]
        if len(record) == record_length:
            records.append(record)
            
        # 次のレコードの位置を計算
        next_pos = pos + record_length
        # 改行文字をスキップ
        while next_pos < content_length and content[next_pos:next_pos + 1] in [b'\r', b'\n']:
            next_pos += 1
            
        # 位置が進まない場合は無限ループを防ぐため終了
        if next_pos <= pos:
            print(f"⚠️ ファイル位置が進まないため処理を終了します: pos={pos}")
            break
            
        pos = next_pos

    print(f"読み込んだレコード数: {len(records)}")

    formatted_records = []
    for index, record in enumerate(records, 1):
        try:
            fields = process_record_func(record, index)
            if fields:
                formatted_records.append(fields)
        except Exception as e:
            print(f"⚠️ レコード {index} の処理中にエラーが発生しました: {str(e)}")
            continue

    try:
        # 出力ディレクトリの作成
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSVファイルの保存
        with open(output_path, "w", encoding="utf-8-sig", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(formatted_records)

        print(f"✅ 整形されたファイルが {output_path} に保存されました。")
        print(f"処理したレコード数: {len(formatted_records)}")
    except Exception as e:
        print(f"⚠️ CSVファイルの保存中にエラーが発生しました: {str(e)}")

    return formatted_records

def process_all_files(file_type, import_dir, export_dir, process_file_func):
    """
    指定されたタイプのすべてのファイルを処理します
    
    Args:
        file_type (str): ファイルタイプ（例: "BAC", "SED"）
        import_dir (Path): 入力ディレクトリ
        export_dir (Path): 出力ディレクトリ
        process_file_func (callable): ファイル処理関数
    """
    import_dir = Path(import_dir)
    export_dir = Path(export_dir)
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルの検索
    target_files = []
    patterns = [f"{file_type}*.txt", f"{file_type}[0-9][0-9][0-9][0-9][0-9][0-9].txt"]
    
    # サブフォルダも含めて検索
    for pattern in patterns:
        target_files.extend(list(import_dir.rglob(pattern)))
    
    # 重複を除去してソート
    target_files = sorted(set(target_files))
    
    print(f"\n処理対象の{file_type}ファイル数: {len(target_files)}")
    print("処理対象ファイル:")
    for file in target_files:
        print(f"- {file.name}")
    
    processed_files = []
    for target_file in target_files:
        output_file = export_dir / f"{target_file.stem}_formatted.csv"
        print(f"\n処理中: {target_file}")
        process_file_func(str(target_file), str(output_file))
        processed_files.append(output_file)
    
    return processed_files 
"""
BACデータ（レース基本情報）の処理モジュール
"""

from horse_racing.data.constants.jra_masters import JRA_MASTERS
from .utils import process_fixed_length_file, process_all_files, convert_year_to_4digits, clean_text

def process_bac_record(record, index, exclude_turf=False, turf_only=False):
    """
    BACレコードを処理します
    
    Args:
        record (bytes): バイナリレコード
        index (int): レコードのインデックス
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
        
    Returns:
        list: 処理されたフィールドのリスト
    """
    try:
        # 記号の処理（3桁の分解）
        記号1 = JRA_MASTERS["記号_1"].get(record[31:32].decode("shift_jis", errors="ignore").strip(), "")
        記号2 = JRA_MASTERS["記号_2"].get(record[32:33].decode("shift_jis", errors="ignore").strip(), "")
        記号3 = JRA_MASTERS["記号_3"].get(record[33:34].decode("shift_jis", errors="ignore").strip(), "")
        記号 = f"{記号1}{記号2}{記号3}".strip()

        # 年の処理（2桁から4桁に変換）
        year_2digit = record[2:4].decode("shift_jis", errors="ignore").strip()
        year_4digit = convert_year_to_4digits(year_2digit, index)

        # 場コードの処理
        場コード = record[0:2].decode("shift_jis", errors="ignore").strip()
        場名 = JRA_MASTERS["場コード"].get(場コード, "")
        if not 場名:
            print(f"⚠️ レコード {index} - 不明な場コード: {場コード}")
            return None

        # 芝ダ障害コードの取得
        芝ダ障害コード_原値 = record[24:25].decode("shift_jis", errors="ignore").strip()
        芝ダ障害コード = JRA_MASTERS["芝ダ障害コード"].get(芝ダ障害コード_原値)
        
        # 芝コースを除外する場合
        if exclude_turf and 芝ダ障害コード == "芝":
            print(f"芝コースを除外: レコード {index}")
            return None
            
        # 芝コースのみ処理する場合
        if turf_only and 芝ダ障害コード != "芝":
            # print(f"芝コース以外を除外: レコード {index}") # ログが冗長なためコメントアウト
            return None

        # レースIDを生成
        レースID = (
            場コード + 
            year_4digit + 
            record[4:5].decode("shift_jis", errors="ignore").strip() + 
            record[5:6].decode("shift_jis", errors="ignore").strip() + 
            record[6:8].decode("shift_jis", errors="ignore").strip()
        )

        # 賞金フィールドのデコードと数値変換
        try:
            prize1 = int(record[125:130].decode("shift_jis", errors="ignore").strip())
            prize2 = int(record[130:135].decode("shift_jis", errors="ignore").strip())
            prize3 = int(record[135:140].decode("shift_jis", errors="ignore").strip())
            prize4 = int(record[140:145].decode("shift_jis", errors="ignore").strip())
            prize5 = int(record[145:150].decode("shift_jis", errors="ignore").strip())
            added_prize1 = int(record[150:155].decode("shift_jis", errors="ignore").strip())
            added_prize2 = int(record[155:160].decode("shift_jis", errors="ignore").strip())
        except ValueError:
            # 数値変換に失敗した場合は、0として扱う
            prize1, prize2, prize3, prize4, prize5, added_prize1, added_prize2 = 0, 0, 0, 0, 0, 0, 0

        # 1着賞金(1着算入賞金込み) の計算
        total_first_prize = prize1 + added_prize1

        # 2着賞金(2着算入賞金込み) の計算
        total_second_prize = prize2 + added_prize2

        # 平均賞金の計算
        total_prize_pool = prize1 + prize2 + prize3 + prize4 + prize5 + added_prize1 + added_prize2
        average_prize = total_prize_pool / 5 if total_prize_pool > 0 else 0

        fields = [
            レースID, # レースIDを追加
            場コード,
            場名,
            year_4digit,  # 年（4桁）
            record[4:5].decode("shift_jis", errors="ignore").strip(),  # 回
            record[5:6].decode("shift_jis", errors="ignore").strip(),  # 日
            record[6:8].decode("shift_jis", errors="ignore").strip(),  # R
            record[8:16].decode("shift_jis", errors="ignore").strip(),  # 年月日
            record[16:20].decode("shift_jis", errors="ignore").strip(),  # 発走時間
            record[20:24].decode("shift_jis", errors="ignore").strip(),  # 距離
            芝ダ障害コード,  # 芝ダ障害コード
            JRA_MASTERS["右左"].get(record[25:26].decode("shift_jis", errors="ignore").strip()),  # 右左
            JRA_MASTERS["内外"].get(record[26:27].decode("shift_jis", errors="ignore").strip()),  # 内外
            JRA_MASTERS["種別"].get(record[27:29].decode("shift_jis", errors="ignore").strip()),  # 種別
            JRA_MASTERS["条件"].get(record[29:31].decode("shift_jis", errors="ignore").strip()),  # 条件
            記号,  # 記号（3桁を結合）
            JRA_MASTERS["重量"].get(record[34:35].decode("shift_jis", errors="ignore").strip()),  # 重量
            JRA_MASTERS["グレード"].get(record[35:36].decode("shift_jis", errors="ignore").strip()),  # グレード
            clean_text(record[36:86].decode("shift_jis", errors="ignore")),  # レース名（50バイト）
            record[86:94].decode("shift_jis", errors="ignore").strip(),  # 回数（8バイト）
            record[94:96].decode("shift_jis", errors="ignore").strip(),  # 頭数（2バイト）
            JRA_MASTERS["コース"].get(record[96:97].decode("shift_jis", errors="ignore").strip()),  # コース（1バイト）
            JRA_MASTERS["開催区分"].get(record[97:98].decode("shift_jis", errors="ignore").strip()),  # 開催区分（1バイト）
            record[98:106].decode("shift_jis", errors="ignore").strip(),  # レース名短縮（8バイト）
            record[106:124].decode("shift_jis", errors="ignore").strip(),  # レース名９文字（18バイト）
            JRA_MASTERS["データ区分"].get(record[124:125].decode("shift_jis", errors="ignore").strip()),  # データ区分（1バイト）
            prize1,  # 1着賞金
            prize2,  # 2着賞金
            prize3,  # 3着賞金
            prize4,  # 4着賞金
            prize5,  # 5着賞金
            added_prize1,  # 1着算入賞金
            added_prize2,  # 2着算入賞金
            record[160:176].decode("shift_jis", errors="ignore").strip(),  # 馬券発売フラグ（16バイト）
            record[176:177].decode("shift_jis", errors="ignore").strip(),  # WIN5フラグ（1バイト）
            total_first_prize, # 1着賞金(1着算入賞金込み)
            total_second_prize, # 2着賞金(2着算入賞金込み)
            average_prize # 平均賞金
        ]

        # 必須フィールドの検証
        if not all([fields[0], fields[1], fields[2], fields[3], fields[4]]):
            print(f"⚠️ レコード {index} - 必須フィールドが不足しています")
            return None

        return fields
    except Exception as e:
        print(f"⚠️ レコード {index} - 処理中にエラーが発生しました: {str(e)}")
        return None

def format_bac_file(input_file, output_file, exclude_turf=False, turf_only=False):
    """
    BACファイルを整形してCSVに変換します。
    
    Args:
        input_file (str): 入力ファイルのパス
        output_file (str): 出力ファイルのパス
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
        
    Returns:
        list: フォーマット済みのレコードのリスト
    """
    # ヘッダー定義
    headers = [
        "レースID", "場コード", "場名", "年", "回", "日", "R", "年月日", "発走時間", "距離",
        "芝ダ障害コード", "右左", "内外", "種別", "条件", "記号", "重量", "グレード",
        "レース名", "回数", "頭数", "コース", "開催区分", "レース名短縮", "レース名９文字",
        "データ区分", "1着賞金", "2着賞金", "3着賞金", "4着賞金", "5着賞金",
        "1着算入賞金", "2着算入賞金", "馬券発売フラグ", "WIN5フラグ",
        "1着賞金(1着算入賞金込み)", "2着賞金(2着算入賞金込み)", "平均賞金"
    ]
    
    # process_recordに芝コース除外オプションを渡す
    def process_record_with_turf_filter(record, index):
        return process_bac_record(record, index, exclude_turf, turf_only)
    
    return process_fixed_length_file(input_file, output_file, 183, headers, process_record_with_turf_filter)

def process_all_bac_files(exclude_turf=False, turf_only=False):
    """
    すべてのBACファイルを処理します。
    
    Args:
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
    """
    # format_bac_fileに芝コース除外オプションを渡すラッパー関数
    def format_bac_file_wrapper(input_file, output_file):
        return format_bac_file(input_file, output_file, exclude_turf, turf_only)
    
    process_all_files("BAC", "import/BAC", "export/BAC/formatted", format_bac_file_wrapper)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BACファイルを処理します')
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', action='store_true', help='芝コースを除外する')
    track_group.add_argument('--turf-only', action='store_true', help='芝コースのみを処理する')
    args = parser.parse_args()
    
    process_all_bac_files(exclude_turf=args.exclude_turf, turf_only=args.turf_only) 
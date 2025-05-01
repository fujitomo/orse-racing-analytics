"""
BACデータ（レース基本情報）の処理モジュール
"""

from horse_racing.data.constants.jra_masters import JRA_MASTERS
from .utils import process_fixed_length_file, process_all_files, convert_year_to_4digits, clean_text

def process_bac_record(record, index):
    """
    BACレコードを処理します
    
    Args:
        record (bytes): バイナリレコード
        index (int): レコードのインデックス
        
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

        fields = [
            場名,  # 場コード
            year_4digit,  # 年（4桁）
            record[4:5].decode("shift_jis", errors="ignore").strip(),  # 回
            record[5:6].decode("shift_jis", errors="ignore").strip(),  # 日
            record[6:8].decode("shift_jis", errors="ignore").strip(),  # R
            record[8:16].decode("shift_jis", errors="ignore").strip(),  # 年月日
            record[16:20].decode("shift_jis", errors="ignore").strip(),  # 発走時間
            record[20:24].decode("shift_jis", errors="ignore").strip(),  # 距離
            JRA_MASTERS["芝ダ障害コード"].get(record[24:25].decode("shift_jis", errors="ignore").strip()),  # 芝ダ障害コード
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
            record[125:130].decode("shift_jis", errors="ignore").strip(),  # 1着賞金（5バイト）
            record[130:135].decode("shift_jis", errors="ignore").strip(),  # 2着賞金（5バイト）
            record[135:140].decode("shift_jis", errors="ignore").strip(),  # 3着賞金（5バイト）
            record[140:145].decode("shift_jis", errors="ignore").strip(),  # 4着賞金（5バイト）
            record[145:150].decode("shift_jis", errors="ignore").strip(),  # 5着賞金（5バイト）
            record[150:155].decode("shift_jis", errors="ignore").strip(),  # 1着算入賞金（5バイト）
            record[155:160].decode("shift_jis", errors="ignore").strip(),  # 2着算入賞金（5バイト）
            record[160:176].decode("shift_jis", errors="ignore").strip(),  # 馬券発売フラグ（16バイト）
            record[176:177].decode("shift_jis", errors="ignore").strip()  # WIN5フラグ（1バイト）
        ]

        # 必須フィールドの検証
        if not all([fields[0], fields[1], fields[2], fields[3], fields[4]]):
            print(f"⚠️ レコード {index} - 必須フィールドが不足しています")
            return None

        return fields
    except Exception as e:
        print(f"⚠️ レコード {index} - 処理中にエラーが発生しました: {str(e)}")
        return None

def format_bac_file(input_file, output_file):
    """
    BACファイルを整形してCSVに変換します。
    
    Args:
        input_file (str): 入力ファイルのパス
        output_file (str): 出力ファイルのパス
        
    Returns:
        list: フォーマット済みのレコードのリスト
    """
    # ヘッダー定義
    headers = [
        "場コード", "年", "回", "日", "R", "年月日", "発走時間", "距離",
        "芝ダ障害コード", "右左", "内外", "種別", "条件", "記号", "重量", "グレード",
        "レース名", "回数", "頭数", "コース", "開催区分", "レース名短縮", "レース名９文字",
        "データ区分", "1着賞金", "2着賞金", "3着賞金", "4着賞金", "5着賞金",
        "1着算入賞金", "2着算入賞金", "馬券発売フラグ", "WIN5フラグ"
    ]
    
    return process_fixed_length_file(input_file, output_file, 183, headers, process_bac_record)

def process_all_bac_files():
    """すべてのBACファイルを処理します。"""
    process_all_files("BAC", "import/BAC", "export/BAC", format_bac_file)

if __name__ == "__main__":
    process_all_bac_files() 
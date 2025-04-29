import csv
from pathlib import Path

def adjust_string_length(text, max_bytes):
    """
    文字列を指定されたバイト数に調整する
    全角文字は2バイト、半角文字は1バイトとして計算
    """
    result = ""
    current_bytes = 0
    
    for char in text.strip():
        # 全角文字は2バイト、半角文字は1バイトとして計算
        char_bytes = 2 if ord(char) > 255 else 1
        if current_bytes + char_bytes <= max_bytes:
            result += char
            current_bytes += char_bytes
        else:
            break
    
    return result

# JRAマスターデータの定義
jra_masters = {
    "場コード": {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
        "21": "旭川", "22": "札幌", "23": "門別", "24": "函館",
        "25": "盛岡", "26": "水沢", "27": "上山", "28": "新潟",
        "29": "三条", "30": "足利", "31": "宇都", "32": "高崎",
        "33": "浦和", "34": "船橋", "35": "大井", "36": "川崎",
        "37": "金沢", "38": "笠松", "39": "名古", "40": "中京",
        "41": "園田", "42": "姫路", "43": "益田", "44": "福山",
        "45": "高知", "46": "佐賀", "47": "荒尾", "48": "中津",
        "61": "英国", "62": "愛国", "63": "仏国", "64": "伊国",
        "65": "独国", "66": "米国", "67": "加国", "68": "UAE",
        "69": "豪州", "70": "新国", "71": "香港", "72": "チリ",
        "73": "星国", "74": "典国", "75": "マカ", "76": "墺国",
        "77": "土国", "78": "華国", "79": "韓国"
    },
    "芝ダ障害コード": {
        "1": "芝", "2": "ダート", "3": "障害"
    },
    "右左": {
        "1": "右", "2": "左", "3": "直", "9": "他"
    },
    "内外": {
        "1": "通常(内)", "2": "外", "3": "直ダ", "9": "他"
    },
    "種別": {
        "11": "２歳", "12": "３歳", "13": "３歳以上",
        "14": "４歳以上", "20": "障害", "99": "その他"
    },
    "条件": {
        "04": "1勝クラス", "05": "1勝クラス",
        "08": "2勝クラス", "09": "2勝クラス", "10": "2勝クラス",
        "15": "3勝クラス", "16": "3勝クラス",
        "A1": "新馬", "A2": "未出走", "A3": "未勝利",
        "OP": "オープン"
    },
    "記号_1": {  # 1桁目：馬の種類による条件
        "0": "なし", "1": "○混", "2": "○父", "3": "○市○抽",
        "4": "九州産限定", "5": "○国際混"
    },
    "記号_2": {  # 2桁目：馬の性別による条件
        "0": "なし", "1": "牡馬限定", "2": "牝馬限定",
        "3": "牡・せん馬限定", "4": "牡・牝馬限定"
    },
    "記号_3": {  # 3桁目：交流競走の指定
        "0": "なし", "1": "○指", "2": "□指",
        "3": "○特指", "4": "若手"
    },
    "重量": {
        "1": "ハンデ", "2": "別定", "3": "馬齢", "4": "定量"
    },
    "グレード": {
        "1": "Ｇ１", "2": "Ｇ２", "3": "Ｇ３",
        "4": "重賞", "5": "特別", "6": "Ｌ"
    },
    "馬場状態": {
        "10": "良", "11": "速良", "12": "遅良",
        "20": "稍重", "21": "速稍重", "22": "遅稍重",
        "30": "重", "31": "速重", "32": "遅重",
        "40": "不良", "41": "速不良", "42": "遅不良"
    },
    "異常区分": {
        "0": "異常なし", "1": "取消", "2": "除外",
        "3": "中止", "4": "失格", "5": "降着", "6": "再騎乗"
    },
    "コース": {
        "1": "A", "2": "A1", "3": "A2",
        "4": "B", "5": "C", "6": "D"
    },
    "開催区分": {
        "1": "関東", "2": "関西", "3": "ローカル"
    },
    "データ区分": {
        "1": "特別登録", "2": "想定確定", "3": "前日"
    }
}

def format_bac_file(input_file, output_file):
    record_length = 184  # 固定長レコードのバイト数（改行含む）

    # ヘッダー定義
    headers = [
        "場コード", "年", "回", "日", "R", "年月日", "発走時間", "距離",
        "芝ダ障害コード", "右左", "内外", "種別", "条件", "記号", "重量", "グレード",
        "レース名", "回数", "頭数", "コース", "開催区分", "レース名短縮", "レース名９文字",
        "データ区分", "1着賞金", "2着賞金", "3着賞金", "4着賞金", "5着賞金",
        "1着算入賞金", "2着算入賞金", "馬券発売フラグ", "WIN5フラグ"
    ]

    with open(input_file, "rb") as infile:
        content = infile.read()

    records = []
    pos = 0
    while pos < len(content):
        record = content[pos:pos + record_length]
        if len(record) == record_length:
            records.append(record)
        pos += record_length

    print(f"読み込んだレコード数: {len(records)}")

    formatted_records = []
    for index, record in enumerate(records):
        try:
            # 記号の処理（3桁の分解）
            記号1 = jra_masters["記号_1"].get(record[31:32].decode("shift_jis"), "")
            記号2 = jra_masters["記号_2"].get(record[32:33].decode("shift_jis"), "")
            記号3 = jra_masters["記号_3"].get(record[33:34].decode("shift_jis"), "")
            記号 = f"{記号1}{記号2}{記号3}".strip()

            # 年の処理（2桁から4桁に変換）
            year_2digit = record[2:4].decode("shift_jis").strip()
            try:
                if not year_2digit.isdigit():
                    print(f"⚠️ 年の値に数字以外の文字が含まれています: '{year_2digit}'")
                    continue
                    
                year_int = int(year_2digit)
                if year_int >= 0 and year_int <= 99:
                    year_4digit = f"20{year_2digit}"
                else:
                    print(f"⚠️ 不正な年の値です（範囲外）: {year_2digit}")
                    continue
            except ValueError as e:
                print(f"⚠️ 年の値の変換に失敗しました: '{year_2digit}' - エラー: {str(e)}")
                continue

            fields = [
                jra_masters["場コード"].get(record[0:2].decode("shift_jis")),  # 場コード
                year_4digit,  # 年（4桁）
                record[4:5].decode("shift_jis"),  # 回
                record[5:6].decode("shift_jis"),  # 日
                record[6:8].decode("shift_jis"),  # R
                record[8:16].decode("shift_jis"),  # 年月日
                record[16:20].decode("shift_jis"),  # 発走時間
                record[20:24].decode("shift_jis"),  # 距離
                jra_masters["芝ダ障害コード"].get(record[24:25].decode("shift_jis")),  # 芝ダ障害コード
                jra_masters["右左"].get(record[25:26].decode("shift_jis")),  # 右左
                jra_masters["内外"].get(record[26:27].decode("shift_jis")),  # 内外
                jra_masters["種別"].get(record[27:29].decode("shift_jis")),  # 種別
                jra_masters["条件"].get(record[29:31].decode("shift_jis")),  # 条件
                記号,  # 記号（3桁を結合）
                jra_masters["重量"].get(record[34:35].decode("shift_jis")),  # 重量
                jra_masters["グレード"].get(record[35:36].decode("shift_jis")),  # グレード
                record[36:86].decode("shift_jis").strip(),  # レース名（50バイト）
                record[86:94].decode("shift_jis").strip(),  # 回数（8バイト）
                record[94:96].decode("shift_jis"),  # 頭数（2バイト）
                jra_masters["コース"].get(record[96:97].decode("shift_jis")),  # コース（1バイト）
                jra_masters["開催区分"].get(record[97:98].decode("shift_jis")),  # 開催区分（1バイト）
                record[98:106].decode("shift_jis").strip(),  # レース名短縮（8バイト）
                record[106:124].decode("shift_jis").strip(),  # レース名９文字（18バイト）
                jra_masters["データ区分"].get(record[124:125].decode("shift_jis")),  # データ区分（1バイト）
                record[125:130].decode("shift_jis").strip(),  # 1着賞金（5バイト）
                record[130:135].decode("shift_jis").strip(),  # 2着賞金（5バイト）
                record[135:140].decode("shift_jis").strip(),  # 3着賞金（5バイト）
                record[140:145].decode("shift_jis").strip(),  # 4着賞金（5バイト）
                record[145:150].decode("shift_jis").strip(),  # 5着賞金（5バイト）
                record[150:155].decode("shift_jis").strip(),  # 1着算入賞金（5バイト）
                record[155:160].decode("shift_jis").strip(),  # 2着算入賞金（5バイト）
                record[160:176].decode("shift_jis"),  # 馬券発売フラグ（16バイト）
                record[176:177].decode("shift_jis").strip()  # WIN5フラグ（1バイト）
            ]
            formatted_records.append(fields)
        except Exception as e:
            print(f"⚠️ レコード {index + 1} の処理中にエラーが発生しました: {str(e)}")
            continue

    with open(output_file, "w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(formatted_records)

    print(f"✅ 整形されたファイルが {output_file} に保存されました。")
    print(f"処理したレコード数: {len(formatted_records)}")

def process_all_bac_files():
    # importフォルダとexportフォルダのパスを設定
    import_dir = Path("import/BAC")
    export_dir = Path("export/BAC")
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # importフォルダ内のすべてのBACファイルを処理
    bac_files = []
    patterns = ["BAC*.txt", "BAC[0-9][0-9][0-9][0-9][0-9][0-9].txt"]
    
    # サブフォルダも含めて検索
    for pattern in patterns:
        bac_files.extend(list(import_dir.rglob(pattern)))
    
    # 重複を除去
    bac_files = list(set(bac_files))
    
    print(f"処理対象のファイル数: {len(bac_files)}")
    print("処理対象ファイル:")
    for file in bac_files:
        print(f"- {file.relative_to(import_dir)}")
    
    for bac_file in bac_files:
        # 出力ファイル名を生成（フォルダ構造なし）
        output_file = export_dir / f"{bac_file.stem}.csv"
        
        print(f"\n処理中: {bac_file.relative_to(import_dir)}")
        format_bac_file(str(bac_file), str(output_file))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # コマンドライン引数が指定された場合
        input_path = Path(sys.argv[1])
        if input_path.is_file():
            # 単一ファイルの処理
            output_file = Path("export/BAC") / f"{input_path.stem}_formatted.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            format_bac_file(str(input_path), str(output_file))
        elif input_path.is_dir():
            # ディレクトリの処理
            process_all_bac_files()
        else:
            print(f"エラー: 指定されたパス '{input_path}' は存在しません。")
    else:
        # 引数なしの場合はデフォルトの処理を実行
        process_all_bac_files() 
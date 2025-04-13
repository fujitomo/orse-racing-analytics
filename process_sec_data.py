import csv
from pathlib import Path

# JRAマスターデータの定義
jra_masters = {
    "場コード": {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉", "21": "旭川", "22": "札幌",
        "23": "門別", "24": "函館", "25": "盛岡", "26": "水沢",
        "27": "上山", "28": "新潟", "29": "三条", "30": "足利",
        "31": "宇都", "32": "高崎", "33": "浦和", "34": "船橋",
        "35": "大井", "36": "川崎", "37": "金沢", "38": "笠松",
        "39": "名古", "40": "中京", "41": "園田", "42": "姫路",
        "43": "益田", "44": "福山", "45": "高知", "46": "佐賀",
        "47": "荒尾", "48": "中津", "61": "英国", "62": "愛国",
        "63": "仏国", "64": "伊国", "65": "独国", "66": "米国",
        "67": "加国", "68": "UAE", "69": "豪州", "70": "新国",
        "71": "香港", "72": "チリ", "73": "星国", "74": "典国",
        "75": "マカ", "76": "墺国", "77": "土国", "78": "華国",
        "79": "韓国"
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
    "馬場状態": {
        "10": "良", "11": "速良", "12": "遅良",
        "20": "稍重", "21": "速稍重", "22": "遅稍重",
        "30": "重", "31": "速重", "32": "遅重",
        "40": "不良", "41": "速不良", "42": "遅不良",
        "1": "良", "2": "稍重", "3": "重", "4": "不良"
    },
    "種別": {
        "11": "2歳", "12": "3歳", "13": "3歳以上",
        "14": "4歳以上", "20": "障害", "99": "その他"
    },
    "条件": {
        "04": "1勝クラス", "05": "1勝クラス", "08": "2勝クラス",
        "09": "2勝クラス", "10": "2勝クラス", "15": "3勝クラス",
        "16": "3勝クラス", "A1": "新馬", "A2": "未出走",
        "A3": "未勝利", "OP": "オープン"
    },
    "記号": {
        "100": "○混", "200": "○父", "300": "○市○抽",
        "400": "九州産限定", "500": "○国際混", "010": "牡馬限定",
        "020": "牝馬限定", "030": "牡・せん馬限定", "040": "牡・牝馬限定",
        "001": "○指", "002": "□指", "003": "○特指", "004": "若手"
    },
    "重量": {
        "1": "ハンデ", "2": "別定", "3": "馬齢", "4": "定量"
    },
    "グレード": {
        "1": "G1", "2": "G2", "3": "G3", "4": "重賞",
        "5": "特別", "6": "L"
    },
    "異常区分": {
        "0": "異常なし", "1": "取消", "2": "除外",
        "3": "中止", "4": "失格", "5": "降着", "6": "再騎乗"
    },
    "コース取り": {
        "1": "最内", "2": "内", "3": "中",
        "4": "外", "5": "大外"
    },
    "馬体コード": {
        "1": "太い", "2": "余裕", "3": "良い",
        "4": "普通", "5": "細い", "6": "張り",
        "7": "緩い"
    },
    "気配コード": {
        "1": "状態良", "2": "平凡", "3": "不安定",
        "4": "イレ込", "5": "気合良", "6": "気不足",
        "7": "チャカ", "8": "イレチ"
    },
    "天候コード": {
        "1": "晴", "2": "曇", "3": "小雨",
        "4": "雨", "5": "小雪", "6": "雪"
    },
    "脚質コード": {
        "1": "逃げ", "2": "先行", "3": "差し",
        "4": "追込", "5": "好位差し", "6": "自在"
    },
    "距離適性コード": {
        "1": "短距離", "2": "中距離", "3": "長距離",
        "5": "哩（マイル）", "6": "万能"
    },
    "上昇度": {
        "1": "AA", "2": "A", "3": "B",
        "4": "C", "5": "?"
    },
    "調教矢印コード": {
        "1": "デキ抜群", "2": "上昇", "3": "平行線",
        "4": "やや下降気味", "5": "デキ落ち"
    },
    "厩舎評価コード": {
        "1": "超強気", "2": "強気", "3": "現状維持",
        "4": "弱気"
    },
    "蹄コード": {
        "01": "大ベタ", "02": "中ベタ", "03": "小ベタ",
        "04": "細ベタ", "05": "大立", "06": "中立",
        "07": "小立", "08": "細立", "09": "大標準",
        "10": "中標準", "11": "小標準", "12": "細標準",
        "17": "大標起", "18": "中標起", "19": "小標起",
        "20": "細標起", "21": "大標ベ", "22": "中標ベ",
        "23": "小標ベ", "24": "細標ベ"
    },
    "重適性コード": {
        "1": "◎", "2": "○", "3": "△"
    },
    "クラスコード": {
        "01": "芝G1", "02": "芝G2", "03": "芝G3",
        "04": "芝OP A", "05": "芝OP B", "06": "芝OP C",
        "07": "芝3勝A", "08": "芝3勝B", "09": "芝3勝C",
        "10": "芝2勝A", "11": "芝2勝B", "12": "芝2勝C",
        "13": "芝1勝A", "14": "芝1勝B", "15": "芝1勝C",
        "16": "芝未 A", "17": "芝未 B", "18": "芝未 C",
        "21": "ダG1", "22": "ダG2", "23": "ダG3",
        "24": "ダOP A", "25": "ダOP B", "26": "ダOP C",
        "27": "ダ3勝A", "28": "ダ3勝B", "29": "ダ3勝C",
        "30": "ダ2勝A", "31": "ダ2勝B", "32": "ダ2勝C",
        "33": "ダ1勝A", "34": "ダ1勝B", "35": "ダ1勝C",
        "36": "ダ未 A", "37": "ダ未 B", "38": "ダ未 C",
        "51": "障G1", "52": "障G2", "53": "障G3",
        "54": "障OP A", "55": "障OP B", "56": "障OP C",
        "57": "障1勝A", "58": "障1勝B", "59": "障1勝C",
        "60": "障未 A", "61": "障未 B", "62": "障未 C"
    },
    "印コード": {
        "1": "◎", "2": "○", "3": "▲",
        "4": "注", "5": "△", "6": "△",
        "9": "☆"
    },
    "毛色コード": {
        "01": "栗毛", "02": "栃栗", "03": "鹿毛",
        "04": "黒鹿", "05": "青鹿", "06": "青毛",
        "07": "芦毛", "08": "栗粕", "09": "鹿粕",
        "10": "青粕", "11": "白毛"
    },
    "馬記号コード": {
        "00": "", "01": "○抽", "02": "□抽",
        "03": "○父", "04": "○市", "05": "○地",
        "06": "○外", "07": "○父○抽", "08": "○父○市",
        "09": "○父○地", "10": "○市○地", "11": "○外○地",
        "12": "○父○市○地", "15": "○招", "16": "○招○外",
        "17": "○招○父", "18": "○招○市", "19": "○招○父○市",
        "20": "○父○外", "21": "□地", "22": "○外□地",
        "23": "○父□地", "24": "○市□地", "25": "○父○市□地",
        "26": "□外", "27": "○父□外"
    },
    "展開記号コード": {
        "1": "<", "2": "@", "3": "*",
        "4": "?", "0": "("
    },
    "休養理由分類コード": {
        "01": "放牧", "02": "放牧(故障、骨折等)", "03": "放牧(不安、ソエ等)",
        "04": "放牧(病気)", "05": "放牧(再審査)", "06": "放牧(出走停止)",
        "07": "放牧(手術）", "11": "調整", "12": "調整(故障、骨折等)",
        "13": "調整(不安、ソエ等)", "14": "調整(病気)", "15": "調整(再審査)",
        "16": "調整(出走停止)", "21": "その他"
    }
}

def get_first_prize_money(bac_file):
    """
    BACファイルから1着賞金の内容を取得する関数
    """
    prize_money_dict = {}  # キー: (場コード, 年, 回, 日, R), 値: 1着賞金

    try:
        with open(bac_file, "r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                # 場コードを数値コードに変換
                venue_code = None
                for code, name in jra_masters["場コード"].items():
                    if name == row["場コード"]:
                        venue_code = code
                        break
                
                if venue_code is None:
                    print(f"⚠️ 場コードが見つかりません: {row['場コード']}")
                    continue

                # キーを作成
                key = (venue_code, row["年"], row["回"], row["日"], row["R"])
                print(f"BAC: キー={key}, 賞金={row["1着賞金"]}")
                prize_money = row["1着賞金"]
                
                if prize_money:  # 空でない場合のみ追加
                    prize_money_dict[key] = prize_money
                    print(f"BAC: キー={key}, 賞金={prize_money}")
    except Exception as e:
        print(f"⚠️ BACファイルの処理中にエラーが発生しました: {str(e)}")

    print(f"取得した1着賞金の数: {len(prize_money_dict)}")
    return prize_money_dict

def format_sec_file(input_file, output_file, prize_money_dict):
    record_length = 346  # 固定長レコードのバイト数（改行含む）

    # ヘッダー定義（1着賞金を追加）
    headers = [
        "場コード", "年", "回", "日", "R", "馬番", "血統登録番号", "年月日", "馬名",
        "距離", "芝ダ障害コード", "右左", "内外", "馬場状態", "種別", "条件", "記号",
        "重量", "グレード", "レース名", "頭数", "レース名略称", "着順", "異常区分",
        "タイム", "斤量", "騎手名", "調教師名", "確定単勝オッズ", "確定単勝人気順位",
        "IDM", "素点", "馬場差", "ペース", "出遅", "位置取", "不利", "前不利",
        "中不利", "後不利", "レース", "コース取り", "上昇度コード", "クラスコード",
        "馬体コード", "気配コード", "レースペース", "馬ペース", "テン指数", "上がり指数",
        "ペース指数", "レースP指数", "1(2)着馬名", "1(2)着タイム差", "前3Fタイム",
        "後3Fタイム", "備考", "予備", "確定複勝オッズ下", "10時単勝オッズ",
        "10時複勝オッズ", "コーナー順位1", "コーナー順位2", "コーナー順位3",
        "コーナー順位4", "前3F先頭差", "後3F先頭差", "騎手コード", "調教師コード",
        "馬体重", "馬体重増減", "天候コード", "コース", "レース脚質", "1着賞金"
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

    print(f"SECファイルから読み込んだレコード数: {len(records)}")

    formatted_records = []
    for index, record in enumerate(records):
        try:
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

            # キーとなる情報を取得
            venue_code = record[0:2].decode("shift_jis")  # 場コード
            round_num = record[4:5].decode("shift_jis")  # 回
            day = record[5:6].decode("shift_jis")  # 日
            race_num = record[6:8].decode("shift_jis")  # R

            # 場コードの変換をデバッグ
            venue_code_raw = record[0:2].decode("shift_jis")
            venue_name = jra_masters["場コード"].get(venue_code_raw)
            print(f"場コード変換: {venue_code_raw} -> {venue_name}")

            # 1着賞金を取得
            key = (venue_code, year_4digit, round_num, day, race_num)
            prize_money = prize_money_dict.get(key, "")
            if prize_money:
                print(f"SEC: キー={key}, 賞金={prize_money}")
            else:
                print(f"SEC: キー={key}, 賞金なし")

            fields = [
                jra_masters["場コード"].get(record[0:2].decode("shift_jis")),  # 場コード(2)
                year_4digit,  # 年(2)
                record[4:5].decode("shift_jis"),  # 回(1)
                record[5:6].decode("shift_jis"),  # 日(1)
                record[6:8].decode("shift_jis"),  # R(2)
                record[8:10].decode("shift_jis"),  # 馬番(2)
                record[10:18].decode("shift_jis").strip(),  # 血統登録番号(8)
                record[18:26].decode("shift_jis"),  # 年月日(8)
                record[26:62].decode("shift_jis").strip(),  # 馬名(36)
                record[62:66].decode("shift_jis"),  # 距離(4)
                jra_masters["芝ダ障害コード"].get(record[66:67].decode("shift_jis")),  # 芝ダ障害コード(1)
                jra_masters["右左"].get(record[67:68].decode("shift_jis")),  # 右左(1)
                jra_masters["内外"].get(record[68:69].decode("shift_jis")),  # 内外(1)
                jra_masters["馬場状態"].get(record[69:71].decode("shift_jis")),  # 馬場状態(2)
                record[71:73].decode("shift_jis"),  # 種別(2)
                record[73:75].decode("shift_jis"),  # 条件(2)
                record[75:78].decode("shift_jis"),  # 記号(3)
                record[78:79].decode("shift_jis"),  # 重量(1)
                record[79:80].decode("shift_jis"),  # グレード(1)
                record[80:130].decode("shift_jis").strip(),  # レース名(50)
                record[130:132].decode("shift_jis"),  # 頭数(2)
                record[132:140].decode("shift_jis").strip(),  # レース名略称(8)
                record[140:142].decode("shift_jis"),  # 着順(2)
                record[142:143].decode("shift_jis"),  # 異常区分(1)
                record[143:147].decode("shift_jis"),  # タイム(4)
                record[147:150].decode("shift_jis"),  # 斤量(3)
                record[150:162].decode("shift_jis").strip(),  # 騎手名(12)
                record[162:174].decode("shift_jis").strip(),  # 調教師名(12)
                record[174:180].decode("shift_jis").strip(),  # 確定単勝オッズ(6)
                record[180:182].decode("shift_jis"),  # 確定単勝人気順位(2)
                record[182:185].decode("shift_jis").strip(),  # IDM(3)
                record[185:188].decode("shift_jis").strip(),  # 素点(3)
                record[188:191].decode("shift_jis").strip(),  # 馬場差(3)
                record[191:194].decode("shift_jis").strip(),  # ペース(3)
                record[194:197].decode("shift_jis").strip(),  # 出遅(3)
                record[197:200].decode("shift_jis").strip(),  # 位置取(3)
                record[200:203].decode("shift_jis").strip(),  # 不利(3)
                record[203:206].decode("shift_jis").strip(),  # 前不利(3)
                record[206:209].decode("shift_jis").strip(),  # 中不利(3)
                record[209:212].decode("shift_jis").strip(),  # 後不利(3)
                record[212:215].decode("shift_jis").strip(),  # レース(3)
                record[215:216].decode("shift_jis"),  # コース取り(1)
                record[216:217].decode("shift_jis"),  # 上昇度コード(1)
                record[217:219].decode("shift_jis"),  # クラスコード(2)
                record[219:220].decode("shift_jis"),  # 馬体コード(1)
                record[220:221].decode("shift_jis"),  # 気配コード(1)
                record[221:222].decode("shift_jis"),  # レースペース(1)
                record[222:223].decode("shift_jis"),  # 馬ペース(1)
                record[223:228].decode("shift_jis").strip(),  # テン指数(5)
                record[228:233].decode("shift_jis").strip(),  # 上がり指数(5)
                record[233:238].decode("shift_jis").strip(),  # ペース指数(5)
                record[238:243].decode("shift_jis").strip(),  # レースP指数(5)
                record[243:255].decode("shift_jis").strip(),  # 1(2)着馬名(12)
                record[255:258].decode("shift_jis"),  # 1(2)着タイム差(3)
                record[258:261].decode("shift_jis"),  # 前3Fタイム(3)
                record[261:264].decode("shift_jis"),  # 後3Fタイム(3)
                record[264:288].decode("shift_jis").strip(),  # 備考(24)
                record[288:290].decode("shift_jis").strip(),  # 予備(2)
                record[290:296].decode("shift_jis").strip(),  # 確定複勝オッズ下(6)
                record[296:302].decode("shift_jis").strip(),  # 10時単勝オッズ(6)
                record[302:308].decode("shift_jis").strip(),  # 10時複勝オッズ(6)
                record[308:310].decode("shift_jis"),  # コーナー順位1(2)
                record[310:312].decode("shift_jis"),  # コーナー順位2(2)
                record[312:314].decode("shift_jis"),  # コーナー順位3(2)
                record[314:316].decode("shift_jis"),  # コーナー順位4(2)
                record[316:319].decode("shift_jis"),  # 前3F先頭差(3)
                record[319:322].decode("shift_jis"),  # 後3F先頭差(3)
                record[322:327].decode("shift_jis"),  # 騎手コード(5)
                record[327:332].decode("shift_jis"),  # 調教師コード(5)
                record[332:335].decode("shift_jis").strip(),  # 馬体重(3)
                record[335:338].decode("shift_jis").strip(),  # 馬体重増減(3)
                jra_masters["天候コード"].get(record[338:339].decode("shift_jis")),  # 天候コード(1)
                record[339:340].decode("shift_jis"),  # コース(1)
                record[340:341].decode("shift_jis"),  # レース脚質(1)
                prize_money  # 1着賞金
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

def process_all_sec_files():
    # importフォルダとexportフォルダのパスを設定
    import_dir = Path("import/SEC")
    export_dir = Path("export/SEC")
    bac_dir = Path("export/BAC")
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # BACファイルから1着賞金を取得
    bac_files = list(bac_dir.glob("*.csv"))
    prize_money_dict = {}
    for bac_file in bac_files:
        print(f"\n処理中: {bac_file}")
        prize_money_dict.update(get_first_prize_money(str(bac_file)))

    print(f"\n取得した1着賞金の総数: {len(prize_money_dict)}")

    # SECファイルの処理
    sec_files = []
    patterns = ["SEC*.txt", "SEC[0-9][0-9][0-9][0-9][0-9][0-9].txt"]
    
    for pattern in patterns:
        sec_files.extend(list(import_dir.glob(pattern)))
    
    # 重複を除去
    sec_files = list(set(sec_files))
    
    print(f"\n処理対象のSECファイル数: {len(sec_files)}")
    print("処理対象ファイル:")
    for file in sec_files:
        print(f"- {file.name}")
    
    for sec_file in sec_files:
        output_file = export_dir / f"{sec_file.stem}_formatted.csv"
        print(f"\n処理中: {sec_file}")
        format_sec_file(str(sec_file), str(output_file), prize_money_dict)

if __name__ == "__main__":
    process_all_sec_files() 
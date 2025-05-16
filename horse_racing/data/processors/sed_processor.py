"""
SEDデータ（競走成績）の処理モジュール
"""

from ..constants.jra_masters import JRA_MASTERS
from .utils import process_fixed_length_file, process_all_files, convert_year_to_4digits
from pathlib import Path

def process_sed_record(record, index, exclude_turf=False, turf_only=False):
    """
    SEDレコードを処理します
    
    Args:
        record (bytes): バイナリレコード
        index (int): レコードのインデックス
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
        
    Returns:
        list: 処理されたフィールドのリスト
    """
    try:
        # 場コードの取得
        場コード = record[0:2].decode("shift_jis", errors="replace").strip()
        
        # 場名の取得
        場名 = JRA_MASTERS["場コード"].get(場コード, "")
        if not 場名:
            print(f"⚠️ レコード {index} - 不明な場コード: {場コード}")
            return None
            
        # 年の処理（2桁から4桁に変換）
        year_2digit = record[2:4].decode("shift_jis", errors="replace").strip()
        year_4digit = convert_year_to_4digits(year_2digit, index)

        # 回の処理
        回 = record[4:5].decode("shift_jis", errors="replace").strip()
        
        # 日の処理（16進数から10進数に変換）
        day_hex = record[5:6].decode("shift_jis", errors="replace").strip()
        try:
            day = str(int(day_hex, 16))
        except ValueError:
            print(f"⚠️ 日の値の変換に失敗しました: '{day_hex}'")
            day = "1"  # デフォルト値を設定
            
        # R（レース番号）の処理
        R = record[6:8].decode("shift_jis", errors="replace").strip()
        
        # 芝ダ障害コードの取得
        芝ダ障害コード_原値 = record[66:67].decode("shift_jis", errors="replace").strip()
        芝ダ障害コード = JRA_MASTERS["芝ダ障害コード"].get(芝ダ障害コード_原値, "")
        
        # 芝コースを除外する場合
        if exclude_turf and 芝ダ障害コード == "芝":
            print(f"芝コースを除外: レコード {index}")
            return None
            
        # 芝コースのみを処理する場合
        if turf_only and 芝ダ障害コード != "芝":
            print(f"芝コース以外を除外: レコード {index}")
            return None

        fields = [
            場コード,  # 場コード(2) - 変更: 場名ではなく場コードを保存
            year_4digit,  # 年(2)
            回,  # 回(1)
            day,  # 日(1) - 16進数から変換
            R,  # R(2)
            record[8:10].decode("shift_jis", errors="replace").strip(),  # 馬番(2)
            record[10:18].decode("shift_jis", errors="replace").strip(),  # 血統登録番号(8)
            record[18:26].decode("shift_jis", errors="replace").strip(),  # 年月日(8)
            record[26:62].decode("shift_jis", errors="replace").strip(),  # 馬名(36)
            record[62:66].decode("shift_jis", errors="replace").strip(),  # 距離(4)
            芝ダ障害コード,  # 芝ダ障害コード(1)
            JRA_MASTERS["右左"].get(record[67:68].decode("shift_jis", errors="replace").strip(), ""),  # 右左(1)
            JRA_MASTERS["内外"].get(record[68:69].decode("shift_jis", errors="replace").strip(), ""),  # 内外(1)
            JRA_MASTERS["馬場状態"].get(record[69:71].decode("shift_jis", errors="replace").strip(), ""),  # 馬場状態(2)
            record[71:73].decode("shift_jis", errors="replace").strip(),  # 種別(2)
            record[73:75].decode("shift_jis", errors="replace").strip(),  # 条件(2)
            record[75:78].decode("shift_jis", errors="replace").strip(),  # 記号(3)
            record[78:79].decode("shift_jis", errors="replace").strip(),  # 重量(1)
            record[79:80].decode("shift_jis", errors="replace").strip(),  # グレード(1)
            record[80:130].decode("shift_jis", errors="replace").strip(),  # レース名(50)
            record[130:132].decode("shift_jis", errors="replace").strip(),  # 頭数(2)
            record[132:140].decode("shift_jis", errors="replace").strip(),  # レース名略称(8)
            record[140:142].decode("shift_jis", errors="replace").strip(),  # 着順(2)
            record[142:143].decode("shift_jis", errors="replace").strip(),  # 異常区分(1)
            record[143:147].decode("shift_jis", errors="replace").strip(),  # タイム(4)
            record[147:150].decode("shift_jis", errors="replace").strip(),  # 斤量(3)
            record[150:162].decode("shift_jis", errors="replace").strip(),  # 騎手名(12)
            record[162:174].decode("shift_jis", errors="replace").strip(),  # 調教師名(12)
            record[174:180].decode("shift_jis", errors="replace").strip(),  # 確定単勝オッズ(6)
            record[180:182].decode("shift_jis", errors="replace").strip(),  # 確定単勝人気順位(2)
            record[182:185].decode("shift_jis", errors="replace").strip(),  # IDM(3)
            record[185:188].decode("shift_jis", errors="replace").strip(),  # 素点(3)
            record[188:191].decode("shift_jis", errors="replace").strip(),  # 馬場差(3)
            record[191:194].decode("shift_jis", errors="replace").strip(),  # ペース(3)
            record[194:197].decode("shift_jis", errors="replace").strip(),  # 出遅(3)
            record[197:200].decode("shift_jis", errors="replace").strip(),  # 位置取(3)
            record[200:203].decode("shift_jis", errors="replace").strip(),  # 不利(3)
            record[203:206].decode("shift_jis", errors="replace").strip(),  # 前不利(3)
            record[206:209].decode("shift_jis", errors="replace").strip(),  # 中不利(3)
            record[209:212].decode("shift_jis", errors="replace").strip(),  # 後不利(3)
            record[212:215].decode("shift_jis", errors="replace").strip(),  # レース(3)
            record[215:216].decode("shift_jis", errors="replace").strip(),  # コース取り(1)
            record[216:217].decode("shift_jis", errors="replace").strip(),  # 上昇度コード(1)
            record[217:219].decode("shift_jis", errors="replace").strip(),  # クラスコード(2)
            record[219:220].decode("shift_jis", errors="replace").strip(),  # 馬体コード(1)
            record[220:221].decode("shift_jis", errors="replace").strip(),  # 気配コード(1)
            record[221:222].decode("shift_jis", errors="replace").strip(),  # レースペース(1)
            record[222:223].decode("shift_jis", errors="replace").strip(),  # 馬ペース(1)
            record[223:228].decode("shift_jis", errors="replace").strip(),  # テン指数(5)
            record[228:233].decode("shift_jis", errors="replace").strip(),  # 上がり指数(5)
            record[233:238].decode("shift_jis", errors="replace").strip(),  # ペース指数(5)
            record[238:243].decode("shift_jis", errors="replace").strip(),  # レースP指数(5)
            record[243:255].decode("shift_jis", errors="replace").strip(),  # 1(2)着馬名(12)
            record[255:258].decode("shift_jis", errors="replace").strip(),  # 1(2)着タイム差(3)
            record[258:261].decode("shift_jis", errors="replace").strip(),  # 前3Fタイム(3)
            record[261:264].decode("shift_jis", errors="replace").strip(),  # 後3Fタイム(3)
            record[264:288].decode("shift_jis", errors="replace").strip(),  # 備考(24)
            record[288:290].decode("shift_jis", errors="replace").strip(),  # 予備(2)
            record[290:296].decode("shift_jis", errors="replace").strip(),  # 確定複勝オッズ下(6)
            record[296:302].decode("shift_jis", errors="replace").strip(),  # 10時単勝オッズ(6)
            record[302:308].decode("shift_jis", errors="replace").strip(),  # 10時複勝オッズ(6)
            record[308:310].decode("shift_jis", errors="replace").strip(),  # コーナー順位1(2)
            record[310:312].decode("shift_jis", errors="replace").strip(),  # コーナー順位2(2)
            record[312:314].decode("shift_jis", errors="replace").strip(),  # コーナー順位3(2)
            record[314:316].decode("shift_jis", errors="replace").strip(),  # コーナー順位4(2)
            record[316:319].decode("shift_jis", errors="replace").strip(),  # 前3F先頭差(3)
            record[319:322].decode("shift_jis", errors="replace").strip(),  # 後3F先頭差(3)
            record[322:327].decode("shift_jis", errors="replace").strip(),  # 騎手コード(5)
            record[327:332].decode("shift_jis", errors="replace").strip(),  # 調教師コード(5)
            record[332:335].decode("shift_jis", errors="replace").strip(),  # 馬体重(3)
            record[335:338].decode("shift_jis", errors="replace").strip(),  # 馬体重増減(3)
            JRA_MASTERS["天候コード"].get(record[338:339].decode("shift_jis", errors="replace").strip(), ""),  # 天候コード(1)
            record[339:340].decode("shift_jis", errors="replace").strip(),  # コース(1)
            record[340:341].decode("shift_jis", errors="replace").strip(),  # レース脚質(1)
            record[341:348].decode("shift_jis", errors="replace").strip(),  # 単勝(7)
            record[348:355].decode("shift_jis", errors="replace").strip(),  # 複勝(7)
            record[355:360].decode("shift_jis", errors="replace").strip(),  # 本賞金(5)
            record[360:365].decode("shift_jis", errors="replace").strip(),  # 収得賞金(5)
            record[365:367].decode("shift_jis", errors="replace").strip(),  # レースペース流れ(2)
            record[367:369].decode("shift_jis", errors="replace").strip(),  # 馬ペース流れ(2)
            record[369:370].decode("shift_jis", errors="replace").strip(),  # 4角コース取り(1)
            record[370:374].decode("shift_jis", errors="replace").strip(),  # 発走時間(4)
            record[355:360].decode("shift_jis", errors="replace").strip(),  # 1着賞金(本賞金と同じ値を使用)
            場名  # 追加: 場名も保存する
        ]
        return fields
    except Exception as e:
        print(f"⚠️ レコード {index} の処理中にエラーが発生しました: {str(e)}")
        return None

def format_sed_file(input_file, output_file, exclude_turf=False, turf_only=False):
    """
    SEDファイルを整形してCSVに変換します。
    
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
        "馬体重", "馬体重増減", "天候コード", "コース", "レース脚質", "単勝",
        "複勝", "本賞金", "収得賞金", "レースペース流れ", "馬ペース流れ",
        "4角コース取り", "発走時間", "1着賞金", "場名"  # 追加: 場名も保存
    ]
    
    # process_recordに芝コース処理オプションを渡す
    def process_record_with_turf_filter(record, index):
        return process_sed_record(record, index, exclude_turf, turf_only)
    
    return process_fixed_length_file(input_file, output_file, 374, headers, process_record_with_turf_filter)

def process_all_sed_files(exclude_turf=False, turf_only=False):
    """
    すべてのSEDファイルを処理します。
    
    Args:
        exclude_turf (bool): 芝コースを除外するかどうか
        turf_only (bool): 芝コースのみを処理するかどうか
    """
    # importフォルダとexportフォルダのパスを設定
    import_dir = Path("import/SED")
    export_dir = Path("export/SED")
    formatted_dir = Path("export/SED/formatted")  # formattedファイル用のディレクトリ
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(exist_ok=True, parents=True)
    formatted_dir.mkdir(exist_ok=True, parents=True)  # formattedディレクトリを作成
    
    # ファイルの検索
    target_files = []
    patterns = ["SED*.txt", "SED[0-9][0-9][0-9][0-9][0-9][0-9].txt"]
    
    # サブフォルダも含めて検索
    for pattern in patterns:
        target_files.extend(list(import_dir.rglob(pattern)))
    
    # 重複を除去してソート
    target_files = sorted(set(target_files))
    
    print(f"\n処理対象のSEDファイル数: {len(target_files)}")
    if exclude_turf:
        print("※ 芝コースのデータは除外されます")
    elif turf_only:
        print("※ 芝コースのデータのみが処理されます") 
        
    print("処理対象ファイル:")
    for file in target_files:
        print(f"- {file.name}")
    
    processed_files = []
    for target_file in target_files:
        # formattedファイルは別のフォルダに出力
        output_file = formatted_dir / f"{target_file.stem}_formatted.csv"
        print(f"処理中: {target_file}")
        processed = format_sed_file(str(target_file), str(output_file), exclude_turf=exclude_turf, turf_only=turf_only)
        
        if processed:
            processed_files.append((target_file, output_file, len(processed)))
    
    print("\n処理されたSEDファイル:")
    for input_file, output_file, count in processed_files:
        print(f"- {input_file.name} -> {output_file.name} ({count}件)")
    
    return len(processed_files) > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SEDファイルを処理します')
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', action='store_true', help='芝コースを除外する')
    track_group.add_argument('--turf-only', action='store_true', help='芝コースのみを処理する')
    args = parser.parse_args()
    
    process_all_sed_files(exclude_turf=args.exclude_turf, turf_only=args.turf_only) 
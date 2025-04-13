import csv
from pathlib import Path

def format_srb_file(input_file, output_file):
    record_length = 852  # 固定長レコードのバイト数

    # バイナリモードで読み込む（改行コードを無視）
    with open(input_file, "rb") as infile:
        content = infile.read()

    # 852バイトごとに分割
    records = [content[i:i + record_length] for i in range(0, len(content), record_length)]

    # 仕様書に従い、各レコードを正しく分割
    formatted_records = []
    for record in records:
        if len(record) != record_length:
            continue  # 852バイトに満たないレコードは無視

        # Shift_JIS でデコード（エラーは無視）
        record_text = record.decode("shift_jis", errors="ignore")

        # 各フィールドを適切に切り出し
        fields = [
            record_text[0:2],  # 場コード
            record_text[2:4].strip(),  # 年
            record_text[4:5].strip(),  # 回
            record_text[5:6].strip(),  # 日 (16進数)
            record_text[6:8].strip(),  # R
            record_text[8:62].strip(),  # ハロンタイム（18 * 3 = 54バイト）
            record_text[62:126].strip(),  # 1コーナー
            record_text[126:190].strip(),  # 2コーナー
            record_text[190:254].strip(),  # 3コーナー
            record_text[254:318].strip(),  # 4コーナー
            record_text[318:320].strip(),  # ペースアップ位置
            record_text[320:323].strip(),  # 1角
            record_text[323:326].strip(),  # 2角
            record_text[326:329].strip(),  # 向正
            record_text[329:332].strip(),  # 3角
            record_text[332:337].strip(),  # 4角
            record_text[337:342].strip(),  # 直線
            record[342:842].decode("shift_jis", errors="ignore").rstrip().ljust(500),  # レースコメント (500バイト, 空白埋め)
            record_text[842:850].strip(),  # 予備 (8バイト)
            record_text[850:852].strip()   # 改行 (2バイト)
        ]
        formatted_records.append(fields)

    # CSVに書き出し（UTF-8で出力）
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        header = [
            "場コード", "年", "回", "日", "Ｒ",
            "ハロンタイム", "１コーナー", "２コーナー", "３コーナー", "４コーナー",
            "ペースアップ位置", "１角", "２角", "向正", "３角", "４角", "直線",
            "レースコメント", "予備", "改行"
        ]
        writer.writerow(header)
        writer.writerows(formatted_records)

    print(f"✅ 整形されたファイルが {output_file} に保存されました。")

def process_all_srb_files():
    # importフォルダとexportフォルダのパスを設定
    import_dir = Path("import/SRB")
    export_dir = Path("export/SRB")
    
    # exportフォルダが存在しない場合は作成
    export_dir.mkdir(exist_ok=True)
    
    # importフォルダ内のすべてのSRBファイルを処理
    for srb_file in import_dir.glob("SRB*.txt"):
        output_file = export_dir / f"{srb_file.stem}_formatted.csv"
        print(f"処理中: {srb_file}")
        format_srb_file(str(srb_file), str(output_file))

if __name__ == "__main__":
    process_all_srb_files() 
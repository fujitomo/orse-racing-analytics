"""
TYB（直前情報データ）プロセッサー
直前オッズ、馬体重、印評価等の前処理と統合

TYBデータ仕様（第4b版対応）:
- 単勝・複勝オッズ（発走直前の市場予想）
- 馬体重・増減（コンディション指標）
- 印評価（直前総合印・パドック印・オッズ印）
- 気配情報（馬体コード・気配コード）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class TYBProcessor:
    """
    TYB（直前情報データ）プロセッサー
    JRDB TYB仕様第4b版に準拠
    """
    
    def __init__(self):
        self.processing_stats = {
            'total_files': 0,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'processing_errors': []
        }
        
    def parse_tyb_record(self, line: str) -> Optional[Dict[str, Any]]:
        """
        TYBレコードの解析（改良版）
        
        Args:
            line: TYBファイルの1行
            
        Returns:
            解析されたレコード辞書またはNone
        """
        if not line or len(line.strip()) < 20:
            return None
            
        try:
            # スペース区切りで分割
            fields = line.split()
            
            if len(fields) < 5:
                return None
                
            record = {}
            
            # レースキー（最初のフィールド）
            if len(fields) > 0 and len(fields[0]) >= 10:
                race_key = fields[0]
                record['レースキー'] = race_key
                
                # 年月日の抽出
                try:
                    record['年'] = int(race_key[0:2]) + 2000 if int(race_key[0:2]) < 50 else int(race_key[0:2]) + 1900
                    record['月'] = int(race_key[2:4])
                    record['日'] = int(race_key[4:6], 16)  # 16進数
                    record['場コード'] = int(race_key[6:8])
                    record['レースNo'] = int(race_key[8:10])
                except:
                    pass
            
            # 指数類（数値フィールド）
            numeric_fields = ['IDM', '騎手指数', '情報指数', '総合指数', '人気指数', '調教指数', '厩舎指数']
            field_idx = 1
            
            for i, field_name in enumerate(numeric_fields):
                if field_idx < len(fields):
                    try:
                        value = float(fields[field_idx])
                        record[field_name] = value if value != 0.0 else None
                    except:
                        record[field_name] = None
                    field_idx += 1
            
            # 複合フィールド（馬番+馬名+その他）を探す
            if field_idx < len(fields):
                complex_field = fields[field_idx]
                
                # 馬番（最初の3桁）
                if len(complex_field) >= 3 and complex_field[:3].isdigit():
                    record['馬番'] = int(complex_field[:3])
                
                # 馬名を抽出（文字化け部分をスキップ）
                if len(complex_field) > 3:
                    horse_name_part = complex_field[3:]
                    record['馬名'] = self._decode_horse_name(horse_name_part[:12])
                    
                    # 性齢・騎手コード等（後ろの部分）
                    if len(horse_name_part) > 12:
                        remaining = horse_name_part[12:]
                        # 性齢（2桁）
                        if len(remaining) >= 2:
                            sex_age = remaining[:2]
                            if len(sex_age) == 2 and sex_age[1].isdigit():
                                record['性別'] = sex_age[0]
                                record['年齢'] = int(sex_age[1])
                        
                        # 騎手コード（5桁）
                        if len(remaining) >= 7:
                            record['騎手コード'] = remaining[2:7]
                
                field_idx += 1
            
            # オッズ情報（後半のフィールドから数値を探す）
            odds_found = 0
            for i in range(field_idx, len(fields)):
                field = fields[i]
                try:
                    value = float(field)
                    if value > 1.0 and value < 1000.0:  # オッズらしい範囲
                        if odds_found == 0:
                            record['単勝オッズ'] = value
                            odds_found += 1
                        elif odds_found == 1:
                            record['複勝オッズ'] = value
                            odds_found += 1
                            break
                except:
                    continue
            
            # 馬体重（3桁の数字を探す）
            for field in fields:
                weight_match = re.search(r'(\d{3})', field)
                if weight_match:
                    weight = int(weight_match.group(1))
                    if 300 <= weight <= 600:  # 馬体重らしい範囲
                        record['馬体重'] = weight
                        
                        # 馬体重増減（符号付き）
                        change_match = re.search(r'([+-]\s*\d+)', field)
                        if change_match:
                            change_str = change_match.group(1).replace(' ', '')
                            try:
                                record['馬体重増減'] = int(change_str)
                            except:
                                pass
                        break
            
            # 印評価（1文字の数字）
            print_fields = []
            for field in fields:
                if len(field) == 1 and field.isdigit():
                    print_fields.append(field)
            
            if len(print_fields) >= 1:
                record['直前総合印'] = print_fields[0]
            if len(print_fields) >= 2:
                record['パドック印'] = print_fields[1]
            if len(print_fields) >= 3:
                record['オッズ印'] = print_fields[2]
            
            # 最低限のデータがあれば有効とする
            if len(record) >= 5:  # レースキー情報＋何らかのデータ
                return record
            else:
                return None
            
        except Exception as e:
            logger.debug(f"TYBレコード解析エラー: {str(e)[:30]}...")
            return None
    
    def _decode_horse_name(self, encoded_name: str) -> Optional[str]:
        """
        馬名の文字化け対応デコード
        """
        if not encoded_name:
            return None
            
        try:
            # Shift_JISでデコード試行
            decoded = encoded_name.encode('latin-1').decode('shift_jis', errors='ignore')
            # 制御文字削除
            decoded = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', decoded)
            return decoded.strip() if decoded.strip() else None
        except:
            # デコード失敗時は元の文字列をクリーニング
            cleaned = re.sub(r'[^\w\s]', '', encoded_name)
            return cleaned.strip() if cleaned.strip() else None
    
    def _parse_weight_change(self, weight_str: str) -> Optional[int]:
        """
        馬体重増減の解析（符号付き数値）
        """
        if not weight_str:
            return None
            
        try:
            # 符号を考慮して解析
            if weight_str.startswith('+'):
                return int(weight_str[1:])
            elif weight_str.startswith('-'):
                return -int(weight_str[1:])
            elif weight_str.isdigit():
                return int(weight_str)
            else:
                return None
        except:
            return None
    
    def process_tyb_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        TYBファイルの処理
        
        Args:
            file_path: TYBファイルのパス
            
        Returns:
            処理済みDataFrameまたはNone
        """
        logger.info(f"📄 TYBファイル処理開始: {file_path.name}")
        
        try:
            records = []
            valid_count = 0
            invalid_count = 0
            
            with open(file_path, 'r', encoding='latin-1') as f:
                for line_no, line in enumerate(f, 1):
                    record = self.parse_tyb_record(line)
                    if record:
                        records.append(record)
                        valid_count += 1
                    else:
                        invalid_count += 1
                        if invalid_count <= 5:  # 最初の5個のエラーのみログ
                            logger.warning(f"   行{line_no}: 解析失敗")
            
            if not records:
                logger.warning(f"   有効なレコードが見つかりません")
                return None
            
            df = pd.DataFrame(records)
            
            # データ型の最適化
            df = self._optimize_data_types(df)
            
            # 統計更新
            self.processing_stats['valid_records'] += valid_count
            self.processing_stats['invalid_records'] += invalid_count
            
            logger.info(f"   ✅ 処理完了: {valid_count}件有効, {invalid_count}件無効")
            
            return df
            
        except Exception as e:
            error_msg = f"{file_path.name}: {str(e)}"
            self.processing_stats['processing_errors'].append(error_msg)
            logger.error(f"   ❌ ファイル処理エラー: {error_msg}")
            return None
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型の最適化
        """
        try:
            # 整数型の最適化
            int_columns = ['年', '月', '日', '場コード', 'レースNo', '馬番', '年齢', '馬体重', '馬体重増減']
            for col in int_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
            # 浮動小数点型の最適化
            float_columns = ['IDM', '騎手指数', '情報指数', '総合指数', '人気指数', '調教指数', '厩舎指数', 
                           '斤量', '単勝オッズ', '複勝オッズ']
            for col in float_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
            
            # 文字列型のカテゴリ化
            categorical_columns = ['性別', '直前総合印', 'パドック印', 'オッズ印', '馬体コード', '気配コード']
            for col in categorical_columns:
                if col in df.columns and df[col].notna().any():
                    df[col] = df[col].astype('category')
            
        except Exception as e:
            logger.warning(f"データ型最適化エラー: {str(e)}")
        
        return df

def process_all_tyb_files(exclude_turf: bool = False, turf_only: bool = False) -> bool:
    """
    すべてのTYBファイルを処理
    
    Args:
        exclude_turf: 芝コースを除外するかどうか
        turf_only: 芝コースのみを処理するかどうか
        
    Returns:
        処理成功可否
    """
    logger.info("🏃 TYBデータ（直前情報）の一括処理開始")
    
    processor = TYBProcessor()
    
    # 入力・出力ディレクトリ
    input_dir = Path('import/TYB')
    output_dir = Path('export/TYB/formatted')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"❌ TYBディレクトリが見つかりません: {input_dir}")
        return False
    
    # TYBファイル一覧取得
    tyb_files = list(input_dir.glob('TYB*.txt'))
    
    if not tyb_files:
        logger.warning(f"⚠️ TYBファイルが見つかりません: {input_dir}")
        return False
    
    logger.info(f"📁 対象ファイル数: {len(tyb_files)}件")
    
    processed_count = 0
    error_count = 0
    
    for tyb_file in sorted(tyb_files):
        try:
            # ファイル処理
            df = processor.process_tyb_file(tyb_file)
            
            if df is not None and len(df) > 0:
                # 出力ファイル名生成
                output_file = output_dir / f"{tyb_file.stem}_formatted.csv"
                
                # CSV出力
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"   進捗: {processed_count}/{len(tyb_files)}件完了")
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            processor.processing_stats['processing_errors'].append(f"{tyb_file.name}: {str(e)}")
            logger.error(f"   ❌ {tyb_file.name}: {str(e)}")
    
    # 処理統計
    logger.info("📊 TYB処理統計:")
    logger.info(f"   📄 処理ファイル数: {processed_count}/{len(tyb_files)}")
    logger.info(f"   ✅ 有効レコード: {processor.processing_stats['valid_records']:,}件")
    logger.info(f"   ❌ 無効レコード: {processor.processing_stats['invalid_records']:,}件")
    logger.info(f"   🚫 エラーファイル: {error_count}件")
    
    if processor.processing_stats['processing_errors']:
        logger.warning("⚠️ 処理エラー一覧:")
        for error in processor.processing_stats['processing_errors'][:10]:  # 最初の10件のみ
            logger.warning(f"   • {error}")
    
    processor.processing_stats['total_files'] = len(tyb_files)
    processor.processing_stats['total_records'] = processor.processing_stats['valid_records'] + processor.processing_stats['invalid_records']
    
    success_rate = (processed_count / len(tyb_files)) * 100 if tyb_files else 0
    logger.info(f"🎯 TYB処理完了: 成功率{success_rate:.1f}%")
    
    return success_rate > 80  # 80%以上の成功率で成功とみなす

if __name__ == "__main__":
    # テスト実行
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = process_all_tyb_files()
    print(f"TYB処理結果: {'成功' if success else '失敗'}")

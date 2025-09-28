"""
TYB（直前情報データ）プロセッサー - 簡易版
単純な解析ロジックでTYBファイルを処理
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleTYBProcessor:
    """
    TYB（直前情報データ）プロセッサー - 簡易版
    基本的なフィールドのみ抽出
    """
    
    def __init__(self):
        self.processing_stats = {
            'total_files': 0,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'processing_errors': []
        }
        
    def parse_tyb_record_simple(self, line: str) -> Optional[Dict[str, Any]]:
        """
        TYBレコードの簡易解析
        最小限のフィールドのみ抽出
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
            
            # オッズ情報（後半のフィールドから数値を探す）
            for i in range(field_idx, len(fields)):
                field = fields[i]
                try:
                    value = float(field)
                    if value > 1.0 and value < 1000.0:  # オッズらしい範囲
                        if '単勝オッズ' not in record:
                            record['単勝オッズ'] = value
                        elif '複勝オッズ' not in record:
                            record['複勝オッズ'] = value
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
                        break
            
            # 最低限のデータがあれば有効とする
            if len(record) >= 3:
                return record
            else:
                return None
            
        except Exception as e:
            logger.debug(f"簡易解析エラー: {str(e)[:30]}...")
            return None
    
    def process_tyb_file_simple(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        TYBファイルの簡易処理
        """
        logger.info(f"📄 TYBファイル簡易処理: {file_path.name}")
        
        try:
            records = []
            valid_count = 0
            invalid_count = 0
            
            with open(file_path, 'r', encoding='latin-1') as f:
                for line_no, line in enumerate(f, 1):
                    if line_no > 100:  # 最初の100行のみテスト
                        break
                        
                    record = self.parse_tyb_record_simple(line)
                    if record:
                        records.append(record)
                        valid_count += 1
                    else:
                        invalid_count += 1
            
            if not records:
                logger.warning(f"   有効なレコードが見つかりません")
                return None
            
            df = pd.DataFrame(records)
            
            logger.info(f"   ✅ 簡易処理完了: {valid_count}件有効, {invalid_count}件無効")
            
            return df
            
        except Exception as e:
            logger.error(f"   ❌ ファイル処理エラー: {str(e)}")
            return None

def test_simple_tyb():
    """簡易TYB処理のテスト"""
    logger.info("🏃 簡易TYB処理テスト開始")
    
    processor = SimpleTYBProcessor()
    
    # テストファイル
    test_file = Path('import/TYB/TYB250427.txt')
    
    if not test_file.exists():
        logger.error(f"テストファイルが見つかりません: {test_file}")
        return False
    
    # ファイル処理
    df = processor.process_tyb_file_simple(test_file)
    
    if df is not None and len(df) > 0:
        logger.info(f"✅ 簡易処理成功: {len(df)}件のレコード")
        logger.info(f"カラム: {list(df.columns)}")
        
        # サンプル表示
        if len(df) > 0:
            logger.info("サンプルデータ:")
            print(df.head(3).to_string())
        
        return True
    else:
        logger.error("❌ 簡易処理失敗")
        return False

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = test_simple_tyb()
    print(f"簡易TYB処理結果: {'成功' if success else '失敗'}")



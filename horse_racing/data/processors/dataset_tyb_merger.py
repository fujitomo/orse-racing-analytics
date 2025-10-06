"""
データセット・TYB統合プロセッサー
SEDベースのデータセットにTYBの直前情報を結合する
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetTYBMerger:
    """
    SEDデータセットとTYBデータの統合プロセッサー
    """
    
    def __init__(self):
        self.processing_stats = {
            'total_datasets': 0,
            'successful_merges': 0,
            'failed_merges': 0,
            'processing_errors': []
        }
        
    def create_race_key_from_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SEDデータセットからレースキーを生成
        
        Args:
            df: SEDデータセット
            
        Returns:
            レースキー付きのデータフレーム
        """
        try:
            # 年、月、日、場コード、Rからレースキーを生成
            # TYBのレースキー形式: YYMMDDCCR (YY=年, MM=月, DD=日, CC=場コード, R=レースNo)
            
            df = df.copy()
            
            # 日付情報の取得（年月日列から）
            if '年月日' in df.columns:
                # 年月日から日付情報を抽出
                df['年月日_str'] = df['年月日'].astype(str)
                df['年_from_date'] = df['年月日_str'].str[:4].astype(int)
                df['月_from_date'] = df['年月日_str'].str[4:6].astype(int)
                df['日_from_date'] = df['年月日_str'].str[6:8].astype(int)
            elif all(col in df.columns for col in ['年', '月', '日']):
                # 既に年月日列がある場合
                df['年_from_date'] = df['年']
                df['月_from_date'] = df['月']
                df['日_from_date'] = df['日']
            else:
                logger.warning("日付情報が見つかりません")
                return df
            
            # レースキーの生成
            # YY: 年の下2桁
            df['年_2桁'] = (df['年_from_date'] % 100).astype(str).str.zfill(2)
            
            # MM: 月（16進数）
            df['月_16進'] = df['月_from_date'].apply(lambda x: f"{x:02X}")
            
            # DD: 日（16進数）
            df['日_16進'] = df['日_from_date'].apply(lambda x: f"{x:02X}")
            
            # CC: 場コード（2桁）
            df['場コード_2桁'] = df['場コード'].astype(str).str.zfill(2)
            
            # R: レースNo（2桁）
            df['レースNo_2桁'] = df['R'].astype(str).str.zfill(2)
            
            # レースキーの結合
            df['レースキー'] = (df['年_2桁'] + df['月_16進'] + df['日_16進'] + 
                              df['場コード_2桁'] + df['レースNo_2桁'])
            
            # 中間列を削除
            columns_to_drop = ['年_2桁', '月_16進', '日_16進', '場コード_2桁', 'レースNo_2桁',
                              '年_from_date', '月_from_date', '日_from_date']
            if '年月日_str' in df.columns:
                columns_to_drop.append('年月日_str')
                
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            logger.debug(f"レースキー生成完了: {len(df)}件")
            return df
            
        except Exception as e:
            logger.error(f"レースキー生成エラー: {str(e)}")
            return df
    
    def merge_dataset_with_tyb(self, dataset_path: Path, tyb_path: Path) -> Optional[pd.DataFrame]:
        """
        SEDデータセットとTYBデータを結合（改良版）
        
        Args:
            dataset_path: SEDデータセットのパス
            tyb_path: TYBデータのパス
            
        Returns:
            結合されたデータフレーム
        """
        try:
            logger.info(f"📊 データセット統合: {dataset_path.name} + {tyb_path.name}")
            
            # SEDデータセット読み込み
            sed_df = pd.read_csv(dataset_path, encoding='utf-8')
            logger.debug(f"SEDデータ読み込み: {len(sed_df)}件")
            
            # TYBデータ読み込み
            tyb_df = pd.read_csv(tyb_path, encoding='utf-8')
            logger.debug(f"TYBデータ読み込み: {len(tyb_df)}件")
            
            # ファイル名から日付を抽出してフィルタリング
            target_date = self.extract_date_from_filename(dataset_path)
            if target_date:
                # SEDデータを対象日付でフィルタ
                sed_df = sed_df[sed_df['年月日'].astype(str) == target_date].copy()
                logger.debug(f"日付フィルタ後: {len(sed_df)}件")
            
            # TYB列名の接頭辞を追加（重複回避）
            tyb_columns_rename = {}
            for col in tyb_df.columns:
                if col not in ['場コード', 'レースNo', '馬番']:
                    tyb_columns_rename[col] = f'TYB_{col}'
            
            tyb_df = tyb_df.rename(columns=tyb_columns_rename)
            
            # 競走経験質指数（REQI）での統合（場コード + レースNo）
            merge_columns = []
            
            # 場コード
            if '場コード' in sed_df.columns and '場コード' in tyb_df.columns:
                merge_columns.append('場コード')
                
            # レースNo（SEDの'R'列 vs TYBの'レースNo'列）
            if 'R' in sed_df.columns and 'レースNo' in tyb_df.columns:
                sed_df['レースNo'] = sed_df['R']
                merge_columns.append('レースNo')
            
            if merge_columns:
                # TYBデータを競走経験質指数（REQI）で集約
                tyb_race_agg = self.aggregate_tyb_by_race(tyb_df)
                
                # データ結合
                merged_df = pd.merge(
                    sed_df, tyb_race_agg,
                    on=merge_columns,
                    how='left'
                )
                
                # TYB結合率の計算
                tyb_rate = 0
                if 'TYB_レース頭数' in merged_df.columns:
                    tyb_rate = merged_df['TYB_レース頭数'].notna().mean()
                elif 'TYB_IDM_平均' in merged_df.columns:
                    tyb_rate = merged_df['TYB_IDM_平均'].notna().mean()
                
                logger.info(f"   ✅ 結合完了: {len(merged_df)}件 (TYBレース結合率: {tyb_rate:.1%})")
                
                return merged_df
                
            else:
                logger.warning("   ❌ 結合可能な列が見つかりません")
                return sed_df
                
        except Exception as e:
            logger.error(f"   ❌ データセット統合エラー: {str(e)}")
            return None
    
    def extract_date_from_filename(self, file_path: Path) -> Optional[str]:
        """
        ファイル名から日付を抽出
        
        Args:
            file_path: ファイルパス
            
        Returns:
            YYYYMMDD形式の日付文字列
        """
        try:
            # SEDファイル名から日付抽出 (例: SED250420_formatted_dataset.csv -> 250420)
            match = re.search(r'SED(\d{6})', file_path.name)
            if match:
                date_str = match.group(1)  # 250420
                
                # YYMMDD -> YYYYMMDD変換
                if len(date_str) == 6:
                    年_2桁 = date_str[:2]  # 25
                    月_2桁 = date_str[2:4]  # 04  
                    日_2桁 = date_str[4:6]  # 20
                    
                    年_数値 = int(年_2桁)
                    西暦年 = 2000 + 年_数値 if 年_数値 < 50 else 1900 + 年_数値
                    
                    return f"{西暦年:04d}{月_2桁}{日_2桁}"
                    
        except Exception as e:
            logger.debug(f"日付抽出エラー: {str(e)}")
            
        return None
    
    def aggregate_tyb_by_race(self, tyb_df: pd.DataFrame) -> pd.DataFrame:
        """
        TYBデータを競走経験質指数（REQI）で集約
        
        Args:
            tyb_df: TYBデータフレーム
            
        Returns:
            競走経験質指数（REQI）で集約されたTYBデータ
        """
        try:
            # レース単位での集約
            race_groups = tyb_df.groupby(['場コード', 'レースNo'])
            
            # 集約辞書
            agg_dict = {}
            
            # 数値列の集約
            numeric_columns = ['TYB_IDM', 'TYB_騎手指数', 'TYB_情報指数', 'TYB_総合指数', 
                              'TYB_人気指数', 'TYB_調教指数', 'TYB_厩舎指数', 'TYB_馬体重']
            
            for col in numeric_columns:
                if col in tyb_df.columns:
                    agg_dict[f'{col}_平均'] = (col, lambda x: x.mean() if x.notna().any() else None)
                    agg_dict[f'{col}_最大'] = (col, lambda x: x.max() if x.notna().any() else None)
                    agg_dict[f'{col}_最小'] = (col, lambda x: x.min() if x.notna().any() else None)
            
            # オッズ関連の集約
            odds_columns = ['TYB_単勝オッズ', 'TYB_複勝オッズ']
            for col in odds_columns:
                if col in tyb_df.columns:
                    agg_dict[f'{col}_平均'] = (col, lambda x: x.mean() if x.notna().any() else None)
                    agg_dict[f'{col}_最低'] = (col, lambda x: x.min() if x.notna().any() else None)
            
            # 馬体重関連
            if 'TYB_馬体重' in tyb_df.columns:
                agg_dict['TYB_馬体重_平均'] = ('TYB_馬体重', lambda x: x.mean() if x.notna().any() else None)
                
            if 'TYB_馬体重増減' in tyb_df.columns:
                agg_dict['TYB_馬体重増減_平均'] = ('TYB_馬体重増減', lambda x: x.mean() if x.notna().any() else None)
            
            # レース基本情報
            agg_dict['TYB_レース頭数'] = ('TYB_IDM', 'count')
            agg_dict['TYB_データ数'] = ('TYB_IDM', lambda x: x.notna().sum())
            
            # 印評価の最頻値
            if 'TYB_直前総合印' in tyb_df.columns:
                agg_dict['TYB_直前総合印_最頻'] = ('TYB_直前総合印', lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            
            # 集約実行
            if agg_dict:
                race_agg = race_groups.agg(**agg_dict).reset_index()
                
                # カラム名を整理
                race_agg.columns = ['場コード', 'レースNo'] + [col for col in race_agg.columns if col not in ['場コード', 'レースNo']]
                
                logger.debug(f"TYBレース集約完了: {len(race_agg)}レース")
                
                return race_agg
            else:
                # 最低限の集約
                race_agg = race_groups.size().reset_index(name='TYB_レース頭数')
                return race_agg
                
        except Exception as e:
            logger.error(f"TYBレース集約エラー: {str(e)}")
            # 空のDataFrameを返す
            return pd.DataFrame(columns=['場コード', 'レースNo', 'TYB_レース頭数'])
    
    def find_matching_tyb_file(self, dataset_file: Path, tyb_dir: Path) -> Optional[Path]:
        """
        データセットファイルに対応するTYBファイルを検索
        
        Args:
            dataset_file: SEDデータセットファイル
            tyb_dir: TYBディレクトリ
            
        Returns:
            対応するTYBファイルのパス
        """
        try:
            # SEDファイル名から日付を抽出 (例: SED250420_formatted_dataset.csv -> 250420)
            match = re.search(r'SED(\d{6})_formatted_dataset\.csv', dataset_file.name)
            if not match:
                return None
                
            date_str = match.group(1)
            
            # 対応するTYBファイルを検索 (例: TYB250420_formatted.csv)
            tyb_filename = f"TYB{date_str}_formatted.csv"
            tyb_path = tyb_dir / tyb_filename
            
            if tyb_path.exists():
                return tyb_path
            else:
                logger.debug(f"対応するTYBファイルが見つかりません: {tyb_filename}")
                return None
                
        except Exception as e:
            logger.debug(f"TYBファイル検索エラー: {str(e)}")
            return None
    
    def process_all_datasets(self, dataset_dir: Path = None, tyb_dir: Path = None, output_dir: Path = None):
        """
        全データセットにTYBデータを統合
        
        Args:
            dataset_dir: データセットディレクトリ
            tyb_dir: TYBディレクトリ  
            output_dir: 出力ディレクトリ
        """
        # デフォルトパスの設定
        if dataset_dir is None:
            dataset_dir = Path('export/dataset')
        if tyb_dir is None:
            tyb_dir = Path('export/TYB/formatted')
        if output_dir is None:
            output_dir = Path('export/dataset_with_tyb')
            
        logger.info("🔗 データセット・TYB統合処理を開始します")
        logger.info(f"   📂 データセット: {dataset_dir}")
        logger.info(f"   📂 TYBデータ: {tyb_dir}")
        logger.info(f"   📂 出力先: {output_dir}")
        
        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # データセットファイル一覧取得
        dataset_files = list(dataset_dir.glob('SED*_formatted_dataset.csv'))
        
        if not dataset_files:
            logger.error("データセットファイルが見つかりません")
            return
            
        logger.info(f"   📄 処理対象: {len(dataset_files)}ファイル")
        
        self.processing_stats['total_datasets'] = len(dataset_files)
        
        # 各データセットを処理
        for i, dataset_file in enumerate(dataset_files, 1):
            try:
                # 対応するTYBファイルを検索
                tyb_file = self.find_matching_tyb_file(dataset_file, tyb_dir)
                
                if tyb_file is None:
                    logger.debug(f"   [{i:3d}/{len(dataset_files)}] TYBファイルなし: {dataset_file.name}")
                    # TYBデータなしでもコピー
                    sed_df = pd.read_csv(dataset_file, encoding='utf-8')
                    output_path = output_dir / f"{dataset_file.stem}_with_tyb.csv"
                    sed_df.to_csv(output_path, index=False, encoding='utf-8')
                    continue
                
                # データセット統合
                merged_df = self.merge_dataset_with_tyb(dataset_file, tyb_file)
                
                if merged_df is not None:
                    # 出力ファイル名
                    output_filename = f"{dataset_file.stem}_with_tyb.csv"
                    output_path = output_dir / output_filename
                    
                    # CSV出力
                    merged_df.to_csv(output_path, index=False, encoding='utf-8')
                    
                    self.processing_stats['successful_merges'] += 1
                    
                    if i % 50 == 0:
                        logger.info(f"   [{i:3d}/{len(dataset_files)}] 処理完了: {self.processing_stats['successful_merges']}件成功")
                        
                else:
                    self.processing_stats['failed_merges'] += 1
                    logger.warning(f"   [{i:3d}/{len(dataset_files)}] 統合失敗: {dataset_file.name}")
                    
            except Exception as e:
                self.processing_stats['failed_merges'] += 1
                self.processing_stats['processing_errors'].append(f"{dataset_file.name}: {str(e)}")
                logger.error(f"   [{i:3d}/{len(dataset_files)}] 処理エラー: {dataset_file.name} - {str(e)}")
        
        # 処理結果サマリー
        logger.info("\n" + "="*60)
        logger.info("📊 データセット・TYB統合結果")
        logger.info("="*60)
        logger.info(f"   総ファイル数: {self.processing_stats['total_datasets']}")
        logger.info(f"   成功: {self.processing_stats['successful_merges']}")
        logger.info(f"   失敗: {self.processing_stats['failed_merges']}")
        logger.info(f"   成功率: {self.processing_stats['successful_merges']/self.processing_stats['total_datasets']:.1%}")
        
        if self.processing_stats['processing_errors']:
            logger.info(f"   エラー詳細: {len(self.processing_stats['processing_errors'])}件")
            for error in self.processing_stats['processing_errors'][:5]:
                logger.info(f"     - {error}")

def main():
    """メイン実行"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    merger = DatasetTYBMerger()
    merger.process_all_datasets()

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import os
import logging
from typing import Union
from datetime import datetime


logging.basicConfig(level=logging.INFO)  # INFOレベル以上を有効にする
logger = logging.getLogger(__name__)

class RaceDataLoader:
    """レースデータ読み込みクラス"""
    
    def __init__(self, input_path: Union[str, Path], encoding: str = "utf-8-sig"):
        """
        レースデータローダーの初期化
        
        Args:
            input_path: 入力パス（ファイル、ディレクトリ、またはzipファイル）
            encoding: ファイルのエンコーディング
        """
        self.input_path = Path(input_path)
        self.encoding = encoding

    def is_sed_file(self, file_path: Union[str, Path]) -> bool:
        """
        ファイルがSEDファイルかどうかを判定
        
        Args:
            file_path: 判定するファイルパス
            
        Returns:
            bool: SEDファイルの場合True
        """
        # パスオブジェクトの場合は名前を取得、文字列の場合はそのまま使用
        name = file_path.name if isinstance(file_path, Path) else os.path.basename(str(file_path))
        return name.upper().startswith('SED')

    def load(self) -> pd.DataFrame:
        """
        データを読み込む
        
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"パスが存在しません: {self.input_path}")

        if self.input_path.is_file():
            if self.input_path.suffix.lower() == '.zip':
                return self._load_zip_file(self.input_path)
            elif self.is_sed_file(self.input_path):
                return self._load_single_file(self.input_path)
            else:
                raise ValueError(f"SEDファイルではありません: {self.input_path}")
        else:
            return self._load_directory(self.input_path)

    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        単一のファイルを読み込む
        
        Args:
            file_path: ファイルパス
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        if file_path.suffix.lower() not in ['.csv', '.txt']:
            raise ValueError(f"サポートされていないファイル形式です: {file_path}")

        try:
            df = pd.read_csv(file_path,
                           encoding="utf-8-sig",
                           low_memory=False)  # メモリ使用量の警告を抑制
            return df
        except UnicodeDecodeError:
            # エンコーディングエラーの場合、cp932で再試行
            try:
                df = pd.read_csv(file_path, 
                               encoding='cp932',
                               low_memory=False)  # メモリ使用量の警告を抑制
                return df
            except Exception as e:
                raise ValueError(f"ファイルの読み込みに失敗しました: {file_path}, {str(e)}")

    def _load_zip_file(self, zip_path: Path) -> pd.DataFrame:
        """
        ZIPファイルからCSVを読み込む
        
        Args:
            zip_path: ZIPファイルのパス
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 最初にZIP内のファイル一覧を取得してSEDファイルをフィルタリング
                sed_files = [f for f in zip_ref.namelist() if f.endswith('.csv') and self.is_sed_file(f)]
                if not sed_files:
                    logger.debug(f"ZIPファイル内にSEDファイルが見つかりません: {zip_path}")
                    raise ValueError(f"ZIPファイル内にSEDファイルが見つかりません: {zip_path}")
                
                # SEDファイルのみを展開
                for file_name in sed_files:
                    zip_ref.extract(file_name, temp_dir)
                    file_path = Path(temp_dir) / file_name
                    try:
                        df = self._load_single_file(file_path)
                        dfs.append(df)
                        logger.info(f"ZIPファイルから読み込み: {file_name} ({len(df)} レコード)")
                    except Exception as e:
                        logger.warning(f"ファイルのスキップ: {file_name} - {str(e)}")
                        continue

        if not dfs:
            raise ValueError(f"有効なSEDデータが見つかりませんでした: {zip_path}")

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"ZIPファイルを読み込みました: {zip_path} (合計 {len(result)} レコード)")
        return result

    def _load_directory(self, dir_path: Path) -> pd.DataFrame:
        """
        ディレクトリからデータを読み込む
        
        Args:
            dir_path: ディレクトリパス
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        dfs = []
        sed_files = []
        
        # まず全てのSEDファイルを列挙（.csvと.txt）
        for pattern in ['*.csv', '*.txt']:
            for file_path in dir_path.rglob(pattern):
                if file_path.is_file() and self.is_sed_file(file_path):
                    sed_files.append(file_path)
        
        logger.info(f"SEDファイルが {len(sed_files)} 件見つかりました")
        
        # 見つかったファイルを処理
        for file_path in sed_files:
            try:
                df = self._load_single_file(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"ファイルのスキップ: {file_path.relative_to(dir_path)} - {str(e)}")
                continue

        if not dfs:
            raise ValueError(f"有効なSEDデータが見つかりませんでした: {dir_path}")

        result = pd.concat(dfs, ignore_index=True)
        return result

    @staticmethod
    def get_date_str(path: Path) -> str:
        """
        パスから日付文字列を取得
        
        Args:
            path: ファイルまたはディレクトリのパス
            
        Returns:
            str: 日付文字列（YYYYMMDD）
        """
        if path.is_file():
            # ファイル名から日付を抽出
            date_str = path.stem[-6:]  # YYMMDD形式を想定
            if date_str.isdigit():
                return f"20{date_str}"  # 20XX年代と仮定
        return datetime.now().strftime("%Y%m%d") 
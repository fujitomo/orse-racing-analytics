import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

class OutputUtils:
    """
    ファイル出力関連のユーティリティクラス
    """
    
    @staticmethod
    def ensure_dir(file_path):
        """
        ファイルパスからディレクトリを抽出し、存在しない場合は作成
        
        Args:
            file_path (str): ファイルパス
        """
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"ディレクトリを作成しました: {dir_path}")
    
    @staticmethod
    def export_to_csv(data, file_path, index=False):
        """
        データをCSVファイルに出力
        
        Args:
            data (DataFrame): 出力するデータフレーム
            file_path (str): 出力先ファイルパス
            index (bool, optional): インデックスを含めるかどうか
        """
        OutputUtils.ensure_dir(file_path)
        
        # DataFrameでない場合はDataFrameに変換
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                # 辞書型の場合は1行のDataFrameに変換
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                # リスト型の場合はDataFrameに変換
                data = pd.DataFrame(data)
        
        # CSVに出力
        data.to_csv(file_path, index=index, encoding='utf-8-sig')
        print(f"CSVファイルに出力しました: {file_path}")
    
    @staticmethod
    def export_to_json(data, file_path, indent=2):
        """
        データをJSONファイルに出力
        
        Args:
            data (dict/list): 出力するデータ
            file_path (str): 出力先ファイルパス
            indent (int, optional): インデント
        """
        OutputUtils.ensure_dir(file_path)
        
        # NumPy型をPython標準型に変換する関数
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, pd.DataFrame):
                return convert_to_serializable(obj.to_dict(orient='records'))
            elif isinstance(obj, pd.Series):
                return convert_to_serializable(obj.to_dict())
            elif hasattr(obj, 'tolist'):  # numpy配列やスカラー値
                return obj.tolist()
            elif hasattr(obj, 'item'):    # numpy.int64, numpy.float64など
                return obj.item()
            else:
                return obj
        
        # データを変換
        serializable_data = convert_to_serializable(data)
        
        # JSONに出力
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=indent)
        
        print(f"JSONファイルに出力しました: {file_path}")
    
    @staticmethod
    def export_figure(fig, file_path, dpi=300):
        """
        Matplotlibの図をファイルに出力
        
        Args:
            fig (Figure): Matplotlibの図オブジェクト
            file_path (str): 出力先ファイルパス
            dpi (int, optional): 解像度
        """
        OutputUtils.ensure_dir(file_path)
        
        # ファイル拡張子に基づいて画像形式を自動判別
        ext = os.path.splitext(file_path)[1].lower()
        
        # 画像として保存
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        print(f"画像ファイル({ext})に出力しました: {file_path}")
    
    @staticmethod
    def get_output_format(file_path):
        """
        ファイルパスから出力形式を取得
        
        Args:
            file_path (str): ファイルパス
            
        Returns:
            str: 出力形式
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.csv']:
            return 'csv'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.tif', '.tiff']:
            return 'image'
        else:
            # デフォルトはCSV
            return 'csv' 
import logging
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class OutputManager:
    """分析結果の出力管理クラス"""

    def __init__(self, output_dir: str, date_str: str = None):
        """
        初期化
        
        Parameters:
        -----------
        output_dir : str
            出力ディレクトリのパス
        date_str : str, optional
            日付文字列（YYYYMMDD形式）。指定がない場合は現在の日付を使用
        """
        self.output_dir = Path(output_dir)
        self.date_str = date_str or datetime.now().strftime('%Y%m%d')
        self._setup_logger()
        self._create_output_dirs()

    def _setup_logger(self) -> None:
        """ロガーの設定"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.FileHandler(
                self.output_dir / f"analysis_{self.date_str}.log",
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _create_output_dirs(self) -> None:
        """出力ディレクトリの作成"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._plots_dir = self.output_dir / "plots"
            self._data_dir = self.output_dir / "data"
            self._plots_dir.mkdir(exist_ok=True)
            self._data_dir.mkdir(exist_ok=True)
        except Exception as e:
            self.logger.error(f"出力ディレクトリの作成中にエラーが発生しました: {str(e)}")
            raise

    def save_analysis_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        分析結果をJSONファイルとして保存
        
        Parameters:
        -----------
        results : Dict[str, Any]
            保存する分析結果
        filename : str
            保存するファイル名
        """
        try:
            output_path = self._data_dir / f"{filename}_{self.date_str}.json"
            
            # DataFrameとモデルをJSON形式に変換可能な形式に変換
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = self._convert_dict_to_serializable(value)
                elif isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict(orient='records')
                else:
                    serializable_results[key] = self._convert_to_serializable(value)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分析結果を保存しました: {output_path}")
        except Exception as e:
            self.logger.error(f"分析結果の保存中にエラーが発生しました: {str(e)}")
            raise

    def _convert_dict_to_serializable(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """辞書をJSON形式に変換可能な形式に変換"""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._convert_dict_to_serializable(v)
            else:
                result[k] = self._convert_to_serializable(v)
        return result

    def _convert_to_serializable(self, obj: Any) -> Any:
        """オブジェクトをJSON形式に変換可能な形式に変換"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (LinearRegression, LogisticRegression)):
            return {
                'type': obj.__class__.__name__,
                'params': obj.get_params(),
                'coef': obj.coef_.tolist() if hasattr(obj, 'coef_') else None,
                'intercept': float(obj.intercept_) if hasattr(obj, 'intercept_') else None
            }
        elif isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {
                'type': obj.__class__.__name__,
                'attributes': {k: self._convert_to_serializable(v) 
                             for k, v in obj.__dict__.items() 
                             if not k.startswith('_')}
            }
        return obj

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        データフレームをCSVファイルとして保存
        
        Parameters:
        -----------
        df : pd.DataFrame
            保存するデータフレーム
        filename : str
            保存するファイル名
        """
        try:
            output_path = self._data_dir / f"{filename}_{self.date_str}.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            self.logger.info(f"データフレームを保存しました: {output_path}")
        except Exception as e:
            self.logger.error(f"データフレームの保存中にエラーが発生しました: {str(e)}")
            raise

    def log_analysis_start(self, analyzer_name: str, config: Dict[str, Any]) -> None:
        """
        分析開始のログを出力
        
        Parameters:
        -----------
        analyzer_name : str
            分析器の名前
        config : Dict[str, Any]
            分析設定
        """
        self.logger.info(f"{analyzer_name}の分析を開始します...")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")

    def log_analysis_complete(self, analyzer_name: str) -> None:
        """
        分析完了のログを出力
        
        Parameters:
        -----------
        analyzer_name : str
            分析器の名前
        """
        self.logger.info(f"{analyzer_name}の分析が完了しました。")
        self.logger.info(f"結果は {self.output_dir} に保存されました。")

    @property
    def plots_directory(self) -> Path:
        """プロット出力用のディレクトリパスを取得"""
        return self._plots_dir

    @property
    def data_directory(self) -> Path:
        """データ出力用のディレクトリパスを取得"""
        return self._data_dir 
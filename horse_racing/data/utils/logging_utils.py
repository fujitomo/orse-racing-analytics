"""
ログ設定ユーティリティ
"""
import logging
from typing import Optional


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """実務レベルのログ設定を初期化する。

    Args:
        log_level (str): ログレベル（例: ``INFO``, ``DEBUG``）。
        log_file (str, optional): ログ出力ファイルパス。``None`` の場合はコンソールのみ。
    """
    # シンプルな設定
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


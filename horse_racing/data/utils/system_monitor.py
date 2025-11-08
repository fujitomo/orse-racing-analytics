"""
システム監視クラス
"""
import logging
import time

logger = logging.getLogger(__name__)


class SystemMonitor:
    """システム監視クラス（簡略版）"""
    
    def __init__(self):
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
    
    def log_system_status(self, stage_name: str):
        """システム状態をログに出力します。

        Args:
            stage_name (str): 出力対象の処理段階名。
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        logger.info(f"💻 [{stage_name}] システム状態:")
        logger.info(f"   ⏱️ 経過時間: {elapsed_time:.1f}秒")


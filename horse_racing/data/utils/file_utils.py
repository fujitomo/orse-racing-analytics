"""
ファイル操作ユーティリティ
"""
import json
import logging
from pathlib import Path

from ..processors.data_quality_checker import DataQualityChecker

logger = logging.getLogger(__name__)


def ensure_export_dirs():
    """出力用ディレクトリの存在確認と作成を行う。"""
    logger = logging.getLogger(__name__)
    
    dirs = [
        'export/BAC', 
        'export/SRB', 
        'export/SED', 
        'export/dataset',          # 実際のSED+SRB統合データ出力先
        'export/quality_reports',     # データ品質レポート保存用
        'export/logs'                 # ログ保存用
    ]
    
    created_dirs = []
    
    for dir_path in dirs:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
            logger.info(f"📁 ディレクトリ作成: {dir_path}")
    
    if created_dirs:
        logger.info(f"✅ {len(created_dirs)}個のディレクトリを作成しました")
    else:
        logger.info("📁 すべてのディレクトリが既に存在します")


def save_quality_report(quality_checker: DataQualityChecker):
    """データ品質レポートを JSON として保存します。

    Args:
        quality_checker (DataQualityChecker): 品質レポートを保持するオブジェクト。
    """
    logger = logging.getLogger(__name__)
    report_path = Path('export/quality_reports/data_quality_report.json')
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_checker.quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 品質レポート保存: {report_path}")
        
    except Exception as e:
        logger.warning(f"⚠️ 品質レポート保存エラー: {str(e)}")


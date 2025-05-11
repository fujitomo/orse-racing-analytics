#!/usr/bin/env python
"""
競馬レース分析コマンドラインツール
レースレベルの分析を実行します。
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import sys
from horse_racing.base.analyzer import AnalysisConfig
from horse_racing.analyzers.race_level_analyzer import RaceLevelAnalyzer

# メインロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_date(date_str: str) -> datetime:
    """日付文字列のバリデーション"""
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise ValueError(f"無効な日付形式です: {date_str}。YYYYMMDD形式で指定してください。")

def validate_args(args):
    """コマンドライン引数の検証"""
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"指定されたパスが存在しません: {input_path}")
    
    if args.min_races < 1:
        raise ValueError("最小レース数は1以上を指定してください")
    
    # 日付範囲のバリデーション
    if args.start_date:
        start_date = validate_date(args.start_date)
    else:
        start_date = None
        
    if args.end_date:
        end_date = validate_date(args.end_date)
        if start_date and end_date < start_date:
            raise ValueError("終了日は開始日以降を指定してください")
    else:
        end_date = None
    
    return args

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='レースレベル分析を実行します',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_path', help='入力ファイルまたはディレクトリのパス')
    parser.add_argument('--output-dir', default='export/analysis', help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=6, help='分析対象とする最小レース数')
    parser.add_argument('--encoding', default='utf-8', help='入力ファイルのエンコーディング')
    parser.add_argument('--start-date', help='分析開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', help='分析終了日（YYYYMMDD形式）')
    
    try:
        args = parser.parse_args()
        args = validate_args(args)

        # 出力ディレクトリの作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 設定の作成
        date_str = datetime.now().strftime('%Y%m%d')
        config = AnalysisConfig(
            input_path=args.input_path,
            min_races=args.min_races,
            output_dir=str(output_dir),
            date_str=date_str,
            start_date=args.start_date,
            end_date=args.end_date
        )

        logger.info(f"分析を開始します...")
        logger.info(f"入力パス: {args.input_path}")
        logger.info(f"出力ディレクトリ: {args.output_dir}")
        logger.info(f"最小レース数: {args.min_races}")
        if args.start_date:
            logger.info(f"分析開始日: {args.start_date}")
        if args.end_date:
            logger.info(f"分析終了日: {args.end_date}")

        # 分析の実行
        analyzer = RaceLevelAnalyzer(config)
        analyzer.df = analyzer.load_data()
        analyzer.df = analyzer.preprocess_data()
        analyzer.df = analyzer.calculate_feature()
        results = analyzer.analyze()
        
        # 結果の可視化
        analyzer.stats = results
        analyzer.visualize()

        logger.info(f"分析が完了しました。結果は {output_dir} に保存されました。")
        return 0

    except FileNotFoundError as e:
        logger.error(f"ファイルエラー: {str(e)}")
        return 1
    except ValueError as e:
        logger.error(f"入力値エラー: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
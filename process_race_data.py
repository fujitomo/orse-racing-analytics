"""
競馬レースデータ処理のコマンドラインエントリーポイント
"""
from horse_racing.data.processors.bac_processor import process_all_bac_files
from horse_racing.data.processors.sed_processor import process_all_sed_files
from horse_racing.data.processors.srb_processor import process_all_srb_files, merge_srb_with_sed
from horse_racing.data.processors.data_quality_checker import DataQualityChecker
from horse_racing.data.utils import (
    setup_logging,
    ensure_export_dirs,
    save_quality_report,
    display_deletion_statistics,
    summarize_processing_log,
    SystemMonitor
)
import argparse
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# モジュール共通ロガー
logger = logging.getLogger(__name__)

# 設定・定数クラスは horse_racing.data.config からインポート
# データ処理クラスは horse_racing.data.processors からインポート
# ユーティリティ関数は horse_racing.data.utils からインポート済み

def process_race_data(exclude_turf: bool = False, turf_only: bool = False, 
                     enable_missing_value_handling: bool = True, enable_quality_check: bool = True) -> bool:
    """競馬レースデータの実務レベル処理（標準版）。

    計画書 Phase 0: データ整備の実装。
    
    この関数はRaceDataProcessorクラスのシンラッパーです。
    後方互換性のために残されています。

    Args:
        exclude_turf (bool): 芝コースを除外するかどうか。
        turf_only (bool): 芝コースのみを処理するかどうか。
        enable_missing_value_handling (bool): 戦略的欠損値処理を実行するかどうか。
        enable_quality_check (bool): データ品質チェックを実行するかどうか。

    Returns:
        bool: 成功時 ``True``、失敗時 ``False``。
    """
    logger.info("🏇 ■ 競馬レースデータの実務レベル処理を開始します ■")
    
    # システム監視開始
    monitor = SystemMonitor()
    
    # 処理オプションの確認
    if exclude_turf and turf_only:
        logger.error("❌ 芝コースを除外するオプションと芝コースのみを処理するオプションは同時に指定できません")
        return False
    
    # 通常の処理設定のログ出力
    logger.info("📋 処理設定:")
    logger.info(f"   🌱 芝コース除外: {'はい' if exclude_turf else 'いいえ'}")
    logger.info(f"   🌱 芝コースのみ: {'はい' if turf_only else 'いいえ'}")
    logger.info(f"   🔧 欠損値処理: {'有効' if enable_missing_value_handling else '無効'}")
    logger.info(f"   📈 品質チェック: {'有効' if enable_quality_check else '無効'}")
    
    # システムコンポーネントの初期化
    quality_checker = DataQualityChecker() if enable_quality_check else None
    
    # 出力用ディレクトリの確認
    ensure_export_dirs()
    monitor.log_system_status("初期化完了")
    
    try:
        # 1. BACデータの処理
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-1: BACデータ（レース基本情報）の処理")
        logger.info("="*60)
        
        process_all_bac_files(exclude_turf=exclude_turf, turf_only=turf_only)
        monitor.log_system_status("BAC処理完了")
    
        # 2. SRBデータの処理
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-2: SRBデータ（レース詳細情報）の処理")
        logger.info("="*60)
        
        process_all_srb_files(exclude_turf=exclude_turf, turf_only=turf_only)
        monitor.log_system_status("SRB処理完了")
    
        # 3. SEDデータの処理とSRB・BACデータとの紐づけ
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-3: SEDデータ（競走成績）の処理と紐づけ")
        logger.info("="*60)
        
        process_all_sed_files(exclude_turf=exclude_turf, turf_only=turf_only)
    
        # 4. SEDデータとSRBデータの紐づけ
        logger.info("\n" + "="*60)
        logger.info("📂 Phase 0-4: SEDデータとSRBデータの統合")
        logger.info("="*60)
        logger.info("📋 バイアス情報完備データのみを保持します")
        
        merge_result = merge_srb_with_sed(
            separate_output=True, 
            exclude_turf=exclude_turf, 
            turf_only=turf_only
        )
        
        if not merge_result:
            logger.error("❌ SEDデータとSRBデータの紐づけに失敗しました")
            return False
        
        logger.info("✅ データ統合完了:")
        logger.info("   📁 SEDデータ: export/SED/")
        logger.info("   📁 SRBデータ: export/SRB/")
        logger.info("   📁 統合データ: export/dataset/")
        
        monitor.log_system_status("データ統合完了")
        
        # 5. データ品質チェック（統合後）
        if enable_quality_check:
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 0-5: データ品質チェック")
            logger.info("="*60)
            
            # サンプルファイルで品質チェック実行
            sample_files = list(Path('export/dataset').glob('*.csv'))
            if sample_files:
                sample_file = sample_files[0]
                logger.info(f"📄 サンプルファイルで品質チェック: {sample_file.name}")
                
                try:
                    sample_df = pd.read_csv(sample_file, encoding='utf-8')
                    quality_checker.check_data_quality(sample_df, "統合後データ")
                except Exception as e:
                    logger.warning(f"⚠️ 品質チェックエラー: {str(e)}")
        
        # 7. 品質レポートの保存
        if enable_quality_check and quality_checker:
            save_quality_report(quality_checker)
        
        # 8. 欠損値処理ログのサマリー生成（実務レベル）
        if enable_missing_value_handling:
            logger.info("\n" + "="*60)
            logger.info("📝 Phase 0-7: 欠損値処理ログの自動整理")
            logger.info("="*60)
            summarize_processing_log()
        
        # 9. グレード欠損削除統計の表示
        if enable_missing_value_handling:
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 0-8: グレード欠損削除統計")
            logger.info("="*60)
            display_deletion_statistics()
        
        # 10. 処理完了サマリー
        logger.info("\n" + "="*60)
        logger.info("🎉 Phase 0: データ整備 完了")
        logger.info("="*60)
        
        total_time = time.time() - monitor.start_time
        logger.info(f"⏱️ 総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        monitor.log_system_status("全処理完了")
        
        logger.info("\n📁 生成されたデータ:")
        if Path('export/dataset').exists():
            bias_files = list(Path('export/dataset').glob('*.csv'))
            logger.info(f"   🔗 統合データ: {len(bias_files)}ファイル")
        
        if enable_quality_check and Path('export/quality_reports').exists():
            logger.info("   📈 品質レポート: export/quality_reports/")
        
        logger.info("\n🎓 実務レベルのデータ整備が完了しました！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ データ処理中に予期せぬエラーが発生しました: {str(e)}")
        logger.error("🔧 スタックトレース:", exc_info=True)
        return False

if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='競馬レースデータの実務レベル処理（計画書Phase 0：データ整備対応版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 使用例:
  python process_race_data.py                                    # 基本処理
  python process_race_data.py --turf-only                      # 芝コースのみで処理
  python process_race_data.py --no-missing-handling              # 欠損値処理を無効化
  python process_race_data.py --no-quality-check                 # 品質チェックを無効化

🔧 このスクリプトの役割:
  このスクリプトは、複数の形式の生レースデータ（BAC, SRB, SED）を読み込み、
  それらを一つの整形されたデータセットに統合します。
  最終的な成果物は `export/dataset/` ディレクトリに出力され、
  これが後続の分析スクリプト（例: analyze_horse_REQI.py）の入力となります。

🔧 実務レベルの品質管理:
  ✅ 戦略的欠損値処理
  ✅ データ品質チェックとレポート
  ✅ 欠損値処理ログの自動サマリー生成
  ✅ システム監視
  ✅ 段階的処理とログ出力
  ✅ エラーハンドリングと復旧機能
        """
    )
    
    # トラック条件オプション
    track_group = parser.add_mutually_exclusive_group()
    track_group.add_argument('--exclude-turf', '--芝コース除外', action='store_true', 
                           help='芝コースのデータを除外する')
    track_group.add_argument('--turf-only', '--芝コースのみ', action='store_true', 
                           help='芝コースのデータのみを処理する')
    
    # 機能オプション
    parser.add_argument('--no-missing-handling', '--欠損値処理無効', action='store_true',
                       help='戦略的欠損値処理を無効化する')
    
    parser.add_argument('--no-quality-check', '--品質チェック無効', action='store_true',
                       help='データ品質チェックを無効化する')
    
    # ログレベルオプション
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='ログレベルの設定')
    
    parser.add_argument('--log-file', help='ログファイルのパス（指定しない場合はコンソールのみ）')
    
    args = parser.parse_args()
    
    # ログ設定の初期化
    log_file = args.log_file
    
    if log_file is None:
        # 自動ログファイル設定（ディレクトリ作成も含む）
        log_dir = Path('export/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = f'export/logs/process_race_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    setup_logging(log_level=args.log_level, log_file=log_file)
    
    # メインロガーでの開始メッセージ
    main_logger = logging.getLogger(__name__)
    main_logger.info("🚀 競馬レースデータ実務レベル処理を開始します")
    main_logger.info(f"📅 実行日時: {datetime.now()}")
    main_logger.info(f"🖥️ ログレベル: {args.log_level}")
    if log_file:
        main_logger.info(f"📝 ログファイル: {log_file}")

    try:
        success = process_race_data(
            exclude_turf=args.exclude_turf,
            turf_only=args.turf_only,
            enable_missing_value_handling=not args.no_missing_handling,
            enable_quality_check=not args.no_quality_check,
        )
    except Exception as e:
        main_logger.error(f"❌ 予期せぬエラー: {str(e)}")
        main_logger.error("🔧 スタックトレース:", exc_info=True)
        success = False

    if success:
        main_logger.info("🎉 実務レベルデータ処理が正常に完了しました")
        exit_code = 0
    else:
        main_logger.error("❌ データ処理が失敗しました")
        exit_code = 1

    main_logger.info(f"🏁 プロセス終了 (終了コード: {exit_code})")
    exit(exit_code)

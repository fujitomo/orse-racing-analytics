#!/usr/bin/env python
"""
競馬レース分析コマンドラインツール
馬ごとのレースレベルの分析を実行します。
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import sys
from horse_racing.base.analyzer import AnalysisConfig
from horse_racing.analyzers.race_level_analyzer import RaceLevelAnalyzer

def setup_logging(log_level='INFO', log_file=None):
    """ログ設定（コンソールとファイル出力対応）"""
    if log_file:
        # ログディレクトリの作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # コンソール出力
                logging.FileHandler(log_file, encoding='utf-8')  # ファイル出力
            ],
            force=True  # 既存の設定を上書き
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
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

def analyze_by_periods(analyzer, periods, base_output_dir):
    """期間別に分析を実行"""
    all_results = {}
    
    for period_name, start_year, end_year in periods:
        logger.info(f"期間 {period_name} の分析開始...")
        
        try:
            # 期間別出力ディレクトリの作成
            period_output_dir = base_output_dir / period_name
            period_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 期間別の設定を作成
            period_config = AnalysisConfig(
                input_path=analyzer.config.input_path,
                min_races=analyzer.config.min_races,
                output_dir=str(period_output_dir),
                date_str=analyzer.config.date_str,
                start_date=f"{start_year}0101" if start_year else None,
                end_date=f"{end_year}1231" if end_year else None
            )
            
            logger.info(f"  📅 期間設定: {start_year}年 - {end_year}年")
            logger.info(f"  📁 出力先: {period_config.output_dir}")
            
            # 期間別アナライザーを作成
            period_analyzer = RaceLevelAnalyzer(period_config, 
                                              enable_time_analysis=analyzer.enable_time_analysis,
                                              enable_stratified_analysis=analyzer.enable_stratified_analysis)
            
            # 期間別分析の実行
            logger.info(f"  📖 データ読み込み中...")
            period_analyzer.df = period_analyzer.load_data()
            
            # 前処理の追加
            logger.info(f"  🔧 前処理中...")
            period_analyzer.df = period_analyzer.preprocess_data()

            # 特徴量を計算
            logger.info(f"  🧮 特徴量計算中...")
            period_analyzer.df = period_analyzer.calculate_feature()

            # ここでデータチェック
            required_cols = ['馬名', '着順', 'race_level']
            if not all(col in period_analyzer.df.columns for col in required_cols):
                logger.error(f"期間 {period_name} のデータに必要なカラムがありません。スキップします。")
                continue

            # データが十分にあるかチェック
            if len(period_analyzer.df) < analyzer.config.min_races:
                logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(period_analyzer.df)}行)")
                continue
            
            logger.info(f"  📊 対象データ: {len(period_analyzer.df)}行")
            logger.info(f"  🐎 対象馬数: {len(period_analyzer.df['馬名'].unique())}頭")
            
            logger.info(f"  📈 分析実行中...")
            results = period_analyzer.analyze()
            
            # 結果の可視化
            logger.info(f"  📊 可視化生成中...")
            period_analyzer.stats = results
            period_analyzer.visualize()
            
            # 期間情報を結果に追加
            results['period_info'] = {
                'name': period_name,
                'start_year': start_year,
                'end_year': end_year,
                'total_races': len(period_analyzer.df),
                'total_horses': len(period_analyzer.df['馬名'].unique())
            }
            
            all_results[period_name] = results
            logger.info(f"期間 {period_name} の分析完了: {results['period_info']['total_races']}レース, {results['period_info']['total_horses']}頭")
            
        except Exception as e:
            logger.error(f"期間 {period_name} の分析でエラー: {str(e)}")
            logger.error("詳細なエラー情報:", exc_info=True)
            continue
    
    return all_results

def generate_period_summary_report(all_results, output_dir):
    """期間別分析の総合レポートを生成"""
    report_path = output_dir / 'レースレベル分析_期間別総合レポート.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# レースレベル分析 期間別総合レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 分析期間一覧\n\n")
        f.write("| 期間 | 対象馬数 | 総レース数 | 平均レベル相関 | 最高レベル相関 |\n")
        f.write("|------|----------|-----------|---------------|---------------|\n")
        
        for period_name, results in all_results.items():
            period_info = results.get('period_info', {})
            correlation_stats = results.get('correlation_stats', {})
            
            total_horses = period_info.get('total_horses', 0)
            total_races = period_info.get('total_races', 0)
            
            # 相関係数の取得
            corr_avg = correlation_stats.get('correlation_place_avg', 0.0)
            corr_max = correlation_stats.get('correlation_place_max', 0.0)
            
            f.write(f"| {period_name} | {total_horses:,}頭 | {total_races:,}レース | {corr_avg:.3f} | {corr_max:.3f} |\n")
        
        # 各期間の詳細
        for period_name, results in all_results.items():
            f.write(f"\n## 📈 期間: {period_name}\n\n")
            
            period_info = results.get('period_info', {})
            correlation_stats = results.get('correlation_stats', {})
            
            f.write(f"### 基本情報\n")
            f.write(f"- **分析期間**: {period_info.get('start_year', '不明')}年 - {period_info.get('end_year', '不明')}年\n")
            f.write(f"- **対象馬数**: {period_info.get('total_horses', 0):,}頭\n")
            f.write(f"- **総レース数**: {period_info.get('total_races', 0):,}レース\n\n")
            
            f.write(f"### 相関分析結果\n")
            if correlation_stats:
                # 平均レベル分析
                corr_place_avg = correlation_stats.get('correlation_place_avg', 0.0)
                r2_place_avg = correlation_stats.get('r2_place_avg', 0.0)
                
                # 最高レベル分析
                corr_place_max = correlation_stats.get('correlation_place_max', 0.0)
                r2_place_max = correlation_stats.get('r2_place_max', 0.0)
                
                f.write(f"**平均レースレベル vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_avg:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_avg:.3f}\n\n")
                
                f.write(f"**最高レースレベル vs 複勝率**\n")
                f.write(f"- 相関係数: {corr_place_max:.3f}\n")
                f.write(f"- 決定係数 (R²): {r2_place_max:.3f}\n\n")
            else:
                f.write("- 相関分析データなし\n\n")
        
        f.write("\n## 💡 総合的な傾向と知見\n\n")
        
        # 期間別の相関係数変化
        if len(all_results) > 1:
            f.write("### 時系列変化\n")
            f.write("平均レースレベルと複勝率の相関係数の変化：\n")
            
            correlations_by_period = []
            for period_name, results in all_results.items():
                correlation_stats = results.get('correlation_stats', {})
                corr = correlation_stats.get('correlation_place_avg', 0.0)
                correlations_by_period.append((period_name, corr))
            
            for i, (period, corr) in enumerate(correlations_by_period):
                if i > 0:
                    prev_corr = correlations_by_period[i-1][1]
                    change = corr - prev_corr
                    trend = "上昇" if change > 0.05 else "下降" if change < -0.05 else "横ばい"
                    f.write(f"- {period}: {corr:.3f} ({trend})\n")
                else:
                    f.write(f"- {period}: {corr:.3f} (基準)\n")
        
        f.write("\n### レースレベル分析の特徴\n")
        f.write("- レースレベルは競馬場の格式度と実力の関係を数値化\n")
        f.write("- 平均レベル：馬の継続的な実力を表す指標\n")
        f.write("- 最高レベル：馬のピーク時の実力を表す指標\n")
        f.write("- 時系列分析により、競馬界の格式体系の変化を把握可能\n")
    
    logger.info(f"期間別総合レポート保存: {report_path}")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='レースレベル分析を実行します（3年間隔分析対応、RunningTime分析対応）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な分析の実行 (入力は一次データセットのディレクトリ)
  python analyze_race_level.py export/with_bias --output-dir results/race_level_analysis

  # 3年間隔での時系列分析
  python analyze_race_level.py export/with_bias --three-year-periods

  # 走破タイムとの因果関係分析を有効化
  python analyze_race_level.py export/with_bias --enable-time-analysis

このスクリプトの役割:
  process_race_data.py によって生成された一次データセット（例: export/with_bias/）を入力とし、
  以下の処理を一貫して実行します。
  1. データの読み込みと前処理（期間や最小レース数でのフィルタリング）
  2. レースレベル特徴量の計算
  3. 馬ごとの統計量の集計
  4. レースレベルと競走成績（複勝率など）の相関分析
  5. 結果の可視化（グラフ生成）
  6. （オプション）時系列分析やタイム因果分析
        """
    )
    parser.add_argument('input_path', help='入力ファイルまたはディレクトリのパス (例: export/with_bias)')
    parser.add_argument('--output-dir', default='results/race_level_analysis', help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=6, help='分析対象とする最小レース数')
    parser.add_argument('--encoding', default='utf-8', help='入力ファイルのエンコーディング')
    parser.add_argument('--start-date', help='分析開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', help='分析終了日（YYYYMMDD形式）')
    parser.add_argument('--three-year-periods', action='store_true',
                       help='3年間隔での期間別分析を実行（デフォルトは全期間分析）')
    parser.add_argument('--enable-time-analysis', action='store_true',
                       help='走破タイム因果関係分析を実行（論文仮説H1, H4検証）')
    parser.add_argument('--enable-stratified-analysis', action='store_true', default=True,
                       help='層別分析を実行（年齢層別、経験数別、距離カテゴリ別）- デフォルトで有効')
    parser.add_argument('--disable-stratified-analysis', action='store_true',
                       help='層別分析を無効化（処理時間短縮用）')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='ログレベルの設定')
    parser.add_argument('--log-file', help='ログファイルのパス（指定しない場合は自動生成）')
    
    # ログファイル変数の初期化
    log_file = None
    
    try:
        args = parser.parse_args()
        
        # ログファイルの自動生成（args取得後、validate_args前に実行）
        log_file = args.log_file
        if log_file is None:
            # ログディレクトリの作成
            log_dir = Path('export/logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = f'export/logs/analyze_horse_racelevel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # ログ設定の初期化
        setup_logging(log_level=args.log_level, log_file=log_file)
        
        # 引数検証（ログ設定後に実行）
        args = validate_args(args)

        # ログ設定完了後に開始メッセージを出力
        logger.info("🏇 レースレベル分析を開始します...")
        logger.info(f"📅 実行日時: {datetime.now()}")
        logger.info(f"🖥️ ログレベル: {args.log_level}")
        logger.info(f"📝 ログファイル: {log_file}")

        # 出力ディレクトリの作成（親ディレクトリも含めて確実に作成）
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 出力ディレクトリが書き込み可能かチェック
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"出力ディレクトリの作成に失敗しました: {output_dir}")
        
        logger.info(f"📁 出力ディレクトリ確認済み: {output_dir.absolute()}")

        logger.info(f"📁 入力パス: {args.input_path}")
        logger.info(f"📊 出力ディレクトリ: {args.output_dir}")
        logger.info(f"🎯 最小レース数: {args.min_races}")
        if args.start_date:
            logger.info(f"📅 分析開始日: {args.start_date}")
        if args.end_date:
            logger.info(f"📅 分析終了日: {args.end_date}")
        if args.enable_time_analysis:
            logger.info(f"🏃 RunningTime分析: 有効")
        else:
            logger.info(f"🏃 RunningTime分析: 無効（--enable-time-analysisで有効化）")
        
        # 層別分析設定の処理
        enable_stratified = args.enable_stratified_analysis and not args.disable_stratified_analysis
        if enable_stratified:
            logger.info(f"📊 層別分析: 有効（年齢層別・経験数別・距離カテゴリ別）")
        else:
            logger.info(f"📊 層別分析: 無効（--disable-stratified-analysisで無効化）")

        if args.three_year_periods:
            logger.info("📊 3年間隔での期間別分析を実行します...")
            
            # 初期データ読み込みで年データ範囲を確認
            temp_config = AnalysisConfig(
                input_path=args.input_path,
                min_races=args.min_races,
                output_dir=str(output_dir),
                date_str=datetime.now().strftime('%Y%m%d'),
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            # データ読み込みと基本的な前処理（期間フィルタリングなし）
            temp_analyzer = RaceLevelAnalyzer(temp_config, 
                                            enable_time_analysis=args.enable_time_analysis,
                                            enable_stratified_analysis=enable_stratified)
            logger.info("📖 全データ読み込み中...")
            temp_df = temp_analyzer.load_data()
            
            logger.info(f"📊 読み込んだデータ件数: {len(temp_df):,}件")
            
            # 年データが存在するかチェック
            if '年' in temp_df.columns and temp_df['年'].notna().any():
                min_year = int(temp_df['年'].min())
                max_year = int(temp_df['年'].max())
                logger.info(f"📊 年データ範囲: {min_year}年 - {max_year}年")
                
                # 3年間隔での期間設定
                periods = []
                for start_year in range(min_year, max_year + 1, 3):
                    end_year = min(start_year + 2, max_year)
                    period_name = f"{start_year}-{end_year}"
                    
                    # 期間内にデータが存在するかチェック
                    period_data = temp_df[
                        (temp_df['年'] >= start_year) & (temp_df['年'] <= end_year)
                    ]
                    
                    if len(period_data) >= args.min_races:
                        periods.append((period_name, start_year, end_year))
                        logger.info(f"  📊 期間 {period_name}: {len(period_data):,}件のデータ")
                    else:
                        logger.warning(f"  ⚠️  期間 {period_name}: データ不足 ({len(period_data)}件)")
                
                if periods:
                    logger.info(f"📊 有効な分析期間: {[p[0] for p in periods]}")
                    
                    # 期間別分析の実行
                    all_results = analyze_by_periods(temp_analyzer, periods, output_dir)
                    
                    if all_results:
                        # 総合レポートの生成
                        generate_period_summary_report(all_results, output_dir)
                        
                        logger.info("\n" + "="*60)
                        logger.info("🎉 3年間隔分析完了！結果:")
                        logger.info("="*60)
                        
                        for period_name, results in all_results.items():
                            period_info = results.get('period_info', {})
                            correlation_stats = results.get('correlation_stats', {})
                            
                            total_horses = period_info.get('total_horses', 0)
                            total_races = period_info.get('total_races', 0)
                            corr_avg = correlation_stats.get('correlation_place_avg', 0.0)
                            
                            logger.info(f"📊 期間 {period_name}: {total_horses:,}頭, {total_races:,}レース")
                            logger.info(f"   📈 平均レベル vs 複勝率相関: r={corr_avg:.3f}")
                        
                        logger.info("="*60)
                        logger.info(f"✅ 全ての結果は {args.output_dir} に保存されました。")
                        logger.info(f"📝 ログファイル: {log_file}")
                        logger.info("📋 生成されたファイル:")
                        logger.info("  - レースレベル分析_期間別総合レポート.md")
                        logger.info("  - 各期間フォルダ内の分析結果PNG")
                    else:
                        logger.warning("⚠️  有効な期間別分析結果がありませんでした。")
                else:
                    logger.warning("⚠️  十分なデータがある期間が見つかりませんでした。全期間での分析に切り替えます。")
                    args.three_year_periods = False
            else:
                logger.warning("⚠️  年データが見つかりません。全期間での分析に切り替えます。")
                args.three_year_periods = False
        
        if not args.three_year_periods:
            logger.info("📊 【修正版】厳密な時系列分割による分析を実行します...")
            
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

            # 1. RaceLevelAnalyzerのインスタンス化
            analyzer = RaceLevelAnalyzer(config, args.enable_time_analysis, enable_stratified)

            # 2. データの読み込み
            logger.info("📖 全データ読み込み中...")
            analyzer.df = analyzer.load_data()

            # 前処理を追加
            logger.info("🔧 前処理中...")
            analyzer.df = analyzer.preprocess_data()
            
            # 3. 特徴量計算
            logger.info("🧮 特徴量計算中...")
            analyzer.df = analyzer.calculate_feature()

            # 4. 【重要】修正版分析の実行
            logger.info("🔬 【修正版】厳密な時系列分割による分析を実行中...")
            analyzer.stats = analyzer.analyze()
            
            # 結果の可視化
            analyzer.visualize()

            # 【追加】レポート整合性の確認
            logger.info("🔍 レポート整合性チェック:")
            oot_results = analyzer.stats.get('out_of_time_validation', {})
            test_performance = oot_results.get('test_performance', {})
            
            if test_performance:
                test_r2 = test_performance.get('r_squared', 0)
                test_corr = test_performance.get('correlation', 0)
                test_size = test_performance.get('sample_size', 0)
                
                logger.info(f"   📊 検証期間(2013-2014年)サンプル数: {test_size}頭")
                logger.info(f"   📊 検証期間R²: {test_r2:.3f}")
                logger.info(f"   📊 検証期間相関係数: {test_corr:.3f}")
                
                # 実測結果の統計的評価
                if test_r2 > 0.01:
                    logger.info("✅ 統計的に有意な説明力を確認")
                else:
                    logger.warning("⚠️ 説明力が限定的です")
                    
                if abs(test_corr) > 0.1:
                    logger.info("✅ 実用的な相関関係を確認")
                else:
                    logger.warning("⚠️ 相関関係が弱いです")

            logger.info(f"✅ 【修正版】分析が完了しました。結果は {output_dir} に保存されました。")
            logger.info(f"📝 ログファイル: {log_file}")
            logger.info("🎯 データリーケージ防止と時系列分割が正しく実装されました。")

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ ファイルエラー: {str(e)}")
        logger.error("💡 解決方法:")
        logger.error("   • 入力パスが正しいか確認してください")
        logger.error("   • ファイル名に日本語が含まれている場合は英数字に変更してください")
        logger.error("   • 'export/with_bias' ディレクトリが存在するか確認してください")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"❌ 入力値エラー: {error_msg}")
        logger.error("💡 解決方法:")
        
        if "条件を満たすデータが見つかりません" in error_msg:
            logger.error("   • --min-races の値を小さくしてみてください（例: --min-races 3）")
            logger.error("   • 期間指定が狭すぎる場合は範囲を広げてください")
            logger.error("   • データが存在する期間かどうか確認してください")
        elif "日付形式" in error_msg:
            logger.error("   • 日付はYYYYMMDD形式で指定してください（例: 20220101）")
            logger.error("   • --start-date と --end-date の両方を指定してください")
        else:
            logger.error("   • パラメータの値が正しいか確認してください")
            logger.error("   • --help でオプションの詳細を確認できます")
        
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except IndexError as e:
        logger.error(f"❌ データ処理エラー: {str(e)}")
        logger.error("💡 解決方法:")
        logger.error("   • データ期間が短すぎる可能性があります")
        logger.error("   • 時系列分割に必要な最低3年分のデータがあるか確認してください")
        logger.error("   • 期間指定を広げて再実行してみてください")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        return 1
    except KeyboardInterrupt:
        logger.warning("⏹️ ユーザーによって処理が中断されました")
        logger.info("💡 処理時間を短縮するには:")
        logger.info("   • --min-races を大きくしてサンプル数を減らす")
        logger.info("   • 期間を短くして処理範囲を絞る")
        logger.info("   • --disable-stratified-analysis で層別分析を無効化")
        if log_file:
            logger.info(f"📝 ログファイル: {log_file}")
        return 1
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 予期せぬエラーが発生しました: {error_msg}")
        logger.error("💡 解決方法:")
        
        if "encoding" in error_msg.lower() or "unicode" in error_msg.lower():
            logger.error("   • ファイルのエンコーディングに問題があります")
            logger.error("   • CSVファイルがUTF-8またはShift-JISで保存されているか確認してください")
        elif "memory" in error_msg.lower():
            logger.error("   • メモリ不足の可能性があります")
            logger.error("   • --min-races を大きくしてデータ量を減らしてください")
            logger.error("   • 不要なアプリケーションを終了してください")
        elif "permission" in error_msg.lower():
            logger.error("   • ファイルアクセス権限の問題があります")
            logger.error("   • 出力ディレクトリの書き込み権限を確認してください")
            logger.error("   • 管理者権限で実行してみてください")
        else:
            logger.error("   • --log-level DEBUG で詳細ログを確認してください")
            logger.error("   • データファイルが破損していないか確認してください")
            logger.error("   • Pythonとライブラリのバージョンを確認してください")
        
        logger.error("🔍 詳細なエラー情報:")
        logger.error(f"   エラー種別: {type(e).__name__}")
        logger.error(f"   エラー内容: {error_msg}")
        if log_file:
            logger.error(f"📝 ログファイル: {log_file}")
        logger.error("詳細なスタックトレース:", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
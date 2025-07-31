#!/usr/bin/env python
"""
競馬レース分析コマンドラインツール
レースレベルの分析を実行します。
3年間隔での時系列分析にも対応。
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

def analyze_by_periods(analyzer, periods, base_output_dir):
    """期間別に分析を実行"""
    all_results = {}
    
    for period_name, start_year, end_year in periods:
        logger.info(f"期間 {period_name} の分析開始...")
        
        try:
            # 期間別の設定を作成
            period_config = AnalysisConfig(
                input_path=analyzer.config.input_path,
                min_races=analyzer.config.min_races,
                output_dir=str(base_output_dir / period_name),
                date_str=analyzer.config.date_str,
                start_date=f"{start_year}0101" if start_year else None,
                end_date=f"{end_year}1231" if end_year else None
            )
            
            logger.info(f"  📅 期間設定: {start_year}年 - {end_year}年")
            logger.info(f"  📁 出力先: {period_config.output_dir}")
            
            # 期間別アナライザーを作成
            period_analyzer = RaceLevelAnalyzer(period_config, enable_time_analysis=analyzer.enable_time_analysis)
            
            # 期間別分析の実行
            logger.info(f"  📖 データ読み込み中...")
            period_analyzer.df = period_analyzer.load_data()
            
            logger.info(f"  🔧 前処理中...")
            period_analyzer.df = period_analyzer.preprocess_data()
            
            # データが十分にあるかチェック
            if len(period_analyzer.df) < analyzer.config.min_races:
                logger.warning(f"期間 {period_name}: データ不足のためスキップ ({len(period_analyzer.df)}行)")
                continue
            
            logger.info(f"  📊 対象データ: {len(period_analyzer.df)}行")
            logger.info(f"  🐎 対象馬数: {len(period_analyzer.df['馬名'].unique())}頭")
            
            logger.info(f"  🧮 特徴量計算中...")
            period_analyzer.df = period_analyzer.calculate_feature()
            
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
  python analyze_race_level.py export/with_bias
  python analyze_race_level.py export/with_bias --output-dir results/race_level_analysis
  python analyze_race_level.py export/with_bias --three-year-periods  # 3年間隔分析
  python analyze_race_level.py export/with_bias --enable-time-analysis  # RunningTime分析
  
分析内容:
  - レースレベル（グレード・賞金・距離による格付け）の計算
  - レースレベルと勝率・複勝率の相関分析
  - オプション: 3年間隔での時系列分析
  - オプション: 走破タイム因果関係分析（論文仮説H1, H4検証）
        """
    )
    parser.add_argument('input_path', help='入力ファイルまたはディレクトリのパス')
    parser.add_argument('--output-dir', default='export/race_level_analysis', help='出力ディレクトリのパス')
    parser.add_argument('--min-races', type=int, default=6, help='分析対象とする最小レース数')
    parser.add_argument('--encoding', default='utf-8', help='入力ファイルのエンコーディング')
    parser.add_argument('--start-date', help='分析開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', help='分析終了日（YYYYMMDD形式）')
    parser.add_argument('--three-year-periods', action='store_true',
                       help='3年間隔での期間別分析を実行（デフォルトは全期間分析）')
    parser.add_argument('--enable-time-analysis', action='store_true',
                       help='走破タイム因果関係分析を実行（論文仮説H1, H4検証）')
    
    try:
        args = parser.parse_args()
        args = validate_args(args)

        # 出力ディレクトリの作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🏇 レースレベル分析を開始します...")
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
            temp_analyzer = RaceLevelAnalyzer(temp_config, enable_time_analysis=args.enable_time_analysis)
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
            logger.info("📊 全期間での分析を実行します...")
            
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

            # 分析の実行
            analyzer = RaceLevelAnalyzer(config, enable_time_analysis=args.enable_time_analysis)
            analyzer.df = analyzer.load_data()
            analyzer.df = analyzer.preprocess_data()
            analyzer.df = analyzer.calculate_feature()
            results = analyzer.analyze()
            
            # 結果の可視化
            analyzer.stats = results
            analyzer.visualize()

            logger.info(f"✅ 分析が完了しました。結果は {output_dir} に保存されました。")

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ ファイルエラー: {str(e)}")
        return 1
    except ValueError as e:
        logger.error(f"❌ 入力値エラー: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"❌ 予期せぬエラーが発生しました: {str(e)}")
        logger.error("詳細なエラー情報:", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
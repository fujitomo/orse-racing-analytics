"""
レポート生成ユーティリティ
"""
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def display_deletion_statistics():
    """グレード欠損による削除統計を表示する。"""
    logger = logging.getLogger(__name__)
    
    try:
        def _count_csv_rows(file_path: Path) -> int:
            buffer_size = 1024 * 1024
            newline_count = 0
            last_char = b'\n'

            with file_path.open('rb') as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    newline_count += chunk.count(b'\n')
                    last_char = chunk[-1:]

            line_count = newline_count
            if last_char not in (b'\n', b''):
                line_count += 1

            return max(line_count - 1, 0)

        # ディレクトリパス
        sed_dir = Path('export/SED/formatted')
        bias_dir = Path('export/dataset')
        
        if not sed_dir.exists() or not bias_dir.exists():
            logger.warning("⚠️ 比較用ディレクトリが見つかりません")
            return
        
        # ファイル一覧取得
        sed_files = list(sed_dir.glob('*.csv'))
        bias_files = list(bias_dir.glob('*.csv'))
        
        if not sed_files or not bias_files:
            logger.warning("⚠️ 比較用ファイルが見つかりません")
            return
        
        # 統計を収集
        total_sed = 0
        total_bias = 0
        total_deleted = 0
        deletion_files = []
        
        # ファイル名でマッピング
        sed_files_dict = {f.stem.replace('_formatted', ''): f for f in sed_files}
        
        for bias_file in bias_files:
            base_name = bias_file.stem.replace('_formatted_dataset', '')
            
            if base_name in sed_files_dict:
                sed_file = sed_files_dict[base_name]
                
                try:
                    # レコード数を数える（ヘッダー除く）
                    sed_count = _count_csv_rows(sed_file)
                    bias_count = _count_csv_rows(bias_file)
                    
                    deleted = sed_count - bias_count
                    total_sed += sed_count
                    total_bias += bias_count
                    total_deleted += deleted
                    
                    if deleted > 0:
                        deletion_rate = (deleted / sed_count * 100) if sed_count > 0 else 0
                        deletion_files.append({
                            'file': base_name,
                            'deleted': deleted,
                            'deletion_rate': deletion_rate
                        })
                
                except Exception:
                    continue
        
        # 統計表示
        logger.info("📈 全体削除統計:")
        logger.info(f"   📥 処理前総レコード: {total_sed:,}件")
        logger.info(f"   📤 処理後総レコード: {total_bias:,}件")
        logger.info(f"   ❌ 削除レコード数: {total_deleted:,}件")
        logger.info(f"   📉 全体削除率: {(total_deleted/total_sed*100 if total_sed > 0 else 0):.2f}%")
        logger.info(f"   🗂️ 削除発生ファイル数: {len(deletion_files)}")
        logger.info(f"   📊 削除発生率: {(len(deletion_files)/len(sed_files_dict)*100 if sed_files_dict else 0):.1f}%")
        
        if deletion_files:
            logger.info("\n📋 削除の多いファイル（上位10件）:")
            deletion_files.sort(key=lambda x: x['deleted'], reverse=True)
            for i, item in enumerate(deletion_files[:10], 1):
                logger.info(f"   {i:2d}. {item['file']}: -{item['deleted']:,}件 (-{item['deletion_rate']:.1f}%)")
        else:
            logger.info("✅ グレード欠損による削除は発生していません")
    
    except Exception as e:
        logger.warning(f"⚠️ 削除統計表示エラー: {str(e)}")


def summarize_processing_log():
    """欠損値処理ログのサマリーを生成する。"""
    logger = logging.getLogger(__name__)
    
    log_file = Path('export/missing_value_processing_log.txt')
    backup_file = Path('export/missing_value_processing_log_original.txt')
    summary_file = Path('export/missing_value_processing_summary.txt')
    
    # ログファイルが存在しない場合はスキップ
    if not log_file.exists():
        logger.info("📝 欠損値処理ログが見つからないため、サマリー生成をスキップします")
        return
    
    logger.info("📊 欠損値処理ログをサマリー形式に整理中...")
    
    try:
        # ログ解析
        stats = _parse_processing_log(log_file)
        
        if not stats:
            logger.warning("⚠️ ログ解析に失敗しました")
            return
        
        # サマリーレポート生成
        _generate_summary_report(stats, summary_file)
        
        # 元ログをバックアップ
        if backup_file.exists():
            backup_file.unlink()  # 既存バックアップを削除
        log_file.rename(backup_file)
        
        # サマリーを新しいログファイルに
        summary_file.rename(log_file)
        
        logger.info("✅ 欠損値処理ログの整理完了")
        logger.info(f"   📋 サマリー: {log_file}")
        logger.info(f"   💾 バックアップ: {backup_file}")
        logger.info(f"   📊 処理ファイル数: {stats['total_files']}ファイル")
        
        # 統計サマリーをログ出力
        if stats['idm_deletions']:
            total_idm = sum(stats['idm_deletions'])
            logger.info(f"   🎯 IDM削除: {total_idm:,}行 ({len(stats['idm_deletions'])}ファイル)")
        
        if stats['grade_estimations']:
            total_grade = sum(stats['grade_estimations'])
            logger.info(f"   🏆 グレード推定: {total_grade:,}件 ({len(stats['grade_estimations'])}ファイル)")
        
    except Exception as e:
        logger.warning(f"⚠️ ログサマリー生成エラー: {str(e)}")


def _parse_processing_log(log_file: Path) -> Optional[Dict[str, Any]]:
    """欠損値処理ログを解析して統計を生成します。

    Args:
        log_file (Path): 解析対象のログファイルパス。

    Returns:
        Optional[Dict[str, Any]]: ログ解析結果の統計情報。
    """
    logger = logging.getLogger(__name__)
    
    # 統計情報格納用
    stats = {
        'idm_deletions': [],
        'grade_estimations': [],
        'median_imputations': defaultdict(list),
        'dropped_columns': set(),
        'categorical_imputations': defaultdict(list),
        'other_imputations': defaultdict(list),
        'total_files': 0,
        'final_shapes': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"ログファイル読み込みエラー: {e}")
        return {}
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('==') or line.startswith('欠損値処理ログ'):
            continue
            
        # IDM削除
        if 'IDM:' in line and '行を削除（重要列）' in line:
            match = re.search(r'IDM: (\d+)行を削除', line)
            if match:
                stats['idm_deletions'].append(int(match.group(1)))
        
        # グレード推定
        elif 'グレード:' in line and '推定→グレード名列追加' in line:
            match = re.search(r'グレード: 賞金・レース名から(\d+)件推定', line)
            if match:
                stats['grade_estimations'].append(int(match.group(1)))
        
        # 中央値補完
        elif 'medianで' in line and '件補完' in line:
            match = re.search(r'• ([^:]+): medianで(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                count = int(match.group(2))
                stats['median_imputations'][column_name].append(count)
        
        # 高欠損率による列削除
        elif '高欠損率により列削除' in line:
            match = re.search(r'• ([^:]+): 高欠損率により列削除', line)
            if match:
                stats['dropped_columns'].add(match.group(1))
        
        # カテゴリ補完（レース名、馬体重増減）
        elif line.startswith('• レース名:') or line.startswith('• レース名略称:') or line.startswith('• 馬体重増減:'):
            match = re.search(r'• ([^:]+): (.+)で(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                value = match.group(2)
                count = int(match.group(3))
                stats['categorical_imputations'][column_name].append((value, count))
        
        # その他の補完処理
        elif '件補完' in line and 'median' not in line:
            match = re.search(r'• ([^:]+): (.+)で(\d+)件補完', line)
            if match:
                column_name = match.group(1)
                value = match.group(2)
                count = int(match.group(3))
                stats['other_imputations'][column_name].append((value, count))
        
        # 最終データ形状
        elif '最終データ形状:' in line:
            match = re.search(r'最終データ形状: \((\d+), (\d+)\)', line)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                stats['final_shapes'].append((rows, cols))
    
    # ファイル数を推定（IDM削除の回数とグレード推定の回数の合計）
    stats['total_files'] = len(stats['idm_deletions']) + len(stats['grade_estimations'])
    
    return stats


def _generate_summary_report(stats: Dict[str, Any], output_file: Path):
    """統計情報からサマリーレポートを生成します。

    Args:
        stats (Dict[str, Any]): ログ解析によって得られた統計情報。
        output_file (Path): 出力先のファイルパス。
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 欠損値処理ログ サマリーレポート（実務レベル）\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 処理ファイル数
        f.write(f"📁 処理ファイル数: {stats['total_files']}ファイル\n\n")
        
        # IDM削除統計
        if stats['idm_deletions']:
            total_idm = sum(stats['idm_deletions'])
            f.write("🎯 IDM欠損値削除処理:\n")
            f.write(f"   • 処理回数: {len(stats['idm_deletions'])}回\n")
            f.write(f"   • 総削除行数: {total_idm:,}行\n")
            f.write(f"   • 平均削除行数: {total_idm/len(stats['idm_deletions']):.1f}行\n\n")
        
        # グレード推定統計
        if stats['grade_estimations']:
            total_grade = sum(stats['grade_estimations'])
            f.write("🏆 グレード推定処理:\n")
            f.write(f"   • 処理回数: {len(stats['grade_estimations'])}回\n")
            f.write(f"   • 総推定件数: {total_grade:,}件\n")
            f.write(f"   • 平均推定件数: {total_grade/len(stats['grade_estimations']):.1f}件\n\n")
        
        # 中央値補完統計
        if stats['median_imputations']:
            f.write("🔢 中央値補完処理:\n")
            for column, counts in stats['median_imputations'].items():
                total_count = sum(counts)
                f.write(f"   • {column}: {len(counts)}回, 総補完{total_count:,}件 (平均{total_count/len(counts):.1f}件)\n")
            f.write("\n")
        
        # 高欠損率列削除
        if stats['dropped_columns']:
            f.write("❌ 高欠損率により削除された列:\n")
            sorted_columns = sorted(stats['dropped_columns'])
            for i, column in enumerate(sorted_columns, 1):
                f.write(f"   {i:2d}. {column}\n")
            f.write(f"\n   📊 削除列数: {len(sorted_columns)}列\n\n")
        
        # カテゴリ補完統計
        if stats['categorical_imputations']:
            f.write("🏷️ カテゴリ補完処理:\n")
            for column, values in stats['categorical_imputations'].items():
                total_count = sum(count for _, count in values)
                unique_values = len(set(value for value, _ in values))
                f.write(f"   • {column}: {len(values)}回, 総補完{total_count:,}件, {unique_values}種類の値\n")
            f.write("\n")
        
        # その他補完統計
        if stats['other_imputations']:
            f.write("🔧 その他補完処理:\n")
            for column, values in stats['other_imputations'].items():
                total_count = sum(count for _, count in values)
                f.write(f"   • {column}: {len(values)}回, 総補完{total_count:,}件\n")
            f.write("\n")
        
        # 最終データ統計
        if stats['final_shapes']:
            total_rows = sum(rows for rows, _ in stats['final_shapes'])
            total_cols = sum(cols for _, cols in stats['final_shapes'])
            avg_rows = total_rows / len(stats['final_shapes']) if stats['final_shapes'] else 0
            avg_cols = total_cols / len(stats['final_shapes']) if stats['final_shapes'] else 0
            
            f.write("📊 最終データ統計:\n")
            f.write(f"   • 総行数: {total_rows:,}行\n")
            f.write(f"   • 平均行数: {avg_rows:.1f}行/ファイル\n")
            f.write(f"   • 平均列数: {avg_cols:.1f}列/ファイル\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("🎉 実務レベル欠損値処理 完了サマリー\n")
        f.write("=" * 80 + "\n")


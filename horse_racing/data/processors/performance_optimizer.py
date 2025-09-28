"""
パフォーマンス最適化モジュール
大量データ処理に対応した実務レベルの最適化機能

主要機能:
1. チャンク処理: 大量データの分割処理
2. 並列処理: マルチプロセス活用
3. データ型最適化: メモリ使用量削減
4. メモリ監視・制御: 自動メモリ管理
"""

import pandas as pd
import numpy as np
import logging
import psutil
import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Iterator, Tuple
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings

logger = logging.getLogger(__name__)

# モジュールレベルの関数定義（pickle対応）
def _process_single_file_for_parallel(args_tuple):
    """並列処理用の単一ファイル処理関数（引数をタプルで受け取る）"""
    file_path, process_func, output_dir, use_chunking, optimize_dtypes, memory_manager, dtype_optimizer, chunk_processor = args_tuple
    
    try:
        output_path = output_dir / file_path.name
        
        if use_chunking and chunk_processor:
            # チャンク処理の場合、enhanced_process_funcを直接定義
            def enhanced_process_func(df: pd.DataFrame) -> pd.DataFrame:
                processed_df = process_func(df)
                if optimize_dtypes and dtype_optimizer:
                    optimized_df, _ = dtype_optimizer.optimize_dtypes(processed_df)
                    return optimized_df
                return processed_df
            
            # チャンク処理
            return chunk_processor.process_csv_in_chunks(
                file_path, enhanced_process_func, output_path
            )
        else:
            # 通常処理（ローカル関数を使わずに直接処理）
            df = pd.read_csv(file_path, encoding='utf-8')
            processed_df = process_func(df)
            
            # データ型最適化
            if optimize_dtypes and dtype_optimizer:
                processed_df, _ = dtype_optimizer.optimize_dtypes(processed_df)
            
            processed_df.to_csv(output_path, index=False, encoding='utf-8')
            
            return {
                'success': True,
                'total_rows_processed': len(df),
                'total_rows_output': len(processed_df),
                'output_file': str(output_path)
            }
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"ファイル処理エラー {file_path}: {str(e)}")
        return {'success': False, 'error': str(e), 'file': str(file_path)}


class MemoryManager:
    """
    メモリ使用量の監視・制御クラス
    実務レベルのメモリ管理機能
    """
    
    def __init__(self, 
                 memory_threshold_gb: float = 4.0,
                 warning_threshold_gb: float = 6.0,
                 emergency_threshold_gb: float = 8.0):
        """
        Args:
            memory_threshold_gb: メモリ使用量の推奨閾値（GB）
            warning_threshold_gb: 警告レベルの閾値（GB）
            emergency_threshold_gb: 緊急停止レベルの閾値（GB）
        """
        self.memory_threshold = memory_threshold_gb
        self.warning_threshold = warning_threshold_gb
        self.emergency_threshold = emergency_threshold_gb
        self.initial_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量（GB）を取得"""
        return psutil.virtual_memory().used / 1024 / 1024 / 1024
    
    def _get_available_memory(self) -> float:
        """利用可能メモリ量（GB）を取得"""
        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    
    def check_memory_status(self, stage_name: str = "") -> Dict[str, Any]:
        """
        メモリ状態チェック
        
        Returns:
            メモリ状態情報
        """
        current_memory = self._get_memory_usage()
        available_memory = self._get_available_memory()
        memory_diff = current_memory - self.initial_memory
        
        status = {
            'current_memory_gb': current_memory,
            'available_memory_gb': available_memory,
            'memory_diff_gb': memory_diff,
            'status': 'OK',
            'action_required': False,
            'recommendations': []
        }
        
        # メモリ状態の判定
        if current_memory >= self.emergency_threshold:
            status['status'] = 'EMERGENCY'
            status['action_required'] = True
            status['recommendations'].append('即座に処理を停止してください')
            logger.critical(f"🚨 [{stage_name}] 緊急: メモリ使用量 {current_memory:.1f}GB > {self.emergency_threshold:.1f}GB")
            
        elif current_memory >= self.warning_threshold:
            status['status'] = 'WARNING'
            status['action_required'] = True
            status['recommendations'].extend([
                'チャンク処理サイズを小さくする',
                '並列処理数を減らす',
                'ガベージコレクションを実行する'
            ])
            logger.warning(f"⚠️ [{stage_name}] 警告: メモリ使用量 {current_memory:.1f}GB > {self.warning_threshold:.1f}GB")
            
        elif current_memory >= self.memory_threshold:
            status['status'] = 'CAUTION'
            status['recommendations'].append('メモリ使用量を監視してください')
            logger.info(f"💡 [{stage_name}] 注意: メモリ使用量 {current_memory:.1f}GB > {self.memory_threshold:.1f}GB")
        
        else:
            logger.info(f"✅ [{stage_name}] メモリ状態良好: {current_memory:.1f}GB (利用可能: {available_memory:.1f}GB)")
        
        return status
    
    def auto_cleanup(self, force_gc: bool = True) -> float:
        """
        自動メモリクリーンアップ
        
        Args:
            force_gc: 強制ガベージコレクション実行
            
        Returns:
            解放されたメモリ量（GB）
        """
        before_memory = self._get_memory_usage()
        
        if force_gc:
            logger.info("🧹 ガベージコレクション実行中...")
            gc.collect()
            
        after_memory = self._get_memory_usage()
        freed_memory = before_memory - after_memory
        
        if freed_memory > 0.1:  # 100MB以上解放された場合
            logger.info(f"✅ メモリクリーンアップ完了: {freed_memory:.2f}GB解放")
        
        return freed_memory
    
    def suggest_chunk_size(self, 
                          data_size_mb: float, 
                          target_memory_gb: float = None) -> int:
        """
        最適なチャンクサイズを提案
        
        Args:
            data_size_mb: データサイズ（MB）
            target_memory_gb: 目標メモリ使用量（GB）
            
        Returns:
            推奨チャンクサイズ（行数）
        """
        if target_memory_gb is None:
            target_memory_gb = self.memory_threshold * 0.8  # 80%を目安
        
        available_memory = self._get_available_memory()
        safe_memory = min(target_memory_gb, available_memory * 0.7)  # 70%を安全域とする
        
        # データサイズに基づくチャンクサイズ計算
        chunk_memory_gb = safe_memory / 4  # 処理のオーバーヘッドを考慮
        chunk_size_mb = chunk_memory_gb * 1024
        
        # 推定行数（1行あたり平均1KBと仮定）
        estimated_chunk_rows = int(chunk_size_mb * 1000)
        
        # 最小・最大値の設定
        min_chunk_size = 1000
        max_chunk_size = 50000
        
        chunk_size = max(min_chunk_size, min(estimated_chunk_rows, max_chunk_size))
        
        logger.info(f"💡 推奨チャンクサイズ: {chunk_size:,}行 (目標メモリ: {safe_memory:.1f}GB)")
        
        return chunk_size

class ChunkProcessor:
    """
    チャンク処理クラス
    大量データを分割して効率的に処理
    """
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 memory_manager: MemoryManager = None):
        """
        Args:
            chunk_size: チャンクサイズ（行数）
            memory_manager: メモリ管理クラス
        """
        self.chunk_size = chunk_size
        self.memory_manager = memory_manager or MemoryManager()
        
    def process_csv_in_chunks(self, 
                             file_path: Path,
                             process_func: Callable[[pd.DataFrame], pd.DataFrame],
                             output_path: Path,
                             **read_csv_kwargs) -> Dict[str, Any]:
        """
        CSVファイルをチャンク単位で処理
        
        Args:
            file_path: 入力ファイルパス
            process_func: 処理関数
            output_path: 出力ファイルパス
            **read_csv_kwargs: pandas.read_csv()の追加引数
            
        Returns:
            処理結果サマリー
        """
        logger.info(f"🔄 チャンク処理開始: {file_path.name}")
        
        start_time = time.time()
        total_rows_processed = 0
        total_rows_output = 0
        chunk_count = 0
        
        # デフォルトのread_csv引数
        default_kwargs = {
            'encoding': 'utf-8',
            'chunksize': self.chunk_size
        }
        default_kwargs.update(read_csv_kwargs)
        
        # 出力ファイルの初期化（ヘッダー書き込み用）
        first_chunk = True
        
        try:
            # チャンク単位での読み込み・処理
            for chunk_df in pd.read_csv(file_path, **default_kwargs):
                chunk_count += 1
                chunk_start_time = time.time()
                
                logger.info(f"   📦 チャンク {chunk_count}: {len(chunk_df):,}行処理中...")
                
                # メモリ状態チェック
                memory_status = self.memory_manager.check_memory_status(f"チャンク{chunk_count}")
                
                if memory_status['status'] == 'EMERGENCY':
                    logger.error("❌ メモリ不足により処理を中断します")
                    break
                
                if memory_status['status'] == 'WARNING':
                    # メモリクリーンアップを実行
                    self.memory_manager.auto_cleanup()
                
                # データ処理の実行
                try:
                    processed_chunk = process_func(chunk_df)
                    
                    # 出力ファイルへの書き込み
                    if first_chunk:
                        processed_chunk.to_csv(output_path, index=False, encoding='utf-8')
                        first_chunk = False
                    else:
                        processed_chunk.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')
                    
                    total_rows_processed += len(chunk_df)
                    total_rows_output += len(processed_chunk)
                    
                    chunk_time = time.time() - chunk_start_time
                    logger.info(f"   ✅ チャンク {chunk_count} 完了: {len(processed_chunk):,}行出力 ({chunk_time:.1f}秒)")
                    
                except Exception as e:
                    logger.error(f"   ❌ チャンク {chunk_count} 処理エラー: {str(e)}")
                    continue
                
                # メモリクリーンアップ（定期実行）
                if chunk_count % 5 == 0:
                    self.memory_manager.auto_cleanup()
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'total_chunks': chunk_count,
                'total_rows_processed': total_rows_processed,
                'total_rows_output': total_rows_output,
                'processing_time_seconds': processing_time,
                'output_file': str(output_path)
            }
            
            logger.info(f"✅ チャンク処理完了: {file_path.name}")
            logger.info(f"   📊 処理サマリー: {chunk_count}チャンク, {total_rows_processed:,}行 → {total_rows_output:,}行")
            logger.info(f"   ⏱️ 処理時間: {processing_time:.1f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ チャンク処理エラー: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_chunks': chunk_count,
                'total_rows_processed': total_rows_processed
            }

class ParallelProcessor:
    """
    並列処理クラス
    マルチプロセス・マルチスレッドでの効率的処理
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 memory_manager: MemoryManager = None):
        """
        Args:
            max_workers: 最大ワーカー数（Noneの場合は自動設定）
            memory_manager: メモリ管理クラス
        """
        if max_workers is None:
            # CPUコア数とメモリ状況を考慮して自動設定
            cpu_cores = cpu_count()
            available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
            
            # メモリ1GBあたり1ワーカーを基準とし、CPUコア数も考慮
            memory_based_workers = max(1, int(available_memory / 2))  # 2GBあたり1ワーカー
            max_workers = min(cpu_cores, memory_based_workers, 8)  # 最大8ワーカー
        
        self.max_workers = max_workers
        self.memory_manager = memory_manager or MemoryManager()
        
        logger.info(f"🔧 並列処理設定: {self.max_workers}ワーカー (CPU: {cpu_count()}コア)")
    
    def process_files_parallel(self,
                              items: List[Any],
                              process_func: Callable[[Any], Any],
                              use_threading: bool = False) -> List[Any]:
        """
        複数アイテムの並列処理（汎用版）
        
        Args:
            items: 処理対象アイテムリスト（ファイルパスまたは任意のオブジェクト）
            process_func: 処理関数
            use_threading: ThreadPoolExecutorを使用するか（IO処理向け）
            
        Returns:
            処理結果リスト
        """
        logger.info(f"🚀 並列処理開始: {len(items)}アイテム, {self.max_workers}ワーカー")
        
        start_time = time.time()
        results = []
        
        # 並列処理の実行
        executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # 並列タスクの投入
                future_to_item = {
                    executor.submit(process_func, item): item 
                    for item in items
                }
                
                # 結果の回収
                completed_count = 0
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # アイテムの表示用名前を取得
                        item_name = self._get_item_name(item)
                        logger.info(f"   ✅ 完了 ({completed_count}/{len(items)}): {item_name}")
                        
                    except Exception as e:
                        item_name = self._get_item_name(item)
                        logger.error(f"   ❌ エラー ({completed_count}/{len(items)}): {item_name} - {str(e)}")
                        results.append({'error': str(e), 'item': str(item)})
                    
                    # メモリ状態のチェック
                    if completed_count % max(1, len(items) // 4) == 0:  # 25%ごとにチェック
                        memory_status = self.memory_manager.check_memory_status(f"並列処理 {completed_count}/{len(items)}")
                        if memory_status['status'] in ['WARNING', 'EMERGENCY']:
                            self.memory_manager.auto_cleanup()
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ 並列処理完了: {len(results)}件処理, {processing_time:.1f}秒")
            logger.info(f"   📊 平均処理時間: {processing_time/len(items):.2f}秒/ファイル")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 並列処理エラー: {str(e)}")
            return []
    
    def _get_item_name(self, item) -> str:
        """アイテムの表示用名前を取得"""
        if hasattr(item, 'name'):
            return str(item.name)
        elif isinstance(item, tuple) and len(item) > 0:
            # タプルの場合、最初の要素がファイルパスと仮定
            first_item = item[0]
            if hasattr(first_item, 'name'):
                return str(first_item.name)
            else:
                return str(first_item)
        else:
            return str(item)

class DataTypeOptimizer:
    """
    データ型最適化クラス
    メモリ使用量の削減とパフォーマンス向上
    """
    
    def __init__(self):
        # 最適化ルール定義
        self.optimization_rules = {
            'integer_columns': {
                'small': {'min': -128, 'max': 127, 'dtype': 'int8'},
                'medium': {'min': -32768, 'max': 32767, 'dtype': 'int16'},
                'large': {'min': -2147483648, 'max': 2147483647, 'dtype': 'int32'},
                'default': 'int64'
            },
            'float_columns': {
                'precision_threshold': 6,  # 有効桁数6桁以下ならfloat32
                'default_float': 'float32',
                'high_precision': 'float64'
            },
            'categorical_threshold': 0.5,  # 50%以下の一意値率でカテゴリ化
            'boolean_keywords': ['is_', 'has_', 'flag_', '_flg']
        }
    
    def optimize_dtypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        データ型の最適化
        
        Args:
            df: 最適化対象のDataFrame
            
        Returns:
            最適化後のDataFrame, 最適化レポート
        """
        logger.info("🔧 データ型最適化開始...")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        optimization_report = {
            'original_memory_mb': original_memory,
            'optimizations': {},
            'errors': []
        }
        
        df_optimized = df.copy()
        
        try:
            # 1. 整数列の最適化
            df_optimized, int_report = self._optimize_integer_columns(df_optimized)
            optimization_report['optimizations']['integers'] = int_report
            
            # 2. 浮動小数点列の最適化
            df_optimized, float_report = self._optimize_float_columns(df_optimized)
            optimization_report['optimizations']['floats'] = float_report
            
            # 3. カテゴリ列の最適化
            df_optimized, cat_report = self._optimize_categorical_columns(df_optimized)
            optimization_report['optimizations']['categoricals'] = cat_report
            
            # 4. ブール列の最適化
            df_optimized, bool_report = self._optimize_boolean_columns(df_optimized)
            optimization_report['optimizations']['booleans'] = bool_report
            
            # 最適化後のメモリ使用量
            optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            memory_reduction = original_memory - optimized_memory
            reduction_rate = (memory_reduction / original_memory) * 100 if original_memory > 0 else 0
            
            optimization_report.update({
                'optimized_memory_mb': optimized_memory,
                'memory_reduction_mb': memory_reduction,
                'reduction_rate_percent': reduction_rate
            })
            
            logger.info(f"✅ データ型最適化完了:")
            logger.info(f"   💾 メモリ使用量: {original_memory:.1f}MB → {optimized_memory:.1f}MB")
            logger.info(f"   📉 削減量: {memory_reduction:.1f}MB ({reduction_rate:.1f}%削減)")
            
            return df_optimized, optimization_report
            
        except Exception as e:
            logger.error(f"❌ データ型最適化エラー: {str(e)}")
            optimization_report['errors'].append(str(e))
            return df, optimization_report
    
    def _optimize_integer_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """整数列の最適化"""
        int_columns = df.select_dtypes(include=['int']).columns
        optimizations = {}
        
        for col in int_columns:
            if df[col].notna().sum() == 0:  # 全て欠損値の場合はスキップ
                continue
                
            try:
                col_min = df[col].min()
                col_max = df[col].max()
                
                # 適切なinteger型を選択
                rules = self.optimization_rules['integer_columns']
                new_dtype = rules['default']
                
                for size, rule in rules.items():
                    if size == 'default':
                        continue
                    if col_min >= rule['min'] and col_max <= rule['max']:
                        new_dtype = rule['dtype']
                        break
                
                if new_dtype != str(df[col].dtype):
                    df[col] = df[col].astype(new_dtype)
                    optimizations[col] = f"{df[col].dtype} → {new_dtype}"
                    
            except Exception as e:
                logger.warning(f"整数列 {col} の最適化に失敗: {str(e)}")
        
        return df, optimizations
    
    def _optimize_float_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """浮動小数点列の最適化"""
        float_columns = df.select_dtypes(include=['float']).columns
        optimizations = {}
        
        for col in float_columns:
            if df[col].notna().sum() == 0:  # 全て欠損値の場合はスキップ
                continue
                
            try:
                # float32で十分な精度か判定
                original_dtype = str(df[col].dtype)
                
                if original_dtype == 'float64':
                    # float32に変換して精度をチェック
                    test_series = df[col].astype('float32')
                    
                    # 元の値との差が許容範囲内か確認
                    if df[col].notna().sum() > 0:
                        max_diff = abs(df[col] - test_series).max()
                        relative_error = max_diff / abs(df[col]).max() if abs(df[col]).max() > 0 else 0
                        
                        if relative_error < 1e-6:  # 相対誤差が十分小さい
                            df[col] = test_series
                            optimizations[col] = f"float64 → float32"
                            
            except Exception as e:
                logger.warning(f"浮動小数点列 {col} の最適化に失敗: {str(e)}")
        
        return df, optimizations
    
    def _optimize_categorical_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """カテゴリ列の最適化"""
        object_columns = df.select_dtypes(include=['object']).columns
        optimizations = {}
        
        for col in object_columns:
            if df[col].notna().sum() == 0:  # 全て欠損値の場合はスキップ
                continue
                
            try:
                # 一意値の割合を計算
                unique_ratio = df[col].nunique() / len(df)
                
                if unique_ratio <= self.optimization_rules['categorical_threshold']:
                    df[col] = df[col].astype('category')
                    optimizations[col] = f"object → category (一意値率: {unique_ratio:.2%})"
                    
            except Exception as e:
                logger.warning(f"カテゴリ列 {col} の最適化に失敗: {str(e)}")
        
        return df, optimizations
    
    def _optimize_boolean_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """ブール列の最適化"""
        optimizations = {}
        
        # ブールっぽい列名を検出
        boolean_candidates = []
        for col in df.columns:
            for keyword in self.optimization_rules['boolean_keywords']:
                if keyword in col.lower():
                    boolean_candidates.append(col)
                    break
        
        # 0,1のみの数値列もブール候補
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                unique_values = set(df[col].dropna().unique())
                if unique_values.issubset({0, 1, 0.0, 1.0}):
                    boolean_candidates.append(col)
        
        # ブール型に変換
        for col in boolean_candidates:
            if col in df.columns:
                try:
                    original_dtype = str(df[col].dtype)
                    df[col] = df[col].astype('bool')
                    optimizations[col] = f"{original_dtype} → bool"
                    
                except Exception as e:
                    logger.warning(f"ブール列 {col} の最適化に失敗: {str(e)}")
        
        return df, optimizations

class PerformanceOptimizer:
    """
    統合パフォーマンス最適化クラス
    チャンク処理・並列処理・データ型最適化の統合管理
    """
    
    def __init__(self, 
                 auto_tune: bool = True,
                 memory_limit_gb: float = 6.0,
                 max_workers: int = None):
        """
        Args:
            auto_tune: 自動チューニング有効化
            memory_limit_gb: メモリ使用量上限
            max_workers: 最大ワーカー数
        """
        self.auto_tune = auto_tune
        self.memory_manager = MemoryManager(
            memory_threshold_gb=memory_limit_gb * 0.6,
            warning_threshold_gb=memory_limit_gb * 0.8,
            emergency_threshold_gb=memory_limit_gb
        )
        self.chunk_processor = ChunkProcessor(memory_manager=self.memory_manager)
        self.parallel_processor = ParallelProcessor(max_workers=max_workers, memory_manager=self.memory_manager)
        self.dtype_optimizer = DataTypeOptimizer()
        
        logger.info("🚀 パフォーマンス最適化システム初期化完了")
    
    def optimize_data_processing(self,
                                file_paths: List[Path],
                                process_func: Callable[[pd.DataFrame], pd.DataFrame],
                                output_dir: Path,
                                use_chunking: bool = None,
                                use_parallel: bool = None,
                                optimize_dtypes: bool = True) -> Dict[str, Any]:
        """
        最適化されたデータ処理の実行
        
        Args:
            file_paths: 処理対象ファイルリスト
            process_func: データ処理関数
            output_dir: 出力ディレクトリ
            use_chunking: チャンク処理使用フラグ（Noneの場合は自動判定）
            use_parallel: 並列処理使用フラグ（Noneの場合は自動判定）
            optimize_dtypes: データ型最適化実行フラグ
            
        Returns:
            処理結果サマリー
        """
        logger.info("🎯 最適化データ処理開始")
        
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自動チューニング
        if self.auto_tune:
            use_chunking, use_parallel = self._auto_tune_processing_strategy(file_paths)
        
        # 処理戦略の決定
        if use_parallel and len(file_paths) > 1:
            logger.info("🔄 並列処理モードで実行")
            results = self._process_files_parallel(file_paths, process_func, output_dir, use_chunking, optimize_dtypes)
        else:
            logger.info("📝 順次処理モードで実行")
            results = self._process_files_sequential(file_paths, process_func, output_dir, use_chunking, optimize_dtypes)
        
        # 処理サマリー
        processing_time = time.time() - start_time
        
        summary = {
            'total_files': len(file_paths),
            'successful_files': sum(1 for r in results if r.get('success', False)),
            'failed_files': sum(1 for r in results if not r.get('success', False)),
            'total_processing_time': processing_time,
            'strategy': {
                'chunking': use_chunking,
                'parallel': use_parallel,
                'dtype_optimization': optimize_dtypes
            },
            'results': results
        }
        
        logger.info(f"✅ 最適化データ処理完了:")
        logger.info(f"   📊 処理結果: {summary['successful_files']}/{summary['total_files']}ファイル成功")
        logger.info(f"   ⏱️ 総処理時間: {processing_time:.1f}秒")
        
        return summary
    
    def _auto_tune_processing_strategy(self, file_paths: List[Path]) -> Tuple[bool, bool]:
        """処理戦略の自動チューニング"""
        # ファイルサイズとシステムリソースから最適戦略を決定
        total_size_mb = sum(f.stat().st_size for f in file_paths if f.exists()) / 1024 / 1024
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        
        # チャンク処理の判定
        use_chunking = total_size_mb > (available_memory_gb * 1024 * 0.3)  # 利用可能メモリの30%以上
        
        # 並列処理の判定
        use_parallel = len(file_paths) > 1 and available_memory_gb > 2.0  # 2GB以上の場合
        
        logger.info(f"🔧 自動チューニング結果:")
        logger.info(f"   📊 総ファイルサイズ: {total_size_mb:.1f}MB")
        logger.info(f"   💾 利用可能メモリ: {available_memory_gb:.1f}GB")
        logger.info(f"   🔄 チャンク処理: {'有効' if use_chunking else '無効'}")
        logger.info(f"   🚀 並列処理: {'有効' if use_parallel else '無効'}")
        
        return use_chunking, use_parallel
    
    def _process_files_parallel(self, file_paths, process_func, output_dir, use_chunking, optimize_dtypes):
        """並列処理での実行（ThreadPoolExecutor強制使用でpickleエラー回避）"""
        # 引数タプルを準備
        args_tuples = [
            (
                file_path,
                process_func,
                output_dir,
                use_chunking,
                optimize_dtypes,
                self.memory_manager,
                self.dtype_optimizer,
                self.chunk_processor
            )
            for file_path in file_paths
        ]
        
        # 強制的にThreadPoolExecutorを使用してpickleエラーを回避
        return self.parallel_processor.process_files_parallel(
            args_tuples, 
            _process_single_file_for_parallel,
            use_threading=True  # 強制的にThreadingを使用
        )
    
    def _process_files_sequential(self, file_paths, process_func, output_dir, use_chunking, optimize_dtypes):
        """順次処理での実行"""
        results = []
        for file_path in file_paths:
            result = self._process_single_file(file_path, process_func, output_dir, use_chunking, optimize_dtypes)
            results.append(result)
        return results
    
    def _process_single_file(self, file_path: Path, process_func, output_dir: Path, use_chunking: bool, optimize_dtypes: bool) -> Dict[str, Any]:
        """単一ファイルの処理"""
        try:
            output_path = output_dir / file_path.name
            
            # データ型最適化を含む処理関数を作成
            def enhanced_process_func(df: pd.DataFrame) -> pd.DataFrame:
                processed_df = process_func(df)
                
                if optimize_dtypes:
                    optimized_df, _ = self.dtype_optimizer.optimize_dtypes(processed_df)
                    return optimized_df
                
                return processed_df
            
            if use_chunking:
                # チャンク処理
                return self.chunk_processor.process_csv_in_chunks(
                    file_path, enhanced_process_func, output_path
                )
            else:
                # 通常処理
                df = pd.read_csv(file_path, encoding='utf-8')
                processed_df = enhanced_process_func(df)
                processed_df.to_csv(output_path, index=False, encoding='utf-8')
                
                return {
                    'success': True,
                    'total_rows_processed': len(df),
                    'total_rows_output': len(processed_df),
                    'output_file': str(output_path)
                }
                
        except Exception as e:
            logger.error(f"ファイル処理エラー {file_path}: {str(e)}")
            return {'success': False, 'error': str(e), 'file': str(file_path)} 
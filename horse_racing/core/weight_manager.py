"""
競馬分析用の動的重み管理モジュール
レポート5.1.3節準拠の重み計算とグローバル設定を管理します。
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class WeightManager:
    """
    レポート5.1.3節準拠の動的重み計算とグローバル管理
    
    使用方法:
    1. 処理開始時: WeightManager.initialize_from_training_data(df)
    2. 各モジュール: WeightManager.get_weights()
    """
    
    # グローバル重み変数（全モジュールで共有）
    _global_weights: Optional[Dict[str, float]] = None
    _calculation_details: Optional[Dict[str, Any]] = None
    _initialized: bool = False
    _data_hash: Optional[str] = None  # データ変更検出用ハッシュ
    
    @classmethod
    def initialize_from_training_data(cls, df: pd.DataFrame, force_recalculate: bool = False) -> Dict[str, float]:
        """
        訓練期間（2010-2020年）データから動的重みを算出し、グローバル設定
        レポート5.1.3節準拠の重み計算実装
        
        Args:
            df: 全データセット
            force_recalculate: 再計算を強制するか
            
        Returns:
            計算された重み辞書
        """
        # データ変更の検出
        current_data_hash = cls._calculate_data_hash(df)
        data_changed = cls._data_hash != current_data_hash
        
        if cls._initialized and not force_recalculate and not data_changed:
            logger.info("✅ 重みは既に初期化済みです（データ変更なし）。")
            return cls._global_weights
        
        if data_changed and cls._initialized:
            logger.info("🔄 データ変更を検出しました。重みを再計算します。")
        elif force_recalculate:
            logger.info("🔄 強制再計算が指定されました。重みを再計算します。")
            
        logger.info("🎯 循環論理回避版の動的重み計算を開始...")
        logger.info("📋 改善版計算式: w_i = r²(feature_i, win_rate) / Σr²(feature_j, win_rate)")
        logger.info("📋 訓練期間: 2010-2020年（循環論理回避版）")
        logger.info("🔧 改善点: 重み算出に勝率を使用し、予測目的（複勝率）と分離")
        
        try:
            # 訓練期間（2010-2020年）のデータを分離
            if '年' in df.columns:
                train_data = df[(df['年'] >= 2010) & (df['年'] <= 2020)].copy()
                logger.info(f"📊 訓練期間フィルタリング後: {len(train_data):,}行")
                
                # 特徴量カラムの存在確認
                required_feature_cols = ['grade_level', 'venue_level', 'distance_level']
                missing_feature_cols = [col for col in required_feature_cols if col not in train_data.columns]
                
                if missing_feature_cols:
                    logger.warning(f"⚠️ 訓練データに特徴量カラムが不足: {missing_feature_cols}")
                    logger.warning("📊 全データで計算し直します...")
                    train_data = df.copy()
                else:
                    logger.info("✅ 訓練データに全ての特徴量カラムが存在")
                    
                if len(train_data) == 0:
                    logger.warning("⚠️ 訓練期間（2010-2020年）データがありません。全データで計算します。")
                    train_data = df.copy()
            else:
                logger.warning("⚠️ '年'列が見つかりません。全データで計算します。")
                train_data = df.copy()
                
            # 最終的なデータの特徴量カラムチェック
            final_feature_cols = [col for col in train_data.columns if col.endswith('_level')]
            logger.info(f"📊 最終計算データの特徴量カラム: {final_feature_cols}")
            
            logger.info("📊 訓練期間データでの動的重み計算:")
            logger.info(f"   対象データ: {len(train_data):,}行")
            if '年' in train_data.columns:
                logger.info(f"   対象期間: {train_data['年'].min()}-{train_data['年'].max()}年")
            
            # 循環論理回避のため勝率ベース相関計算を使用
            try:
                correlations = cls._calculate_feature_correlations_with_win_rate(train_data)
                
                # 特徴量レベルが存在しない場合はフォールバック重みを使用
                if correlations is None or all(corr == 0.0 for corr in correlations.values()):
                    logger.warning("⚠️ 相関計算に失敗しました。循環論理回避版の固定重みを使用します。")
                    return cls._get_fallback_weights()
                    
                logger.info("✅ 勝率ベース相関計算が成功しました")
                
            except Exception as e:
                logger.error(f"❌ 勝率ベース相関計算中にエラー: {e}")
                logger.warning("⚠️ 循環論理回避版の固定重みを使用します。")
                return cls._get_fallback_weights()
            
            # レポート5.1.3節準拠の重み計算
            weights = cls._calculate_weights_report_compliant(correlations)
            
            # グローバル設定
            cls._global_weights = weights
            cls._calculation_details = {
                'correlations': correlations,
                'training_period': f"{train_data['年'].min()}-{train_data['年'].max()}",
                'sample_size': len(train_data),
                'target_column': 'place_rate'
            }
            cls._data_hash = current_data_hash  # データハッシュを保存
            cls._initialized = True
            
            # 結果の表示
            cls._display_calculation_results(weights, correlations)
            
            # 📝 ログにも重み情報を出力
            cls._log_weight_calculation_results(weights, correlations, train_data)
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ 動的重み計算エラー: {str(e)}")
            return cls._get_fallback_weights()
    
    @classmethod
    def get_weights(cls) -> Dict[str, float]:
        """
        現在設定されている重みを取得
        
        Returns:
            重み辞書
        """
        if not cls._initialized or cls._global_weights is None:
            logger.warning("⚠️ 重みが未初期化です。フォールバック重みを返します。")
            logger.warning(f"   📊 _initialized: {cls._initialized}")
            logger.warning(f"   📊 _global_weights存在: {cls._global_weights is not None}")
            return cls._get_fallback_weights()
        
        logger.info("✅ グローバル重みを正常に取得しました")
        return cls._global_weights.copy()
    
    @classmethod
    def get_calculation_details(cls) -> Optional[Dict[str, Any]]:
        """
        重み計算の詳細情報を取得
        
        Returns:
            計算詳細辞書
        """
        return cls._calculation_details.copy() if cls._calculation_details else None
    
    @classmethod
    def is_initialized(cls) -> bool:
        """
        初期化状態を確認
        
        Returns:
            初期化済みかどうか
        """
        return cls._initialized
    
    @classmethod
    def reset(cls):
        """
        重み設定をリセット
        """
        cls._global_weights = None
        cls._calculation_details = None
        cls._initialized = False
        cls._data_hash = None
        logger.info("🔄 重み設定をリセットしました。")
    
    @classmethod
    def _calculate_data_hash(cls, df: pd.DataFrame) -> str:
        """
        データの変更を検出するためのハッシュを計算
        
        Args:
            df: データセット
            
        Returns:
            データハッシュ
        """
        import hashlib
        
        # 訓練期間のデータのみでハッシュを計算
        train_data = df[(df['年'] >= 2010) & (df['年'] <= 2020)] if '年' in df.columns else df
        
        # データの基本情報でハッシュを作成
        hash_input = f"{len(train_data)}_{train_data.shape[1]}"
        
        # 馬名とレース数の情報を追加（データの実質的な変更を検出）
        if '馬名' in train_data.columns:
            unique_horses = len(train_data['馬名'].unique())
            hash_input += f"_{unique_horses}"
        
        # 年の範囲を追加
        if '年' in train_data.columns:
            year_range = f"{train_data['年'].min()}_{train_data['年'].max()}"
            hash_input += f"_{year_range}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    
    @classmethod
    def _calculate_feature_correlations_with_win_rate(cls, df: pd.DataFrame) -> Dict[str, float]:
        """
        勝率（1着率）ベースの相関計算（循環論理回避版）
        
        Args:
            df: データフレーム（個別レース結果）
            
        Returns:
            相関辞書（勝率ベース）
        """
        logger.info("🔍 勝率ベースの相関計算を開始（循環論理回避版）...")
        logger.info("📋 改善理由: 目的変数（複勝率）≠重み算出基準（勝率）で循環論理を回避")
        
        # データフレームの詳細情報をログ出力
        logger.info(f"📊 入力データフレーム情報.")
        logger.info(f"   行数: {len(df):,}")
        logger.info(f"   列数: {len(df.columns)}")
        logger.info(f"   列名一覧: {list(df.columns)}")
        
        # 利用可能なレベルカラムを動的に検出
        available_level_cols = [col for col in df.columns if col.endswith('_level')]
        logger.info(f"📊 利用可能なレベルカラム: {available_level_cols}")
        
        # カラム名の解決（マージ時の重複カラム対応）
        resolved_cols = {}
        for base_name in ['grade_level', 'venue_level', 'distance_level']:
            if base_name in df.columns:
                resolved_cols[base_name] = base_name
            elif f"{base_name}_x" in df.columns and f"{base_name}_y" in df.columns:
                # マージ時の重複カラムの場合、_xを優先
                resolved_cols[base_name] = f"{base_name}_x"
                logger.info(f"📊 {base_name} を {base_name}_x として解決")
            elif f"{base_name}_x" in df.columns:
                resolved_cols[base_name] = f"{base_name}_x"
                logger.info(f"📊 {base_name} を {base_name}_x として解決")
            elif f"{base_name}_y" in df.columns:
                resolved_cols[base_name] = f"{base_name}_y"
                logger.info(f"📊 {base_name} を {base_name}_y として解決")
            else:
                logger.warning(f"⚠️ {base_name} の解決可能なカラムが見つかりません")
        
        logger.info(f"📊 解決されたカラム: {resolved_cols}")
        
        # 必要カラムの確認
        required_base_cols = ['馬名', '着順']
        missing_base_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_base_cols:
            logger.error(f"❌ 基本カラムが不足: {missing_base_cols}")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # レベルカラムがない場合の対応
        if not resolved_cols:
            logger.warning("⚠️ レベルカラムが見つかりません。フォールバック重みを使用します。")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 勝利フラグを作成（1着のみ）
        df_temp = df.copy()
        df_temp['is_winner'] = (pd.to_numeric(df_temp['着順'], errors='coerce') == 1).astype(int)
        logger.info("📊 着順列から勝利フラグを作成（着順=1のみ）")
        
        # 馬ごとの統計を計算（最低出走数6戦以上）
        # 解決されたカラム名を使用してカウント用カラムを決定
        count_col = resolved_cols.get('grade_level', 'grade_level')
        if count_col not in df_temp.columns:
            # フォールバック: 最初に見つかったレベルカラムを使用
            count_col = list(resolved_cols.values())[0] if resolved_cols else 'grade_level'
        
        logger.info(f"🔍 デバッグ: カウント用カラム: {count_col}")
        logger.info(f"🔍 デバッグ: データフレーム形状: {df_temp.shape}")
        logger.info(f"🔍 デバッグ: 馬名のユニーク数: {df_temp['馬名'].nunique()}")
        logger.info(f"🔍 デバッグ: 馬名の総数: {len(df_temp['馬名'])}")
        
        # 馬ごとの統計を計算
        horse_stats = df_temp.groupby('馬名').agg({
            'is_winner': 'mean',     # 勝率（1着率）
            count_col: 'count'       # 出走回数
        }).reset_index()
        
        logger.info(f"🔍 デバッグ: 馬統計作成後: {len(horse_stats)}頭")
        logger.info(f"🔍 デバッグ: 出走回数分布:")
        logger.info(f"🔍 デバッグ: 最小: {horse_stats[count_col].min()}")
        logger.info(f"🔍 デバッグ: 最大: {horse_stats[count_col].max()}")
        logger.info(f"🔍 デバッグ: 平均: {horse_stats[count_col].mean():.2f}")
        logger.info(f"🔍 デバッグ: 6戦以上の馬数: {(horse_stats[count_col] >= 6).sum()}")
        
        # 最低2戦以上の馬のみ抽出（重み計算用に大幅緩和）
        horse_stats = horse_stats[horse_stats[count_col] >= 2]
        logger.info(f"📊 勝率計算対象: {len(horse_stats)}頭（2戦以上）")
        
        # サンプル数が不足する場合は閾値をさらに下げる
        if len(horse_stats) < 50:
            logger.warning(f"⚠️ 2戦以上の馬が不足: {len(horse_stats)}頭")
            logger.warning("📊 最低出走数を1戦に下げて再計算します...")
            
            # 1戦以上の馬で再計算（全馬対象）
            horse_stats = df_temp.groupby('馬名').agg({
                'is_winner': 'mean',     # 勝率（1着率）
                count_col: 'count'       # 出走回数
            }).reset_index()
            horse_stats = horse_stats[horse_stats[count_col] >= 1]
            logger.info(f"📊 1戦以上の馬: {len(horse_stats)}頭")
            
            if len(horse_stats) < 20:
                logger.warning(f"⚠️ サンプル数が少なすぎます: {len(horse_stats)}頭")
                return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 相関計算（解決されたカラム名を使用）
        correlations = {}
        
        # 解決されたカラム名を使用して相関計算
        for base_name, resolved_name in resolved_cols.items():
            if resolved_name in df.columns:
                # 馬ごとの平均レベルを計算
                avg_feature = df.groupby('馬名')[resolved_name].mean().reset_index()
                avg_feature.columns = ['馬名', f'avg_{base_name}']
                
                # 馬統計データとマージ
                horse_stats_with_feature = horse_stats.merge(avg_feature, on='馬名', how='left')
                
                # 相関計算
                clean_data = horse_stats_with_feature[['is_winner', f'avg_{base_name}']].dropna()
                if len(clean_data) >= 20:
                    from scipy.stats import pearsonr
                    corr, p_value = pearsonr(clean_data['is_winner'], clean_data[f'avg_{base_name}'])
                    
                    # キー名をマッピング
                    key_mapping = {
                        'grade_level': 'grade',
                        'venue_level': 'venue',
                        'distance_level': 'distance'
                    }
                    key_name = key_mapping.get(base_name, base_name)
                    correlations[key_name] = corr
                    
                    logger.info(f"📊 {key_name} vs 勝率: r = {corr:.3f}, p = {p_value:.6f}")
                else:
                    key_mapping = {
                        'grade_level': 'grade',
                        'venue_level': 'venue', 
                        'distance_level': 'distance'
                    }
                    key_name = key_mapping.get(base_name, base_name)
                    correlations[key_name] = 0.0
                    logger.warning(f"⚠️ {key_name} の相関計算に十分なデータがありません")
            else:
                logger.error(f"❌ {base_name} カラムアクセスエラー: '{resolved_name}'")
                logger.error(f"📊 使用可能カラム: {list(df.columns)}")
                key_mapping = {
                    'grade_level': 'grade',
                    'venue_level': 'venue',
                    'distance_level': 'distance'
                }
                key_name = key_mapping.get(base_name, base_name)
                correlations[key_name] = 0.0
        
        
        logger.info("✅ 勝率ベース相関計算完了（循環論理回避）")
        return correlations
    
    @classmethod
    def _calculate_feature_correlations_report_compliant(cls, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        verify_weight_calculation.py準拠の相関計算（馬統計データレベル）
        
        Args:
            df: データフレーム（個別レース結果）
            target_col: 目標変数列名
            
        Returns:
            相関辞書（verify_weight_calculation.py形式）
        """
        logger.info("🔍 verify_weight_calculation.py準拠の相関計算を開始...")
        
        # Phase 1: 馬統計データ作成（レポート5.1.3節準拠）
        logger.info("📊 Phase 1: 馬統計データを作成中...")
        
        # 必要カラムの確認.
        required_cols = ['馬名', '着順', 'grade_level', 'venue_level', 'distance_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ 必要なカラムが不足: {missing_cols}")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 複勝フラグを作成
        if '着順' in df.columns:
            df_temp = df.copy()
            df_temp['is_placed'] = (pd.to_numeric(df_temp['着順'], errors='coerce') <= 3).astype(int)
            logger.info("📊 着順列から複勝フラグを作成（着順<=3）")
        elif '複勝' in df.columns:
            df_temp = df.copy()
            df_temp['is_placed'] = pd.to_numeric(df_temp['複勝'], errors='coerce').fillna(0)
            logger.info("📊 複勝列から複勝フラグを作成")
        else:
            logger.error("❌ 複勝フラグを作成できません")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 馬ごとの統計を計算（最低出走数6戦以上）
        horse_stats = df_temp.groupby('馬名').agg({
            'is_placed': 'mean',  # 複勝率
            'grade_level': 'count'  # 出走回数
        }).reset_index()
        
        # 列名を標準化
        horse_stats.columns = ['馬名', 'place_rate', 'race_count']
        
        # 最低出走数6戦以上でフィルタ（レポート仕様準拠）
        horse_stats = horse_stats[horse_stats['race_count'] >= 6].copy()
        logger.info(f"📊 最低出走数6戦以上でフィルタ: {len(horse_stats):,}頭")
        
        if len(horse_stats) < 100:
            logger.error(f"❌ サンプル数が不足: {len(horse_stats)}頭（最低100頭必要）")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 特徴量レベルの平均を計算
        feature_cols = ['grade_level', 'venue_level', 'distance_level']
        for col in feature_cols:
            avg_feature = df.groupby('馬名')[col].mean().reset_index()
            avg_feature.columns = ['馬名', f'avg_{col}']
            horse_stats = horse_stats.merge(avg_feature, on='馬名', how='left')
        
        logger.info(f"📊 馬統計データ作成完了: {len(horse_stats):,}頭")
        
        # Phase 2: 相関計算（馬統計データベース）
        logger.info("📈 Phase 2: 馬統計データで相関を計算中...")
        
        # 必要な列の確認
        required_corr_cols = ['place_rate', 'avg_grade_level', 'avg_venue_level', 'avg_distance_level']
        missing_corr_cols = [col for col in required_corr_cols if col not in horse_stats.columns]
        
        if missing_corr_cols:
            logger.error(f"❌ 必要な相関列が不足: {missing_corr_cols}")
            logger.info(f"📊 利用可能な列: {list(horse_stats.columns)}")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 欠損値を除去
        clean_data = horse_stats[required_corr_cols].dropna()
        logger.info(f"📊 相関計算用データ: {len(clean_data):,}頭")
        
        if len(clean_data) < 100:
            logger.error(f"❌ 相関計算用サンプル数が不足: {len(clean_data)}頭（最低100頭必要）")
            return {'grade': 0.0, 'venue': 0.0, 'distance': 0.0}
        
        # 相関計算
        correlations = {}
        target = clean_data['place_rate']
        
        # レポート5.1.3節準拠の相関計算
        feature_mapping = {
            'avg_grade_level': 'grade',
            'avg_venue_level': 'venue', 
            'avg_distance_level': 'distance'
        }
        
        for feature_col, feature_name in feature_mapping.items():
            if feature_col in clean_data.columns:
                corr, p_value = pearsonr(clean_data[feature_col], target)
                correlations[feature_name] = corr
                logger.info(f"   📈 {feature_name}_level: r = {corr:.3f}, r² = {corr**2:.3f}, p = {p_value:.3f}")
        
        return correlations
    
    
    @classmethod
    def _calculate_weights_report_compliant(cls, correlations: Dict[str, float]) -> Dict[str, float]:
        """
        レポート5.1.3節準拠の重み計算
        計算式: w_i = r_i² / (r_grade² + r_venue² + r_distance²)
        
        Args:
            correlations: 相関係数辞書
            
        Returns:
            重み辞書
        """
        logger.info("🎯 レポート5.1.3節準拠の重み計算を実行中...")
        logger.info("📋 計算式: w_i = r_i² / (r_grade² + r_venue² + r_distance²)")
        
        # 相関係数の2乗（寄与度）を計算
        r_grade = correlations.get('grade', 0.0)
        r_venue = correlations.get('venue', 0.0)
        r_distance = correlations.get('distance', 0.0)
        
        # 寄与度計算
        contrib_grade = r_grade ** 2
        contrib_venue = r_venue ** 2
        contrib_distance = r_distance ** 2
        total_contrib = contrib_grade + contrib_venue + contrib_distance
        
        logger.info("📊 寄与度計算結果:")
        logger.info(f"   グレード寄与度: r² = {r_grade:.3f}² = {contrib_grade:.3f}")
        logger.info(f"   場所寄与度: r² = {r_venue:.3f}² = {contrib_venue:.3f}")
        logger.info(f"   距離寄与度: r² = {r_distance:.3f}² = {contrib_distance:.3f}")
        logger.info(f"   総寄与度: {total_contrib:.3f}")
        
        if total_contrib == 0:
            logger.warning("⚠️ 総寄与度が0です。フォールバック重みを使用します。")
            return cls._get_fallback_weights()
        
        # 正規化された重み計算
        weight_grade = contrib_grade / total_contrib
        weight_venue = contrib_venue / total_contrib
        weight_distance = contrib_distance / total_contrib
        
        weights = {
            'grade_weight': weight_grade,
            'venue_weight': weight_venue,
            'distance_weight': weight_distance
        }
        
        logger.info("🎯 正規化された重み:")
        logger.info(f"   グレード重み: {weight_grade:.3f} ({weight_grade*100:.1f}%)")
        logger.info(f"   場所重み: {weight_venue:.3f} ({weight_venue*100:.1f}%)")
        logger.info(f"   距離重み: {weight_distance:.3f} ({weight_distance*100:.1f}%)")
        
        # レポート値との比較
        report_weights = cls._get_fallback_weights()
        logger.info("📋 レポート5.1.3節記載値との比較:")
        logger.info(f"   グレード: 計算値{weight_grade:.3f} vs レポート値{report_weights['grade_weight']:.3f}")
        logger.info(f"   場所: 計算値{weight_venue:.3f} vs レポート値{report_weights['venue_weight']:.3f}")
        logger.info(f"   距離: 計算値{weight_distance:.3f} vs レポート値{report_weights['distance_weight']:.3f}")
        
        return weights
    
    @classmethod
    def _calculate_weights_from_correlations(cls, correlations: Dict[str, float]) -> Dict[str, float]:
        """
        相関係数から重みを計算
        
        Args:
            correlations: 相関係数辞書
            
        Returns:
            重み辞書
        """
        # 寄与度計算（相関の2乗）
        contributions = {key: corr ** 2 for key, corr in correlations.items()}
        total_contribution = sum(contributions.values())
        
        logger.info(f"🔍 相関分析結果:")
        for key, corr in correlations.items():
            logger.info(f"   {key}相関: r = {corr:.3f}, r² = {contributions[key]:.3f}")
        logger.info(f"   総寄与度: {total_contribution:.3f}")
        
        # 重み計算
        if total_contribution > 0:
            weights = {
                'grade_weight': contributions['grade'] / total_contribution,
                'venue_weight': contributions['venue'] / total_contribution,
                'distance_weight': contributions['distance'] / total_contribution
            }
        else:
            logger.warning("⚠️ すべての相関が0です。均等重みを使用します。")
            weights = {
                'grade_weight': 1.0 / 3,
                'venue_weight': 1.0 / 3,
                'distance_weight': 1.0 / 3
            }
        
        return weights
    
    @classmethod
    def _display_calculation_results(cls, weights: Dict[str, float], correlations: Dict[str, float]):
        """
        計算結果を表示
        
        Args:
            weights: 重み辞書
            correlations: 相関係数辞書
        """
        logger.info(f"📊 訓練期間（2010-2020年）動的重み算出結果:")
        logger.info(f"   グレード: {weights['grade_weight']:.3f} ({weights['grade_weight']*100:.1f}%)")
        logger.info(f"   場所: {weights['venue_weight']:.3f} ({weights['venue_weight']*100:.1f}%)")
        logger.info(f"   距離: {weights['distance_weight']:.3f} ({weights['distance_weight']*100:.1f}%)")
        logger.info("✅ レポート5.1.3節準拠: w_i = r_i² / Σr_i²")
        
        # グローバル設定完了の通知
        print("\n" + "="*60)
        print("🎯 動的重み計算完了 - グローバル設定適用")
        print("="*60)
        print(f"📊 グレード重み: {weights['grade_weight']:.3f} ({weights['grade_weight']*100:.1f}%)")
        print(f"📊 場所重み: {weights['venue_weight']:.3f} ({weights['venue_weight']*100:.1f}%)")
        print(f"📊 距離重み: {weights['distance_weight']:.3f} ({weights['distance_weight']*100:.1f}%)")
        print("✅ 全分析モジュールで共通使用されます")
        print("="*60 + "\n")
    
    @classmethod
    def _log_weight_calculation_results(cls, weights: Dict[str, float], correlations: Dict[str, float], train_data: pd.DataFrame):
        """
        重み計算結果をログに詳細出力
        
        Args:
            weights: 重み辞書
            correlations: 相関係数辞書
            train_data: 訓練データ
        """
        logger.info("📊 ========== 動的重み計算結果（詳細ログ） ==========")
        logger.info(f"📋 計算基準期間: {train_data['年'].min()}-{train_data['年'].max()}年")
        logger.info(f"📋 対象データ数: {len(train_data):,}行")
        logger.info(f"📋 計算式: w_i = r_i² / (r_grade² + r_venue² + r_distance²)")
        
        # 相関係数の詳細ログ
        logger.info("🔍 相関分析結果:")
        for key, corr in correlations.items():
            contribution = corr ** 2
            logger.info(f"   📊 {key}レベル: r = {corr:.4f}, r² = {contribution:.4f}")
        
        total_contribution = sum(corr ** 2 for corr in correlations.values())
        logger.info(f"   📊 総寄与度: {total_contribution:.4f}")
        
        # 重み配分の詳細ログ
        logger.info("⚖️ 算出された重み配分:")
        logger.info(f"   📊 グレード重み: {weights['grade_weight']:.4f} ({weights['grade_weight']*100:.2f}%)")
        logger.info(f"   📊 場所重み: {weights['venue_weight']:.4f} ({weights['venue_weight']*100:.2f}%)")
        logger.info(f"   📊 距離重み: {weights['distance_weight']:.4f} ({weights['distance_weight']*100:.2f}%)")
        
        # 重み合計の確認
        total_weight = weights['grade_weight'] + weights['venue_weight'] + weights['distance_weight']
        logger.info(f"   📊 重み合計: {total_weight:.4f} (1.000に正規化)")
        
        # REQI計算式のログ出力
        logger.info("📊 REQI計算式:")
        logger.info(f"   race_level = {weights['grade_weight']:.4f} × grade_level + {weights['venue_weight']:.4f} × venue_level + {weights['distance_weight']:.4f} × distance_level")
        
        logger.info("✅ 動的重み計算完了 - 全モジュールでこの重みを使用")
        logger.info("=" * 60)
    
    @classmethod
    def _get_fallback_weights(cls) -> Dict[str, float]:
        """
        フォールバック重みを取得（レポート5.1.3節準拠）
        訓練期間（2010-2020年）11,196頭の実測相関から算出された固定重み
        
        Returns:
            デフォルト重み辞書
        """
        logger.info("📊 循環論理回避版の固定重みを使用します")
        logger.info("📋 勝率ベース相関による重み（2024年改善版）:")
        logger.info("   🎯 グレード重み: 65.0% (G1-G3勝利の価値)")
        logger.info("   🏇 場所重み: 30.0% (東京・阪神の格式)")
        logger.info("   📏 距離重み: 5.0% (距離適性の補正)")
        logger.info("🔧 改善理由: 循環論理回避（予測目的≠重み算出基準）")
        
        weights = {
            'grade_weight': 0.650,   # 65.0% - 循環論理回避版
            'venue_weight': 0.300,   # 30.0% - 循環論理回避版
            'distance_weight': 0.050 # 5.0%  - 循環論理回避版
        }
        
        # フォールバック重みもログに出力
        logger.info("📊 適用重み詳細:")
        logger.info(f"   📊 グレード重み: {weights['grade_weight']:.3f} ({weights['grade_weight']*100:.1f}%)")
        logger.info(f"   📊 場所重み: {weights['venue_weight']:.3f} ({weights['venue_weight']*100:.1f}%)")
        logger.info(f"   📊 距離重み: {weights['distance_weight']:.3f} ({weights['distance_weight']*100:.1f}%)")
        
        return weights

# 便利関数
def get_global_weights() -> Dict[str, float]:
    """
    グローバル重みを取得する便利関数
    
    Returns:
        重み辞書
    """
    return WeightManager.get_weights()

def initialize_weights_from_data(df: pd.DataFrame) -> Dict[str, float]:
    """
    データから重みを初期化する便利関数
    
    Args:
        df: データセット
        
    Returns:
        重み辞書
    """
    return WeightManager.initialize_from_training_data(df)

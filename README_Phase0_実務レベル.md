# 📚 Phase 0 実務レベルデータ整備 - README

## 🎯 概要

競馬における複勝率とタイムの因果関係分析計画書の**Phase 0: データ整備**を、実務レベルの品質管理機能付きで実装しました。

**実務未経験者から実務レベルを目指す**方向けに、プロダクション環境で使われるレベルのデータ処理技術を学習できます。

---

## 📋 計画書要件との対応

| 計画書要件 | 実装内容 | 実務レベル対応 |
|-----------|----------|---------------|
| **JRA-VANデータの取得・結合** | ✅ BAC/SRB/SED統合処理 | エラーハンドリング・進捗表示 |
| **欠損値の適切な処理** | ✅ 戦略的欠損値処理（CSV作成時） | 処理方針の自動選択・ログ記録 |
| **特徴量エンジニアリング** | ✅ 137カラム包括的特徴量作成 | 品質チェック・統計検証 |
| **クリーンなデータテーブル** | ✅ 日付別分割CSV出力 | ファイルサイズ・メモリ監視 |

---

## 🚀 使用方法

### 📝 全オプションパターン一覧

#### 🔬 基本実行パターン

```bash
# 1. 基本処理のみ（BAC→SRB→SED→統合まで）
python process_race_data.py

# 2. Phase 0完全版（基本処理 + レースレベル分析）
python process_race_data.py --race-level-analysis
python process_race_data.py --レースレベル分析

# 3. レースレベル分析のみ実行（既存統合データ利用）
python process_race_data.py --race-level-analysis-only
python process_race_data.py --レースレベル分析のみ
```

#### 🌱 トラック条件指定パターン

```bash
# 4. 芝コースのみで処理
python process_race_data.py --turf-only
python process_race_data.py --芝コースのみ

# 5. 芝コースを除外（ダートのみ）
python process_race_data.py --exclude-turf
python process_race_data.py --芝コース除外

# 6. 芝コースのみ + レースレベル分析
python process_race_data.py --turf-only --race-level-analysis
python process_race_data.py --芝コースのみ --レースレベル分析

# 7. ダートのみ + レースレベル分析
python process_race_data.py --exclude-turf --race-level-analysis
python process_race_data.py --芝コース除外 --レースレベル分析

# 8. 芝コースのみ + レースレベル分析のみ
python process_race_data.py --turf-only --race-level-analysis-only
python process_race_data.py --芝コースのみ --レースレベル分析のみ

# 9. ダートのみ + レースレベル分析のみ
python process_race_data.py --exclude-turf --race-level-analysis-only
python process_race_data.py --芝コース除外 --レースレベル分析のみ
```

#### 🔧 機能制御パターン

```bash
# 10. 欠損値処理無効
python process_race_data.py --no-missing-handling
python process_race_data.py --欠損値処理無効

# 11. 品質チェック無効
python process_race_data.py --no-quality-check
python process_race_data.py --品質チェック無効

# 12. 欠損値処理 + 品質チェック両方無効
python process_race_data.py --no-missing-handling --no-quality-check
python process_race_data.py --欠損値処理無効 --品質チェック無効

# 13. レースレベル分析 + 欠損値処理無効
python process_race_data.py --race-level-analysis --no-missing-handling
python process_race_data.py --レースレベル分析 --欠損値処理無効

# 14. レースレベル分析 + 品質チェック無効
python process_race_data.py --race-level-analysis --no-quality-check
python process_race_data.py --レースレベル分析 --品質チェック無効

# 15. レースレベル分析 + 両方無効（高速処理）
python process_race_data.py --race-level-analysis --no-missing-handling --no-quality-check
python process_race_data.py --レースレベル分析 --欠損値処理無効 --品質チェック無効

# 16. レースレベル分析のみ + 欠損値処理無効
python process_race_data.py --race-level-analysis-only --no-missing-handling
python process_race_data.py --レースレベル分析のみ --欠損値処理無効

# 17. レースレベル分析のみ + 品質チェック無効
python process_race_data.py --race-level-analysis-only --no-quality-check
python process_race_data.py --レースレベル分析のみ --品質チェック無効

# 18. レースレベル分析のみ + 両方無効（最高速）
python process_race_data.py --race-level-analysis-only --no-missing-handling --no-quality-check
python process_race_data.py --レースレベル分析のみ --欠損値処理無効 --品質チェック無効
```

#### 📝 ログ出力パターン

```bash
# 19. ログレベル指定（DEBUG: 最詳細）
python process_race_data.py --race-level-analysis --log-level DEBUG
python process_race_data.py --レースレベル分析 --log-level DEBUG

# 20. ログレベル指定（WARNING: 警告のみ）
python process_race_data.py --race-level-analysis --log-level WARNING
python process_race_data.py --レースレベル分析 --log-level WARNING

# 21. ログレベル指定（ERROR: エラーのみ）
python process_race_data.py --race-level-analysis --log-level ERROR
python process_race_data.py --レースレベル分析 --log-level ERROR

# 22. ログファイル出力指定
python process_race_data.py --race-level-analysis --log-file my_analysis.log
python process_race_data.py --レースレベル分析 --log-file my_analysis.log

# 23. ログレベル + ファイル出力
python process_race_data.py --race-level-analysis --log-level DEBUG --log-file debug_analysis.log
python process_race_data.py --レースレベル分析 --log-level DEBUG --log-file debug_analysis.log
```

#### 🎯 実務パターン（組み合わせ例）

```bash
# 24. 本番運用想定（芝コース + 詳細ログ + ファイル出力）
python process_race_data.py --turf-only --race-level-analysis --log-level INFO --log-file production.log

# 25. デバッグ用（品質チェック無効 + 詳細ログ）
python process_race_data.py --race-level-analysis --no-quality-check --log-level DEBUG

# 26. 高速テスト用（レースレベル分析のみ + 全機能無効）
python process_race_data.py --race-level-analysis-only --no-missing-handling --no-quality-check --log-level WARNING

# 27. メモリ制約環境用（芝コースのみ + 品質チェック無効）
python process_race_data.py --turf-only --race-level-analysis --no-quality-check

# 28. 完全版詳細ログ（全機能有効 + 最詳細ログ + ファイル出力）
python process_race_data.py --race-level-analysis --log-level DEBUG --log-file complete_analysis.log

# 29. データ更新用（レースレベル分析のみ + 警告レベル）
python process_race_data.py --race-level-analysis-only --log-level WARNING

# 30. トラブルシューティング用（基本処理のみ + 詳細ログ）
python process_race_data.py --log-level DEBUG --log-file troubleshoot.log
```

#### 💡 用途別推奨パターン

| 用途 | 推奨コマンド | 理由 |
|------|-------------|------|
| **初回実行** | `python process_race_data.py --レースレベル分析` | 全機能を体験できる |
| **日常更新** | `python process_race_data.py --レースレベル分析のみ` | 高速で最新分析データ作成 |
| **芝専用分析** | `python process_race_data.py --芝コースのみ --レースレベル分析のみ` | 芝レース特化の高速分析 |
| **デバッグ** | `python process_race_data.py --レースレベル分析 --log-level DEBUG` | 問題特定に必要な詳細情報 |
| **本番運用** | `python process_race_data.py --レースレベル分析 --log-file prod.log` | ログ永続化で運用監視 |
| **高速テスト** | `python process_race_data.py --レースレベル分析のみ --no-quality-check` | 開発時の迅速な動作確認 |

### 基本実行（実務レベル全機能有効）

```bash
# 計画書Phase 0の完全実装
python process_race_data.py --レースレベル分析

# 英語版
python process_race_data.py --race-level-analysis
```

### 高度なオプション

```bash
# 芝コースのみで実務レベル処理
python process_race_data.py --turf-only --レースレベル分析

# 欠損値処理を無効化（デバッグ用）
python process_race_data.py --レースレベル分析 --no-missing-handling

# 品質チェックを無効化（高速処理用）
python process_race_data.py --レースレベル分析 --no-quality-check

# ログレベル調整 + ファイル出力
python process_race_data.py --レースレベル分析 --log-level DEBUG --log-file my_analysis.log
```

---

## 🔧 実務レベル機能の詳細

### 1. 戦略的欠損値処理（CSV作成時）

#### 処理戦略

| 列タイプ | 処理方法 | 実務的判断基準 |
|----------|----------|---------------|
| **重要列**（着順・タイム・距離・馬名・IDM） | 行削除 | 分析に必須の項目 |
| **数値列** | 中央値補完 | 外れ値の影響を受けにくい |
| **カテゴリ列** | 最頻値補完 | ビジネス的に最も一般的な値 |
| **高欠損列** | 列削除 | 50%以上欠損は信頼性低 |

#### 欠損値処理ログの確認

```bash
# 処理ログファイルの確認
cat export/missing_value_processing_log.txt
```

**ログ例**:
```
欠損値処理ログ - 2024-01-15 14:30:25
==================================================

• 着順: 45行を削除（重要列）
• IDM: 234行を削除（重要列）
• 調教師名: 安田隆行で12件補完
• 残存欠損値: 8行削除

最終データ形状: (15234, 137)
残存欠損値: 0件
```

### 2. データ品質チェック機能

#### 自動チェック項目

- **📊 基本統計**: 行数・列数・メモリ使用量
- **❓ 欠損値分析**: 列別欠損率・パターン分析
- **🔄 重複検出**: 完全重複行の特定
- **📈 外れ値検出**: IQR法による数値列の異常値
- **📋 ビジネスルール**: 競馬特有の妥当性チェック

#### 品質レポートの確認

```bash
# JSON形式の詳細レポート
cat export/quality_reports/data_quality_report.json
```

**レポート例**:
```json
{
  "特徴量エンジニアリング後_SED230326": {
    "total_rows": 479,
    "total_columns": 137,
    "memory_usage_mb": 12.5,
    "missing_values": {
      "total_missing_cells": 23,
      "columns_with_missing": {
        "調教師名": 12,
        "IDM": 11
      }
    },
    "warnings": [],
    "recommendations": ["データ品質は良好です"]
  }
}
```

### 3. システムリソース監視

実行中に以下の情報がリアルタイムで表示されます：

```
💻 [BAC処理完了] システム状態:
   ⏱️ 経過時間: 45.2秒
   🧠 メモリ使用量: 2.3GB (差分: +0.5GB)
   🔥 CPU使用率: 85.4%
```

### 4. エラーハンドリングと復旧

- **📁 ファイル不存在**: 適切なエラーメッセージと解決策提示
- **💾 メモリ不足**: 処理の分割・サンプリング提案
- **🔧 データ破損**: ファイル単位での除外・継続処理
- **⚠️ 警告レベル**: 処理は継続するが要注意事項をログ出力

---

## 📊 出力データの構造

### ファイル構成

```
export/race_level_analysis/
├── SED230326_race_level_analysis.csv    # 日付別分割データ
├── SED230327_race_level_analysis.csv
├── ...
├── feature_summary.json                 # 全体統計サマリー
└── ...

export/quality_reports/
├── data_quality_report.json             # 品質チェック結果
└── ...

export/logs/
├── process_race_data_20240115_143025.log  # 実行ログ
└── ...

export/
├── missing_value_processing_log.txt     # 欠損値処理詳細
└── ...
```

### CSV各ファイルの特徴量（137カラム）

#### 基本情報（JRA-VAN由来）
- `場コード`, `年`, `回`, `日`, `R`, `馬番`, `馬名`, `距離`, `芝ダ障害コード`, `馬場状態`, `着順`, `タイム`, `騎手名` 等

#### Phase 0で追加される分析用特徴量

| カテゴリ | 特徴量例 | 説明 |
|----------|----------|------|
| **レースレベル** | `race_level` | G1=9.8, 未勝利=1.2など0-10スケール |
| **馬能力** | `horse_ability_observed`, `horse_ability_corrected` | IDM等統合指標、バイアス補正版 |
| **トラックバイアス** | `track_bias_total`, `frame_bias`, `running_style_bias` | 脚質・枠順・馬場の有利不利数値化 |
| **走破タイム** | `time_zscore`, `distance_adjusted_time`, `speed_index` | 距離補正・正規化・速度指標 |
| **複勝フラグ** | `is_win`, `is_placed` | 1着フラグ、3着以内フラグ |
| **その他要因** | `jockey_*`, `dist_cat_*`, `weight_burden` | 騎手・距離・斤量ダミー変数 |

---

## 💡 実務レベルで学べる技術

### 1. データエンジニアリング

#### ログ設計
```python
# 実務レベルのログ設定例
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
```
**学習ポイント**: 本番環境では「いつ・誰が・どこで・何を」が特定できるログが必要

#### メモリ監視
```python
current_memory = psutil.virtual_memory().used / 1024 / 1024 / 1024  # GB
```
**学習ポイント**: 大量データ処理では、メモリ使用量の監視が障害予防に重要

### 2. 品質管理手法

#### 欠損値処理の戦略設計
```python
strategy_config = {
    'critical_columns': {'着順': 'drop', 'IDM': 'drop'},  # ビジネス判断
    'numeric_columns': {'method': 'median'},              # 統計的判断
    'categorical_columns': {'method': 'mode'},            # ドメイン判断
}
```
**学習ポイント**: 一律の処理ではなく、データの性質・ビジネス要件に応じた戦略選択

#### 品質指標の定義
```python
missing_rate = missing_count / len(df)
if missing_rate > 0.5:  # 50%以上欠損
    action = "列削除"
```
**学習ポイント**: 実務では「何%以上で危険」といった具体的な閾値設定が必要

### 3. エラーハンドリング

#### 例外処理の階層化
```python
try:
    # メイン処理
    result = process_data()
except SpecificError as e:
    logger.warning(f"特定エラー対応: {e}")
    # 代替処理
except Exception as e:
    logger.error(f"予期せぬエラー: {e}")
    # 安全な終了
```
**学習ポイント**: エラーレベルに応じた適切な対応（続行・代替・停止）

#### 復旧可能な設計
```python
for file in files:
    try:
        process_file(file)
    except Exception as e:
        logger.error(f"ファイル{file}の処理失敗: {e}")
        continue  # 他のファイルは処理継続
```
**学習ポイント**: 一部の失敗が全体を止めない、回復力のある設計

---

## 🎓 学習効果の確認

### 実行結果で確認できる実務レベル要素

1. **📊 処理量の把握**
   ```
   📊 読み込んだデータ件数: 528,088件
   💾 メモリ使用量: 580.5MB
   ⏱️ 総処理時間: 187.3秒 (3.1分)
   ```

2. **🔧 品質管理の実行**
   ```
   📊 【特徴量エンジニアリング後】品質サマリー:
      📏 データ規模: 479行 x 137列
      ❓ 欠損セル数: 1,247
      ⚠️ 警告: 0件
   ```

3. **🛠️ 問題解決の記録**
   ```
   🔧 欠損値処理実行: SED230326_race_level_analysis.csv
   📊 欠損値処理結果: 479行 → 456行 (除去: 23行)
   ```

### 理解度チェック

- [ ] ログメッセージから処理状況を把握できる
- [ ] 品質レポートから問題箇所を特定できる
- [ ] エラーが発生した際の対処法を理解している
- [ ] 出力データの各特徴量の意味を説明できる
- [ ] 欠損値処理の戦略判断根拠を理解している

---

## 🔗 次のステップ（Phase 1準備）

### 生成されたデータでの基礎分析

```bash
# Phase 1: 基礎分析の実行
python analyze_race_level.py export/race_level_analysis

# 時系列分析
python analyze_race_level.py export/race_level_analysis --three-year-periods

# タイム因果関係分析
python analyze_race_level.py export/race_level_analysis --enable-time-analysis
```

### データ品質の継続監視

- 月次での品質レポート確認
- 欠損率の推移モニタリング
- 処理時間・メモリ使用量のベンチマーク

---

## ⚠️ トラブルシューティング

### よくある問題と解決法

#### 1. メモリ不足エラー

**現象**:
```
MemoryError: Unable to allocate 2.3 GiB for an array
```

**解決法**:
```bash
# 芝コースのみで処理量を削減
python process_race_data.py --turf-only --レースレベル分析

# または、一部期間のみ処理するよう input_dir を調整
```

#### 2. 日本語エンコーディングエラー

**現象**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**解決法**:
```python
# race_level_processor.py で encoding 指定を確認
pd.read_csv(file_path, encoding='utf-8')
# または encoding='shift_jis' を試行
```

#### 3. ディスク容量不足

**現象**:
```
OSError: [Errno 28] No space left on device
```

**解決法**:
- 既存の export/ フォルダ内の不要ファイル削除
- より容量の大きいドライブへの出力先変更

---

## 📞 サポート情報

### ログファイルの確認

```bash
# 最新の実行ログ確認
ls -la export/logs/
tail -f export/logs/process_race_data_[最新タイムスタンプ].log
```

### デバッグモードでの実行

```bash
# より詳細なログ出力
python process_race_data.py --レースレベル分析 --log-level DEBUG
```

### 性能測定

```bash
# 処理時間測定
time python process_race_data.py --レースレベル分析
```

---

## 🎉 実務レベル達成の証明

この実装を理解し実行できれば、以下の実務レベル技術を習得したことになります：

✅ **データパイプライン設計**: 段階的処理・エラーハンドリング  
✅ **品質管理**: 自動チェック・閾値設定・レポート生成  
✅ **システム監視**: リソース監視・パフォーマンス測定  
✅ **ログ設計**: 問題特定・デバッグ効率化  
✅ **例外処理**: 適切な分類・継続可能な設計  
✅ **ドキュメント**: 運用・保守を考慮した記述  

---

**🎓 実務未経験から実務レベルへのステップアップが完了です！**

これらの技術は、データサイエンス・エンジニアリングの実際の現場で日常的に使われているものです。このスキルセットを身につけることで、プロダクション環境でも通用するデータ処理システムの構築・運用が可能になります。

---

**最終更新**: 2024年1月15日  
**対象レベル**: 実務未経験者 → 実務レベル  
**学習時間**: 2-4時間（理解＋実行） 
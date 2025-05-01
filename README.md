# 競馬データ分析ツール

競馬のレースデータを分析し、レースレベルや成績の傾向を可視化するツールです。

## 機能

- レースレベルの分析
  - グレードベースの評価
  - 賞金額による補正
  - 距離による重み付け
  - 2000m特別評価
  - グレードと距離の相互作用考慮
- 基本統計情報の算出
- グレード別分析
- 相関分析
- 回帰分析
- データの可視化
- JRAデータの変換処理
  - BAC（レース基本情報）データの変換
  - SED（競走成績）データの変換

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/horse-racing-analysis.git
cd horse-racing-analysis

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 実行順序

データ分析を行うには、以下の順序で処理を実行する必要があります：

1. BACデータの変換（レース基本情報の処理）
2. SEDデータの変換（競走成績データの処理）
3. レース分析の実行

この順序で実行することで、正確な分析結果が得られます。

### 1. BACデータの変換
```bash
# 単一のBACファイルを処理
python process_bac_data.py "import/BAC/BAC20240101.txt"

# importフォルダ内の全BACファイルを処理
python process_bac_data.py
```

### 2. SEDデータの変換
```bash
# 単一のSEDファイルを処理
python process_sed_data.py "import/SED/SED20240101.txt"

# importフォルダ内の全SEDファイルを処理
python process_sed_data.py
```

### 3. レース分析の実行

```bash
python analyze_race_level.py "入力ファイルパス" --output-dir "出力ディレクトリ" --min-races 6
```

### オプション

- `input_path`: 入力ファイルまたはディレクトリのパス（必須）
- `--output-dir`: 分析結果の出力ディレクトリ（デフォルト: export/analysis）
- `--min-races`: 分析対象とする最小レース数（デフォルト: 6）
- `--encoding`: 入力ファイルのエンコーディング（デフォルト: utf-8）

### 使用例

```bash
# 1. BACデータの変換
python process_bac_data.py "import/BAC"

# 2. SEDデータの変換
python process_sed_data.py

# 3. 変換済みのSEDデータを分析
python analyze_race_level.py "export/SED/SED20240101_formatted.csv" --output-dir "results/20240101"

# または、ディレクトリ内の全SEDファイルを分析
python analyze_race_level.py "export/SED" --output-dir "results/all" --min-races 6
```

## レースレベル分析の詳細

### レベル計算の要素

1. グレード評価（60%）
   - G1: 基本レベル9、勝利時+10.0
   - G2: 基本レベル8、勝利時+8.0
   - G3: 基本レベル7、勝利時+7.0
   - 重賞: 基本レベル6、勝利時+6.0
   - L: 基本レベル5.5、勝利時+5.5
   - 特別: 基本レベル5、勝利時+5.0

2. 賞金評価（40%）
   - G1相当: 10,000万円以上
   - G2相当: 7,000万円以上
   - G3相当: 4,500万円以上
   - 重賞相当: 3,500万円以上
   - L相当: 2,000万円以上

3. 距離補正
   - スプリント（-1400m）: 0.85倍
   - マイル（1401-1800m）: 1.00倍
   - 中距離（1801-2000m）: 1.35倍
   - 中長距離（2001-2400m）: 1.45倍
   - 長距離（2401m-）: 1.25倍

4. 特別ボーナス
   - 2000m前後（1900-2100m）: 1.35倍
   - 高グレード（G1-G3）×最適距離（1800-2400m）: 1.15倍

### 分析結果の解釈

1. レースレベル（0-10スケール）
   - 8-10: 最高峰レース（G1-G2レベル）
   - 6-8: 高レベル（G3-重賞レベル）
   - 4-6: 中レベル（L-特別レベル）
   - 0-4: 一般レベル

2. 距離評価
   - 中距離（1800-2000m）と中長距離（2001-2400m）を重視
   - 2000m前後のレースに特別ボーナス
   - 短距離と長距離は相対的に低評価

## 出力ファイル

### データ変換結果

1. BACデータ変換結果
   - 出力先: `export/BAC/`
   - 形式: CSV（UTF-8 with BOM）

2. SEDデータ変換結果
   - 出力先: `export/SED/`
   - レース結果データ: `{元ファイル名}_formatted.csv`
   - 馬ごとの成績集計: `{元ファイル名}_horse_stats.csv`

### 分析結果

1. `grade_win_rate.png`: グレード別の勝率・複勝率グラフ
2. `race_level_correlation.png`: レースレベルと成績の相関分析
3. `race_level_distribution.png`: レースレベルの分布
4. `race_level_trends.png`: レースレベルの推移
5. `distance_level_heatmap.png`: 距離とレースレベルの関係

## ディレクトリ構成

```
horse_racing/
├── __init__.py
├── analyzers/          # 分析モジュール
├── base/              # 基底クラス
├── data/              # データ操作
└── visualization/     # 可視化モジュール

import/                # 入力データ配置ディレクトリ
├── BAC/              # BACファイル配置場所
└── SED/              # SEDファイル配置場所

export/               # 変換済みデータ出力ディレクトリ
├── BAC/              # BAC変換結果
├── SED/              # SED変換結果
└── analysis/         # 分析結果
```

## 開発者向け情報

### 環境構築

```bash
# 開発環境のセットアップ
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### テスト実行

```bash
# 全テストの実行
pytest tests/

# 特定のテストの実行
pytest horse_racing/tests/analyzers/test_race_level_analyzer.py -v
```

### コードの品質管理

- テストカバレッジの維持
- PEP 8スタイルガイドの遵守
- 型ヒントの活用
- ドキュメンテーションの充実

## ライセンス

MIT License

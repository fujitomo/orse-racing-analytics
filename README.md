# 競馬データ分析プロジェクト

## 概要

このプロジェクトは、競馬のレースデータを分析し、競走馬のパフォーマンスに影響を与える要因を探るためのツールです。独自の合成特徴量 `HorseREQI` を算出し、それが競走馬の複勝率（3着以内に入る確率）とどの程度相関があるかを統計的に分析・可視化します。

分析の背景、仮説、詳細な結果については、[`race_level_analysis_report.md`](race_level_analysis_report.md) を参照してください。

## 主な機能

*   **データ処理**: JRA-VANの生データ（SED, SRB, BAC形式）を読み込み、クリーニング、欠損値補完、特徴量エンジニアリングを行い、分析用のデータセットを生成します。
*   **特徴量生成**: レースの格（グレード）、開催場所、距離を統合した独自の指標 `RacePoint` および、馬ごとの実績を示す `HorseREQI`（平均・最大）を算出します。
*   **相関分析**: `HorseREQI` と複勝率との間の相関関係を分析し、その有効性を評価します。
*   **時系列分析**: 3年ごとの期間別分析を実行し、分析結果の時間的な安定性（再現性）を検証します。
*   **可視化**: 分析結果を箱ひげ図や散布図として出力し、直感的な理解をサポートします。

## ディレクトリ構成

```
.
├── horse_racing/      # Pythonパッケージのソースコード
│   ├── analyzers/     # 分析ロジック
│   ├── data/          # データ処理ロジック
│   └── visualization/ # 可視化ロジック
├── import/            # JRA-VANからダウンロードした生データを配置するディレクトリ
├── export/            # 処理済みデータやログが出力されるディレクトリ
│   └── dataset/       # 分析に使用する最終的なデータセット
├── results/           # 分析結果（画像ファイルなど）の出力先
├── process_race_data.py         # データ処理を実行するスクリプト
├── analyze_horse_REQI.py   # 競走経験質指数（REQI）分析を実行するスクリプト
├── requirements.txt             # 本番環境用の依存ライブラリ
├── setup.py                     # パッケージ設定ファイル
└── race_level_analysis_report.md # 分析の詳細レポート
```

## セットアップ

1.  リポジトリをクローンします。
    ```bash
    git clone https://github.com/yourusername/horse-racing-analysis.git
    cd horse-racing-analysis
    ```

2.  Pythonの仮想環境を作成し、有効化します。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windowsの場合は venv\Scripts\activate
    ```

3.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## 使い方

### ステップ1: データ処理

1.  `import/` ディレクトリ配下に、JRA-VANからダウンロードした生データ（`.txt`ファイル）を配置します。
2.  以下のコマンドを実行して、データの前処理と分析用データセットの生成を行います。

    ```bash
    python process_race_data.py
    ```

    処理が完了すると、`export/dataset/` ディレクトリにCSVファイルが出力されます。

### ステップ2: 競走経験質指数（REQI）分析

以下のコマンドを実行して、競走経験質指数（REQI）の分析と結果の可視化を行います。

```bash
# 全期間を対象に基本的な分析を実行
python analyze_horse_REQI.py export/dataset --output-dir results/race_level_analysis

# 3年ごとの時系列分析を実行
python analyze_horse_REQI.py export/dataset --output-dir results/race_level_analysis --three-year-periods
```

分析結果のレポートや画像は `results/race_level_analysis` ディレクトリに出力されます。

## ライセンス

このプロジェクトは MIT License のもとで公開されています。
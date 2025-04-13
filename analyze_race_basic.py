import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import argparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
plt.rcParams['font.size'] = 12  # フォントサイズ

# コース種別の定義
track_type_mapping = {
    1: "芝",
    2: "ダート",
    3: "障害"
}

def load_and_process_data(input_path):
    """
    CSVファイルを読み込んで処理します。
    input_path:  単一のCSVファイルパス、またはCSVファイルを含むディレクトリパス
    """
    input_path = Path(input_path)
    sec_dfs = []
    total_files = 0
    processed_files = 0

    if input_path.is_file():
        # 単一のCSVファイルの場合
        if input_path.suffix.lower() == '.csv':
            print(f"ファイルを読み込み中: {input_path}")
            df = pd.read_csv(input_path, encoding="utf-8")
            # 芝ダ障害コードを文字列に変換
            df["芝ダ障害コード"] = df["芝ダ障害コード"].map(track_type_mapping)
            # 芝レースのみをフィルタリング
            df = df[df["芝ダ障害コード"] == "芝"]
            sec_dfs.append(df)
            total_files = 1
            processed_files = 1
    elif input_path.is_dir():
        # ディレクトリ内のすべてのCSVファイルを検索
        csv_files = list(input_path.rglob("*_formatted.csv"))
        total_files = len(csv_files)
        print(f"\n検出されたCSVファイル数: {total_files}")
        
        if total_files == 0:
            raise ValueError(f"指定されたディレクトリ '{input_path}' に_formatted.csvファイルが見つかりません。")
        
        # 各ファイルを処理
        for file in csv_files:
            try:
                print(f"\n処理中 ({processed_files + 1}/{total_files}): {file.name}")
                df = pd.read_csv(file, encoding="utf-8")
                # 芝ダ障害コードを文字列に変換
                df["芝ダ障害コード"] = df["芝ダ障害コード"].map(track_type_mapping)
                # 芝レースのみをフィルタリング
                df = df[df["芝ダ障害コード"] == "芝"]
                print(f"読み込んだレコード数: {len(df)}")
                
                # データの基本チェック
                if len(df) == 0:
                    print(f"警告: {file.name} は空のファイルです")
                    continue
                
                required_columns = [
                    "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
                    "レース名", "種別", "芝ダ障害コード", "馬番", "騎手名", "調教師名"
                ]
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: {file.name} に必要な列が不足しています: {missing_cols}")
                    continue
                
                sec_dfs.append(df)
                processed_files += 1
                
            except Exception as e:
                print(f"エラー - ファイル {file.name} の処理中: {str(e)}")
                continue
    
    if not sec_dfs:
        raise ValueError("有効なデータを含むCSVファイルが見つかりませんでした。")
    
    print("\n処理概要:")
    print(f"- 検出ファイル数: {total_files}")
    print(f"- 正常処理ファイル数: {processed_files}")
    
    # すべてのデータフレームを結合
    sec_df = pd.concat(sec_dfs, ignore_index=True)
    print(f"- 総レコード数: {len(sec_df)}")
    
    # 必要な列の抽出
    required_columns = [
        "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
        "レース名", "種別", "芝ダ障害コード", "馬番", "騎手名", "調教師名"
    ]
    sec_df = sec_df[required_columns]
    
    # データ型の変換
    print("\nデータ型を変換中...")
    sec_df["距離"] = pd.to_numeric(sec_df["距離"], errors="coerce")
    sec_df["着順"] = pd.to_numeric(sec_df["着順"], errors="coerce")
    sec_df["種別"] = pd.to_numeric(sec_df["種別"], errors="coerce")
    sec_df["レース名"] = sec_df["レース名"].astype(str)
    sec_df["馬名"] = sec_df["馬名"].astype(str)
    
    # 着順の欠損値を持つレコードを除外
    before_count = len(sec_df)
    sec_df = sec_df.dropna(subset=["着順"])
    after_count = len(sec_df)
    if before_count > after_count:
        print(f"\n着順の欠損値を持つ {before_count - after_count} レコードを除外しました")
        print(f"- 除外前: {before_count} レコード")
        print(f"- 除外後: {after_count} レコード")
    
    # その他の欠損値の確認
    null_counts = sec_df.isnull().sum()
    if null_counts.any():
        print("\n警告: 以下の列に欠損値が存在します:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"- {col}: {count}件")
    
    return sec_df

def calculate_race_level(df):
    """
    レースレベルの計算を距離と着順のみで行います。
    - 距離帯による重み付け
    - 着順による評価
    """
    # 基本設定
    df["race_level"] = 0
    df["is_win"] = df["着順"] == 1
    df["is_placed"] = df["着順"] <= 3

    # 1. 距離帯による重み付け
    distance_weights = {
        (1150, 1250): 2.0,   # 1200m付近
        (1550, 1650): 3.0,   # 1600m付近
        (1900, 2100): 4.0,   # 2000m付近（最重要）
        (2350, 2450): 3.0,   # 2400m付近
        (2500, 9999): 2.0,   # 2500m以上
    }
    
    # ベースレベルの設定
    df["race_level"] = 1.0
    
    # 重要距離帯の重み付け
    for (min_dist, max_dist), weight in distance_weights.items():
        mask = (df["距離"] >= min_dist) & (df["距離"] <= max_dist)
        df.loc[mask, "race_level"] = weight

    # 2. 着順による評価
    df.loc[df["着順"] <= 3, "race_level"] *= 1.2  # 複勝圏内
    df.loc[df["着順"] == 1, "race_level"] *= 1.3  # 優勝

    # 3. レースレベルの正規化（0-10のスケールに）
    df["race_level"] = (df["race_level"] - df["race_level"].min()) / (df["race_level"].max() - df["race_level"].min()) * 10

    return df

def analyze_win_rates(df):
    """
    勝率分析を行います。
    """
    # 馬ごとの基本統計
    horse_stats = df.groupby("馬名").agg({
        "race_level": ["max", "mean"],
        "is_win": "sum",
        "is_placed": "sum",
        "着順": "count"
    }).reset_index()

    # カラム名の整理
    horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数"]
    
    # 勝率と複勝率の計算
    horse_stats["win_rate"] = horse_stats["勝利数"] / horse_stats["出走回数"]
    horse_stats["place_rate"] = horse_stats["複勝数"] / horse_stats["出走回数"]
    
    # 距離帯別の統計
    distance_stats = df.groupby(pd.cut(df["距離"], 
                                     bins=[0, 1400, 1800, 2000, 2400, 9999],
                                     labels=["スプリント", "マイル", "中距離", "中長距離", "長距離"])).agg({
        "is_win": ["mean", "count"],
        "is_placed": "mean",
        "race_level": ["mean", "std"]
    }).reset_index()
    
    distance_stats.columns = ["距離帯", "勝率", "レース数", "複勝率", "平均レベル", "レベル標準偏差"]
    
    return horse_stats, distance_stats

def visualize_results(horse_stats, distance_stats, output_dir):
    """
    結果の可視化を行います。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # フォント設定
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

    # 1. 距離帯別の勝率・複勝率
    plt.figure(figsize=(15, 8))
    x = np.arange(len(distance_stats))
    width = 0.35

    plt.bar(x - width/2, distance_stats["勝率"], width, 
            label="単勝率", color="skyblue")
    plt.bar(x + width/2, distance_stats["複勝率"], width, 
            label="複勝率", color="lightcoral")

    plt.title("距離帯別 勝率・複勝率")
    plt.xlabel("距離帯")
    plt.ylabel("確率")
    plt.xticks(x, distance_stats["距離帯"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "distance_win_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. レースレベルと勝率の散布図（サイズを出走回数で変化）
    plt.figure(figsize=(15, 8))
    scatter = plt.scatter(horse_stats["最高レベル"], horse_stats["win_rate"], 
                         s=horse_stats["出走回数"]*20, alpha=0.5,
                         c=horse_stats["平均レベル"], cmap='viridis')
    plt.title("レースレベルと勝率の関係\n（円の大きさは出走回数、色は平均レベルを表す）")
    plt.xlabel("最高レースレベル")
    plt.ylabel("勝率")
    plt.colorbar(scatter, label="平均レベル")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "race_level_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 距離帯別の平均レースレベル
    plt.figure(figsize=(15, 8))
    plt.bar(x, distance_stats["平均レベル"], yerr=distance_stats["レベル標準偏差"],
            color="lightgreen", alpha=0.7)
    plt.title("距離帯別 平均レースレベル")
    plt.xlabel("距離帯")
    plt.ylabel("平均レースレベル")
    plt.xticks(x, distance_stats["距離帯"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "distance_level_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def perform_regression_analysis(df):
    """
    距離と着順のみを使用した重回帰分析を実行します。
    """
    # 説明変数の準備
    X = df[["距離"]].copy()
    y = df["着順"]

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの学習（Ridge回帰を使用）
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 予測と評価
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 係数の取得
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)

    return coef_df, r2, rmse

def perform_multivariate_analysis(df):
    """
    距離と着順のみの多変量解析を実行します。
    """
    # 相関分析用のデータ準備
    numeric_cols = ['距離', '着順']
    correlation_data = df[numeric_cols].copy()
    
    # 相関行列の計算
    correlation_matrix = correlation_data.corr()
    
    # 基本統計量の計算
    descriptive_stats = correlation_data.describe()
    
    return correlation_matrix, descriptive_stats

def visualize_multivariate_results(correlation_matrix, coef_df, r2, rmse, output_dir):
    """
    多変量解析の結果を可視化します。
    """
    # 相関行列のヒートマップ
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('変数間の相関行列')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 回帰係数のプロット
    plt.figure(figsize=(12, 6))
    sns.barplot(data=coef_df.head(10), x='coefficient', y='feature')
    plt.title(f'重回帰分析の係数（上位10項目）\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
    plt.xlabel('係数')
    plt.ylabel('特徴量')
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='芝レースのレースレベルと勝率の分析を行います。')
    parser.add_argument('input_path', help='入力CSVファイルのパス、またはCSVファイルを含むディレクトリのパス')
    parser.add_argument('--output-dir', '-o', default='export/analysis',
                      help='出力ディレクトリのパス（デフォルト: export/analysis）')
    args = parser.parse_args()

    # データの読み込みと処理
    df = load_and_process_data(args.input_path)
    
    # 入力ファイルから日付情報を取得
    input_path = Path(args.input_path)
    if input_path.is_file():
        date_str = input_path.stem.split('_')[0][3:]  # SECyymmdd から yymmdd を抽出
    else:
        # ディレクトリの場合は現在の日付を使用
        from datetime import datetime
        date_str = datetime.now().strftime('%y%m%d')
    
    # 出力ディレクトリの設定
    output_base = Path(args.output_dir)
    output_dir = output_base / f"analysis_turf_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # レースレベルの計算
    df = calculate_race_level(df)
    
    # 勝率の分析
    horse_stats, distance_stats = analyze_win_rates(df)
    
    # 重回帰分析の実行
    coef_df, r2, rmse = perform_regression_analysis(df)
    
    # 多変量解析の実行
    correlation_matrix, descriptive_stats = perform_multivariate_analysis(df)
    
    # 結果の可視化
    visualize_results(horse_stats, distance_stats, output_dir)
    visualize_multivariate_results(correlation_matrix, coef_df, r2, rmse, output_dir)
    
    # 結果の保存
    horse_stats.to_csv(output_dir / f"horse_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    distance_stats.to_csv(output_dir / f"distance_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    coef_df.to_csv(output_dir / f"regression_coefficients_turf_{date_str}.csv", index=False, encoding="utf-8")
    correlation_matrix.to_csv(output_dir / f"correlation_matrix_turf_{date_str}.csv", encoding="utf-8")
    descriptive_stats.to_csv(output_dir / f"descriptive_stats_turf_{date_str}.csv", encoding="utf-8")
    
    print(f"\n芝レースの分析が完了しました。結果は {output_dir} に保存されています：")
    print(f"- {output_dir}/horse_stats_turf_{date_str}.csv: 馬ごとの統計")
    print(f"- {output_dir}/distance_stats_turf_{date_str}.csv: 距離帯ごとの統計")
    print(f"- {output_dir}/regression_coefficients_turf_{date_str}.csv: 重回帰分析の係数")
    print(f"- {output_dir}/correlation_matrix_turf_{date_str}.csv: 相関行列")
    print(f"- {output_dir}/descriptive_stats_turf_{date_str}.csv: 基本統計量")
    print(f"- {output_dir}/distance_win_rate.png: 距離帯別勝率")
    print(f"- {output_dir}/race_level_analysis.png: レースレベルと勝率の散布図")
    print(f"- {output_dir}/distance_level_analysis.png: 距離帯別平均レースレベル")
    print(f"- {output_dir}/correlation_matrix.png: 相関行列のヒートマップ")
    print(f"- {output_dir}/regression_coefficients.png: 重回帰分析の係数プロット")
    print("\n重回帰分析の結果：")
    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main() 
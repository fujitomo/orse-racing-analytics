import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import argparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
plt.rcParams['font.size'] = 12  # フォントサイズ

# グレード定義
GRADE_LEVELS = {
    1: {"name": "G1", "weight": 10.0, "base_level": 9},
    2: {"name": "G2", "weight": 8.0, "base_level": 8},
    3: {"name": "G3", "weight": 7.0, "base_level": 7},
    4: {"name": "重賞", "weight": 6.0, "base_level": 6},
    5: {"name": "特別", "weight": 5.0, "base_level": 5},
    6: {"name": "L", "weight": 5.5, "base_level": 5.5}  # リステッド競走
}

# コース種別の定義
track_type_mapping = {
    "芝": "芝",
    "ダート": "ダート",
    "障害": "障害"
}

# 競走種別の定義
RACE_TYPE_CODES = {
    11: "２歳",
    12: "３歳",
    13: "３歳以上",
    14: "４歳以上",
    20: "障害",
    99: "その他"
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

    # 種別からグレードを判定する関数（グレード列が存在しない場合のバックアップ）
    def determine_grade(row):
        """
        レース名と種別コードからグレードを判定する
        """
        race_name = str(row["レース名"]) if pd.notna(row["レース名"]) else ""
        race_type = row["種別"] if pd.notna(row["種別"]) else 99

        # レース名からグレードを判定
        if "G1" in race_name or "Ｇ１" in race_name:
            return 1
        elif "G2" in race_name or "Ｇ２" in race_name:
            return 2
        elif "G3" in race_name or "Ｇ３" in race_name:
            return 3
        elif "重賞" in race_name:
            return 4
        elif "L" in race_name or "Ｌ" in race_name:
            return 6
        
        # 種別コードに基づく判定
        if race_type in [11, 12]:  # 2歳・3歳戦
            return 5  # 特別
        elif race_type in [13, 14]:  # 3歳以上・4歳以上
            if "特別" in race_name:
                return 5
            else:
                return 5  # デフォルトは特別扱い
        elif race_type == 20:  # 障害
            if "J.G1" in race_name:
                return 1
            elif "J.G2" in race_name:
                return 2
            elif "J.G3" in race_name:
                return 3
            else:
                return 5
        else:  # その他
            return 5

    if input_path.is_file():
        # 単一のCSVファイルの場合
        if input_path.suffix.lower() == '.csv':
            print(f"ファイルを読み込み中: {input_path}")
            try:
                # エンコーディングを指定して読み込み
                df = pd.read_csv(input_path, encoding="utf-8-sig")
                print(f"読み込んだレコード数: {len(df)}")
                # データの基本チェック
                if len(df) == 0:
                    print(f"警告: {input_path.name} は空のファイルです")
                else:
                    # デバッグ情報の表示
                    print(f"芝ダ障害コードのユニークな値: {df['芝ダ障害コード'].unique()}")
                    print(f"芝ダ障害コードの型: {df['芝ダ障害コード'].dtype}")
                    
                    # マッピング前の値の確認
                    print(f"マッピング前の芝ダ障害コードの分布:\n{df['芝ダ障害コード'].value_counts()}")
                    
                    # 芝レースのみをフィルタリング
                    df = df[df["芝ダ障害コード"] == "芝"]
                    print(f"芝レースの数: {len(df)}")
                    
                    if len(df) > 0:
                        sec_dfs.append(df)
                        processed_files += 1
            except Exception as e:
                print(f"エラー - ファイル {input_path.name} の処理中: {str(e)}")
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
                # エンコーディングを指定して読み込み
                df = pd.read_csv(file, encoding="utf-8-sig")
                print(f"読み込んだレコード数: {len(df)}")
                
                # データの基本チェック
                if len(df) == 0:
                    print(f"警告: {file.name} は空のファイルです")
                    continue
                
                # デバッグ情報の表示
                print(f"芝ダ障害コードのユニークな値: {df['芝ダ障害コード'].unique()}")
                print(f"芝ダ障害コードの型: {df['芝ダ障害コード'].dtype}")
                
                # マッピング前の値の確認
                print(f"マッピング前の芝ダ障害コードの分布:\n{df['芝ダ障害コード'].value_counts()}")
                
                # 芝レースのみをフィルタリング
                df = df[df["芝ダ障害コード"] == "芝"]
                print(f"芝レースの数: {len(df)}")
                
                if len(df) > 0:
                    required_columns = [
                        "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
                        "レース名", "種別", "芝ダ障害コード", "馬番", "騎手名", "調教師名", "グレード"
                    ]
                    
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        print(f"警告: {file.name} に必要な列が不足しています: {missing_cols}")
                        if "グレード" in missing_cols:
                            print("グレード列が存在しないため、レース名と種別から判定します")
                            df["グレード"] = df.apply(determine_grade, axis=1)
                        continue
                    
                    sec_dfs.append(df)
                    processed_files += 1
                
            except Exception as e:
                print(f"エラー - ファイル {file.name} の処理中: {str(e)}")
                continue
    
    if not sec_dfs:
        raise ValueError("有効なデータを含むCSVファイルが見つかりませんでした。")
    
    print(f"\n処理概要:")
    print(f"- 検出ファイル数: {total_files}")
    print(f"- 正常処理ファイル数: {processed_files}")
    
    # すべてのデータフレームを結合
    sec_df = pd.concat(sec_dfs, ignore_index=True)
    print(f"- 総レコード数: {len(sec_df)}")
    
    # 必要な列の抽出
    required_columns = [
        "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
        "レース名", "種別", "芝ダ障害コード", "馬番", "騎手名", "調教師名", "グレード"
    ]
    sec_df = sec_df[required_columns]
    
    # データ型の変換
    print("\nデータ型を変換中...")
    sec_df["距離"] = pd.to_numeric(sec_df["距離"], errors="coerce")
    sec_df["着順"] = pd.to_numeric(sec_df["着順"], errors="coerce")
    sec_df["グレード"] = pd.to_numeric(sec_df["グレード"], errors="coerce")
    sec_df["種別"] = pd.to_numeric(sec_df["種別"], errors="coerce")
    sec_df["レース名"] = sec_df["レース名"].astype(str)
    sec_df["馬名"] = sec_df["馬名"].astype(str)
    
    # 欠損値の確認
    null_counts = sec_df.isnull().sum()
    if null_counts.any():
        print("\n警告: 以下の列に欠損値が存在します:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"- {col}: {count}件")
            # グレードの欠損値を種別とレース名から補完
            if col == "グレード":
                print("グレードの欠損値をレース名と種別から補完します")
                mask = sec_df["グレード"].isnull()
                if mask.any():
                    print(f"グレードの欠損値: {mask.sum()}件")
                    sec_df.loc[mask, "グレード"] = sec_df[mask].apply(determine_grade, axis=1)
    
    return sec_df

def calculate_race_level(df):
    """
    より詳細なレースレベルの計算を行います。
    賞金による重み付けを追加。
    """
    # 基本設定
    df["race_level"] = 0
    df["is_win"] = df["着順"] == 1
    df["is_placed"] = df["着順"] <= 3

    # 1. グレードによる重み付け
    for grade, values in GRADE_LEVELS.items():
        mask = df["グレード"] == grade
        df.loc[mask, "race_level"] += values["base_level"]
        # 勝利時のボーナス
        df.loc[mask & df["is_win"], "race_level"] += values["weight"]
        # 複勝時のボーナス（勝利の半分）
        df.loc[mask & df["is_placed"] & ~df["is_win"], "race_level"] += values["weight"] * 0.5

    # 2. 距離帯による重み付け
    distance_weights = {
        (0, 1400): 1.0,      # スプリント
        (1401, 1800): 1.2,   # マイル
        (1801, 2000): 1.5,   # 中距離
        (2001, 2400): 1.8,   # 中長距離
        (2401, 9999): 2.0,   # 長距離
    }
    
    for (min_dist, max_dist), weight in distance_weights.items():
        mask = (df["距離"] >= min_dist) & (df["距離"] <= max_dist)
        df.loc[mask, "race_level"] *= weight

    # 3. 2000m特別ボーナス（重要な距離帯）
    mask_2000m = (df["距離"] >= 1900) & (df["距離"] <= 2100)
    df.loc[mask_2000m, "race_level"] *= 1.2

    # 4. 連続好走のボーナス
    df["prev_placed"] = df.groupby("馬名")["is_placed"].shift(1).fillna(False)
    df["consecutive_placed"] = df.groupby("馬名")["prev_placed"].cumsum()
    df.loc[df["consecutive_placed"] >= 2, "race_level"] += 0.5

    # 5. 馬番による補正（内枠有利を考慮）
    df["post_position_factor"] = 1 - (df["馬番"] - 1) * 0.01
    df["race_level"] *= df["post_position_factor"]

    # 6. レースレベルの正規化（0-10のスケールに）
    df["race_level"] = (df["race_level"] - df["race_level"].min()) / (df["race_level"].max() - df["race_level"].min()) * 10

    return df

def analyze_win_rates(df):
    """
    より詳細な勝率分析を行います。
    グレード別の分析を追加。
    """
    # 馬ごとの基本統計
    horse_stats = df.groupby("馬名").agg({
        "race_level": ["max", "mean"],
        "is_win": "sum",
        "is_placed": "sum",
        "着順": "count",
        "グレード": lambda x: x.value_counts().index[0] if len(x) > 0 else None  # 最も多く出走したグレード
    }).reset_index()

    # カラム名の整理
    horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数", "主戦グレード"]
    
    # 勝率と複勝率の計算
    horse_stats["win_rate"] = horse_stats["勝利数"] / horse_stats["出走回数"]
    horse_stats["place_rate"] = horse_stats["複勝数"] / horse_stats["出走回数"]
    
    # グレード別の統計
    grade_stats = df.groupby("グレード").agg({
        "is_win": ["mean", "count"],
        "is_placed": "mean",
        "race_level": ["mean", "std"]
    }).reset_index()
    
    grade_stats.columns = ["グレード", "勝率", "レース数", "複勝率", "平均レベル", "レベル標準偏差"]
    
    return horse_stats, grade_stats

def visualize_results(horse_stats, grade_stats, output_dir):
    """
    結果の可視化を改善します。
    グレード別の分析を追加。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # フォント設定
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

    # 1. グレード別の勝率・複勝率
    plt.figure(figsize=(15, 8))
    x = np.arange(len(grade_stats))
    width = 0.35

    plt.bar(x - width/2, grade_stats["勝率"], width, 
            label="単勝率", color="skyblue")
    plt.bar(x + width/2, grade_stats["複勝率"], width, 
            label="複勝率", color="lightcoral")

    plt.title("グレード別 勝率・複勝率")
    plt.xlabel("グレード")
    plt.ylabel("確率")
    plt.xticks(x, [f"{GRADE_LEVELS[g]['name']}" for g in grade_stats["グレード"]])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grade_win_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. レースレベルと勝率の散布図（サイズを出走回数で変化）
    plt.figure(figsize=(15, 8))
    scatter = plt.scatter(horse_stats["最高レベル"], horse_stats["win_rate"], 
                         s=horse_stats["出走回数"]*20, alpha=0.5,
                         c=horse_stats["主戦グレード"], cmap='viridis')
    plt.title("レースレベルと勝率の関係\n（円の大きさは出走回数、色はメイングレードを表す）")
    plt.xlabel("最高レースレベル")
    plt.ylabel("勝率")
    plt.colorbar(scatter, label="主戦グレード")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "race_level_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. グレード別の平均レースレベル
    plt.figure(figsize=(15, 8))
    plt.bar(x, grade_stats["平均レベル"], yerr=grade_stats["レベル標準偏差"],
            color="lightgreen", alpha=0.7)
    plt.title("グレード別 平均レースレベル")
    plt.xlabel("グレード")
    plt.ylabel("平均レースレベル")
    plt.xticks(x, [f"{GRADE_LEVELS[g]['name']}" for g in grade_stats["グレード"]])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grade_level_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def perform_regression_analysis(df):
    """
    重回帰分析を実行します。
    """
    # 説明変数の準備
    features = ['距離', '馬番']
    # ダミー変数の作成
    df_dummy = pd.get_dummies(df, columns=['芝ダ障害コード', '種別'])
    
    # 説明変数の選択
    X = df_dummy[[col for col in df_dummy.columns if 
                  col.startswith(('距離', '馬番', '芝ダ障害コード_', '種別_'))]].copy()
    y = df_dummy['着順']

    # 欠損値の処理
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの学習
    model = LinearRegression()
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
    多変量解析を実行します。
    """
    # 相関分析用のデータ準備
    numeric_cols = ['距離', '着順', '馬番', 'race_level']
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

def perform_logistic_regression(df):
    """
    ロジスティック回帰による勝利予測を実行します。
    """
    # 説明変数の準備
    features = ['距離', '馬番']
    
    # 目的変数（1:勝利, 0:非勝利）
    y = (df['着順'] == 1).astype(int)
    
    # 説明変数の選択
    X = df[features].copy()

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデルの学習
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 予測
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 係数の取得
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)

    return coef_df, accuracy, conf_matrix, class_report, y_test, y_pred_proba

def visualize_logistic_results(coef_df, conf_matrix, y_test, y_pred_proba, output_dir):
    """
    ロジスティック回帰の結果を可視化します。
    """
    # 1. 係数のプロット
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='coefficient', y='feature')
    plt.title('ロジスティック回帰の係数')
    plt.xlabel('係数')
    plt.ylabel('特徴量')
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 混同行列のヒートマップ
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('混同行列')
    plt.xlabel('予測値')
    plt.ylabel('実際の値')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. ROC曲線
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率')
    plt.ylabel('真陽性率')
    plt.title('ROC曲線')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
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
    horse_stats, grade_stats = analyze_win_rates(df)
    
    # 重回帰分析の実行
    coef_df, r2, rmse = perform_regression_analysis(df)
    
    # 多変量解析の実行
    correlation_matrix, descriptive_stats = perform_multivariate_analysis(df)
    
    # ロジスティック回帰分析の実行
    log_coef_df, accuracy, conf_matrix, class_report, y_test, y_pred_proba = perform_logistic_regression(df)
    
    # 結果の可視化
    visualize_results(horse_stats, grade_stats, output_dir)
    visualize_multivariate_results(correlation_matrix, coef_df, r2, rmse, output_dir)
    visualize_logistic_results(log_coef_df, conf_matrix, y_test, y_pred_proba, output_dir)
    
    # 結果の保存
    horse_stats.to_csv(output_dir / f"horse_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    grade_stats.to_csv(output_dir / f"grade_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    coef_df.to_csv(output_dir / f"regression_coefficients_turf_{date_str}.csv", index=False, encoding="utf-8")
    correlation_matrix.to_csv(output_dir / f"correlation_matrix_turf_{date_str}.csv", encoding="utf-8")
    descriptive_stats.to_csv(output_dir / f"descriptive_stats_turf_{date_str}.csv", encoding="utf-8")
    log_coef_df.to_csv(output_dir / f"logistic_coefficients_turf_{date_str}.csv", index=False, encoding="utf-8")
    
    print(f"\n芝レースの分析が完了しました。結果は {output_dir} に保存されています：")
    print(f"- {output_dir}/horse_stats_turf_{date_str}.csv: 馬ごとの統計")
    print(f"- {output_dir}/grade_stats_turf_{date_str}.csv: グレードごとの統計")
    print(f"- {output_dir}/regression_coefficients_turf_{date_str}.csv: 重回帰分析の係数")
    print(f"- {output_dir}/correlation_matrix_turf_{date_str}.csv: 相関行列")
    print(f"- {output_dir}/descriptive_stats_turf_{date_str}.csv: 基本統計量")
    print(f"- {output_dir}/grade_win_rate.png: グレード別勝率")
    print(f"- {output_dir}/race_level_analysis.png: レースレベルと勝率の散布図")
    print(f"- {output_dir}/grade_level_analysis.png: グレード別平均レースレベル")
    print(f"- {output_dir}/correlation_matrix.png: 相関行列のヒートマップ")
    print(f"- {output_dir}/regression_coefficients.png: 重回帰分析の係数プロット")
    print(f"- {output_dir}/logistic_coefficients_turf_{date_str}.csv: ロジスティック回帰の係数")
    print(f"- {output_dir}/logistic_coefficients.png: ロジスティック回帰の係数プロット")
    print(f"- {output_dir}/confusion_matrix.png: 混同行列")
    print(f"- {output_dir}/roc_curve.png: ROC曲線")
    print(f"\n重回帰分析の結果：")
    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print("\nロジスティック回帰の結果：")
    print(f"Accuracy: {accuracy:.3f}")
    print("\n分類レポート:")
    print(class_report)

if __name__ == "__main__":
    main() 
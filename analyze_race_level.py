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
    1: {"name": "G1", "weight": 10.0, "base_level": 9},  # 元の設定に戻す
    2: {"name": "G2", "weight": 8.0, "base_level": 8},   # 元の設定に戻す
    3: {"name": "G3", "weight": 7.0, "base_level": 7},   # 元の設定に戻す
    4: {"name": "重賞", "weight": 6.0, "base_level": 6}, # 元の設定に戻す
    5: {"name": "特別", "weight": 5.0, "base_level": 5}, # 元の設定に戻す
    6: {"name": "L", "weight": 5.5, "base_level": 5.5}   # 元の設定に戻す
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

# レースレベル計算の重み付け定義
RACE_LEVEL_WEIGHTS = {
    "grade_weight": 0.60,      # 元の設定に戻す
    "prize_weight": 0.40,      # 元の設定に戻す
    "field_size_weight": 0.10, # 出走頭数の重み
    "competition_weight": 0.20, # 競争力の重み
}

# 季節要因の補正係数
SEASONAL_FACTORS = {
    1: 1.0,  # 1月
    2: 1.0,  # 2月
    3: 1.1,  # 3月（春のG1シーズン）
    4: 1.1,  # 4月（春のG1シーズン）
    5: 1.0,  # 5月
    6: 1.0,  # 6月
    7: 1.0,  # 7月
    8: 1.0,  # 8月
    9: 1.1,  # 9月（秋のG1シーズン）
    10: 1.2, # 10月（秋のG1シーズン）
    11: 1.1, # 11月（秋のG1シーズン）
    12: 1.0, # 12月
}

# 競馬場の難易度係数
COURSE_DIFFICULTY = {
    "札幌": 1.0,
    "函館": 1.0,
    "福島": 1.0,
    "新潟": 1.1,
    "東京": 1.2,
    "中山": 1.2,
    "中京": 1.1,
    "京都": 1.2,
    "阪神": 1.2,
    "小倉": 1.0,
}

def determine_grade_by_prize(row):
    """
    1着賞金からグレードを判定する関数
    """
    prize = row["1着賞金"] if pd.notna(row["1着賞金"]) else None
    
    # 賞金が欠損している場合はNoneを返す
    if prize is None:
        return None
        
    # 賞金によるグレード判定（単位：万円）
    if prize >= 10000:  # 1億円以上
        return 1  # G1相当
    elif prize >= 7000:  # 7000万円以上
        return 2  # G2相当
    elif prize >= 4500:  # 4500万円以上
        return 3  # G3相当
    elif prize >= 3500:  # 3500万円以上
        return 4  # 重賞相当
    elif prize >= 2000:  # 2000万円以上
        return 6  # L相当
    else:
        return 5  # 特別・一般

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
    
    # 賞金からグレードを判定（1着賞金カラムが存在する場合）
    if "1着賞金" in row.index:
        prize_grade = determine_grade_by_prize(row)
        if prize_grade is not None:
            return prize_grade
    
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
                    
                    if len(df) > 0:
                        sec_dfs.append(df)
                        processed_files += 1
            except Exception as e:
                print(f"エラー - ファイル {input_path.name} の処理中: {str(e)}")
    elif input_path.is_dir():
        # ディレクトリ内のすべてのCSVファイルを検索
        csv_files = list(input_path.glob("**/*.csv"))  # サブディレクトリも含めて検索
        total_files = len(csv_files)
        print(f"\n検出されたCSVファイル数: {total_files}")
        
        if total_files == 0:
            raise ValueError(f"指定されたディレクトリ '{input_path}' にCSVファイルが見つかりません。")
        
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
                
                # 必要なカラムが存在するかチェック
                required_columns = ["芝ダ障害コード", "馬名", "着順"]
                if not all(col in df.columns for col in required_columns):
                    print(f"警告: {file.name} に必要なカラムが不足しています")
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
                        "レース名", "種別", "芝ダ障害コード", "馬番", "騎手名", "調教師名", "グレード", "1着賞金"
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

    # レース回数3回以上の馬のみを抽出
    race_counts = sec_df['馬名'].value_counts()
    horses_with_3_or_more = race_counts[race_counts >= 6].index
    sec_df = sec_df[sec_df['馬名'].isin(horses_with_3_or_more)]
    print(f"- レース回数3回以上の馬の総レコード数: {len(sec_df)}")
    
    # 必要な列の抽出（騎手名と調教師名を除外）
    required_columns = [
        "場コード", "年", "回", "日", "R", "馬名", "距離", "着順",
        "レース名", "種別", "芝ダ障害コード", "馬番", "グレード", "1着賞金"
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
                print("グレードの欠損値をレース名、1着賞金、種別から補完します")
                mask = sec_df["グレード"].isnull()
                if mask.any():
                    print(f"グレードの欠損値: {mask.sum()}件")
                    sec_df.loc[mask, "グレード"] = sec_df[mask].apply(determine_grade, axis=1)
                    
                    # 補完後の確認
                    remaining_nulls = sec_df["グレード"].isnull().sum()
                    if remaining_nulls > 0:
                        print(f"補完後も残っているグレードの欠損値: {remaining_nulls}件")
    
    return sec_df

def calculate_race_level(df):
    """
    より詳細なレースレベルの計算を行います。
    賞金、出走頭数などを考慮。
    """
    # 基本設定
    df["race_level"] = 0.0
    df["is_win"] = df["着順"] == 1
    df["is_placed"] = df["着順"] <= 3

    # 1. グレードによる基本レベル計算
    grade_level = calculate_grade_level(df)
    
    # 2. 賞金によるレベル計算
    prize_level = calculate_prize_level(df)
    
    # 各要素の重み付け合算
    df["race_level"] = (
        grade_level * RACE_LEVEL_WEIGHTS["grade_weight"] +
        prize_level * RACE_LEVEL_WEIGHTS["prize_weight"]
    )

    # レースレベルの正規化（0-10のスケールに）
    df["race_level"] = normalize_level(df["race_level"])
    
    # 距離帯による補正（微調整）
    distance_weights = {
        (0, 1400): 1.0,      # スプリント
        (1401, 1800): 1.1,   # マイル
        (1801, 2000): 1.25,  # 中距離（1.3から1.25に調整）
        (2001, 2400): 1.35,  # 中長距離（1.4から1.35に調整）
        (2401, 9999): 1.1,   # 長距離
    }
    
    for (min_dist, max_dist), weight in distance_weights.items():
        mask = (df["距離"] >= min_dist) & (df["距離"] <= max_dist)
        df.loc[mask, "race_level"] *= weight
    
    # 2000m特別ボーナス（微調整）
    mask_2000m = (df["距離"] >= 1900) & (df["距離"] <= 2100)
    df.loc[mask_2000m, "race_level"] *= 1.08  # 1.1から1.08に微調整
    
    # 最終的な正規化
    df["race_level"] = normalize_level(df["race_level"])
    
    return df

def calculate_distance_factor(df):
    """
    距離に基づく補正係数を計算
    """
    # 距離を正規化
    normalized_distance = (df["距離"] - df["距離"].min()) / (df["距離"].max() - df["距離"].min())
    
    # 距離帯ごとの補正係数を計算
    distance_factor = np.ones(len(df))
    
    # スプリント（1400m以下）
    sprint_mask = df["距離"] <= 1400
    distance_factor[sprint_mask] = 1.0 + 0.1 * normalized_distance[sprint_mask]
    
    # マイル（1401-1800m）
    mile_mask = (df["距離"] > 1400) & (df["距離"] <= 1800)
    distance_factor[mile_mask] = 1.1 + 0.15 * normalized_distance[mile_mask]
    
    # 中距離（1801-2000m）
    middle_mask = (df["距離"] > 1800) & (df["距離"] <= 2000)
    distance_factor[middle_mask] = 1.3 + 0.2 * normalized_distance[middle_mask]
    
    # 中長距離（2001-2400m）
    long_middle_mask = (df["距離"] > 2000) & (df["距離"] <= 2400)
    distance_factor[long_middle_mask] = 1.4 + 0.15 * normalized_distance[long_middle_mask]
    
    # 長距離（2401m以上）
    long_mask = df["距離"] > 2400
    distance_factor[long_mask] = 1.1 + 0.1 * normalized_distance[long_mask]
    
    return distance_factor

def calculate_prize_level(df):
    """
    賞金に基づくレベルを計算（微調整）
    """
    # 賞金の対数変換でスケールを調整
    prize_level = np.log1p(df["1着賞金"]) / np.log1p(df["1着賞金"].max()) * 9.9  # 10から9.9に微調整
    return normalize_level(prize_level)

def calculate_grade_level(df):
    """
    グレードに基づくレベルを計算
    """
    grade_level = df["グレード"].map(lambda x: GRADE_LEVELS[x]["base_level"] if pd.notna(x) else 5.0)
    
    # 勝利・複勝のボーナス
    for grade, values in GRADE_LEVELS.items():
        mask = df["グレード"] == grade
        # 勝利時のボーナス
        grade_level.loc[mask & df["is_win"]] += values["weight"]
        # 複勝時のボーナス
        grade_level.loc[mask & df["is_placed"] & ~df["is_win"]] += values["weight"] * 0.5
    
    return normalize_level(grade_level)

def calculate_field_size_level(df):
    """
    出走頭数に基づくレベルを計算
    """
    # レースごとの出走頭数を計算
    field_size = df.groupby(["場コード", "年", "回", "日", "R"])["馬番"].max()
    field_size_level = field_size / field_size.max() * 10
    return normalize_level(field_size_level)

def calculate_competition_level(df):
    """
    レースの競争力（出走馬の平均レベル）を計算
    """
    # 馬ごとの平均レベルを計算
    horse_levels = df.groupby("馬名")["race_level"].mean()
    
    # レースごとの出走馬の平均レベルを計算
    race_competition = df.groupby(["場コード", "年", "回", "日", "R"])["馬名"].apply(
        lambda x: horse_levels[x].mean()
    )
    
    return normalize_level(race_competition)

def normalize_level(series):
    """
    数値を0-10のスケールに正規化
    """
    if series.max() == series.min():
        return series.map(lambda x: 5.0)
    return (series - series.min()) / (series.max() - series.min()) * 10

def analyze_win_rates(df):
    """
    より詳細な勝率分析を行います。
    グレード別の分析を追加。
    """
    # 馬ごとの基本統計（騎手と調教師の統計を除外）
    horse_stats = df.groupby("馬名").agg({
        "race_level": ["max", "mean"],
        "is_win": "sum",
        "is_placed": "sum",
        "着順": "count",
        "グレード": lambda x: x.value_counts().index[0] if len(x) > 0 else None
    }).reset_index()

    # カラム名の整理
    horse_stats.columns = ["馬名", "最高レベル", "平均レベル", "勝利数", "複勝数", "出走回数", "主戦グレード"]
    
    # レース回数が3回以上の馬のみをフィルタリング
    horse_stats = horse_stats[horse_stats["出走回数"] >= 6]
    
    # 勝率と複勝率の計算
    horse_stats["win_rate"] = horse_stats["勝利数"] / horse_stats["出走回数"]
    horse_stats["place_rate"] = horse_stats["複勝数"] / horse_stats["出走回数"]
    
    # グレード別の統計（フィルタリング後のデータを使用）
    grade_stats = df[df["馬名"].isin(horse_stats["馬名"])].groupby("グレード").agg({
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

    # 4. 外れ値の確認（レースレベル別）
    plt.figure(figsize=(15, 10))
    
    # サブプロット1: 箱ひげ図（外れ値あり）
    plt.subplot(2, 1, 1)
    horse_stats['レースレベル区分'] = np.round(horse_stats['最高レベル']).astype(int)
    sns.boxplot(
        data=horse_stats,
        x='レースレベル区分',
        y='win_rate',
        color='lightblue',
        showfliers=True  # 外れ値を表示
    )
    plt.title("レースレベル別の勝率分布（外れ値を含む）")
    plt.xlabel("最高レースレベル")
    plt.ylabel("勝率")
    plt.grid(True, alpha=0.3)

    # サブプロット2: バイオリンプロット
    plt.subplot(2, 1, 2)
    sns.violinplot(
        data=horse_stats,
        x='レースレベル区分',
        y='win_rate',
        color='lightgreen'
    )
    plt.title("レースレベル別の勝率分布（確率密度）")
    plt.xlabel("最高レースレベル")
    plt.ylabel("勝率")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "race_level_outliers.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 外れ値の確認（グレード別）
    plt.figure(figsize=(15, 10))
    
    # サブプロット1: 箱ひげ図（外れ値あり）
    plt.subplot(2, 1, 1)
    sns.boxplot(
        data=horse_stats,
        x='主戦グレード',
        y='win_rate',
        color='lightblue',
        showfliers=True  # 外れ値を表示
    )
    plt.title("グレード別の勝率分布（外れ値を含む）")
    plt.xlabel("主戦グレード")
    plt.ylabel("勝率")
    plt.xticks(range(len(GRADE_LEVELS)), [GRADE_LEVELS[g]['name'] for g in sorted(GRADE_LEVELS.keys())])
    plt.grid(True, alpha=0.3)

    # サブプロット2: バイオリンプロット
    plt.subplot(2, 1, 2)
    sns.violinplot(
        data=horse_stats,
        x='主戦グレード',
        y='win_rate',
        color='lightgreen'
    )
    plt.title("グレード別の勝率分布（確率密度）")
    plt.xlabel("主戦グレード")
    plt.ylabel("勝率")
    plt.xticks(range(len(GRADE_LEVELS)), [GRADE_LEVELS[g]['name'] for g in sorted(GRADE_LEVELS.keys())])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "grade_outliers.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 外れ値の詳細情報
    # 各レースレベル区分での外れ値を特定
    outliers_info = []
    for level in horse_stats['レースレベル区分'].unique():
        level_data = horse_stats[horse_stats['レースレベル区分'] == level]['win_rate']
        Q1 = level_data.quantile(0.25)
        Q3 = level_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = horse_stats[
            (horse_stats['レースレベル区分'] == level) & 
            ((horse_stats['win_rate'] < (Q1 - 1.5 * IQR)) | 
             (horse_stats['win_rate'] > (Q3 + 1.5 * IQR)))
        ]
        if not outliers.empty:
            outliers_info.append({
                'レースレベル': level,
                '外れ値の数': len(outliers),
                '最小外れ値': outliers['win_rate'].min(),
                '最大外れ値': outliers['win_rate'].max(),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            })
    
    # 外れ値情報をDataFrameに変換してCSVに保存
    if outliers_info:
        outliers_df = pd.DataFrame(outliers_info)
        outliers_df.to_csv(output_dir / "outliers_analysis.csv", index=False, encoding="utf-8")

def perform_correlation_analysis(df, horse_stats):
    """
    勝率とレースレベルの相関分析を行います。
    欠損値を適切に処理します。
    """
    try:
        # 欠損値の確認と出力
        print("\n相関分析前の欠損値確認:")
        print(horse_stats[['最高レベル', 'win_rate']].isnull().sum())
        
        # 欠損値を含む行を除外
        analysis_data = horse_stats.dropna(subset=['最高レベル', 'win_rate'])
        print(f"\n分析対象データ数: {len(analysis_data)}")
        
        if len(analysis_data) == 0:
            print("警告: 有効なデータがありません")
            return None, None, None
        
        # 相関係数の計算
        correlation = np.corrcoef(analysis_data['最高レベル'], analysis_data['win_rate'])[0, 1]
        
        # 単回帰分析
        X = analysis_data['最高レベル'].values.reshape(-1, 1)
        y = analysis_data['win_rate'].values
        
        # 線形回帰モデルの作成と学習
        model = LinearRegression()
        model.fit(X, y)
        
        # R2スコアの計算
        r2 = r2_score(y, model.predict(X))
        
        return correlation, model, r2
        
    except Exception as e:
        print(f"\n相関分析中にエラーが発生しました: {str(e)}")
        return None, None, None

def visualize_correlation_results(horse_stats, correlation, model, r2, output_dir):
    """
    相関分析の結果を可視化します。
    """
    plt.figure(figsize=(15, 8))
    
    # 散布図
    scatter = plt.scatter(horse_stats["最高レベル"], horse_stats["win_rate"], 
                         s=horse_stats["出走回数"]*20, alpha=0.5,
                         c=horse_stats["主戦グレード"], cmap='viridis')
    
    # 回帰直線
    X_plot = np.linspace(horse_stats["最高レベル"].min(), horse_stats["最高レベル"].max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', linestyle='--', 
             label=f'回帰直線 (R² = {r2:.3f})')
    
    plt.title(f"レースレベルと勝率の関係\n相関係数: {correlation:.3f}")
    plt.xlabel("最高レースレベル")
    plt.ylabel("勝率")
    plt.colorbar(scatter, label="主戦グレード")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "race_level_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_win_level_distribution(df, output_dir, date_str):
    """
    勝利時のレースレベルの分布を分析します。
    """
    # 勝利時のレースレベル分布
    win_races = df[df["is_win"] == True]
    
    plt.figure(figsize=(15, 8))
    sns.histplot(data=win_races, x="race_level", bins=30, kde=True)
    plt.title("勝利時のレースレベル分布")
    plt.xlabel("レースレベル")
    plt.ylabel("勝利回数")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "win_level_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 基本統計量の計算
    win_level_stats = win_races["race_level"].describe()
    print("\n勝利時のレースレベル統計:")
    print(win_level_stats)
    win_level_stats.to_csv(output_dir / f"win_level_stats_{date_str}.csv", encoding="utf-8")

def analyze_grade_win_trends(df, output_dir):
    """
    レースグレード別の勝率傾向を分析します。
    """
    # グレード別・年別の勝率推移
    grade_year_stats = df.groupby(["グレード", "年"])["is_win"].agg(["count", "mean"]).reset_index()
    grade_year_stats.columns = ["グレード", "年", "レース数", "勝率"]
    
    # グレード別の勝率トレンド
    plt.figure(figsize=(15, 8))
    for grade in sorted(df["グレード"].unique()):
        grade_data = grade_year_stats[grade_year_stats["グレード"] == grade]
        plt.plot(grade_data["年"], grade_data["勝率"], 
                marker='o', label=f"{GRADE_LEVELS[grade]['name']}")
    
    plt.title("グレード別の勝率推移")
    plt.xlabel("年")
    plt.ylabel("勝率")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grade_win_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # グレード別の勝率分布（箱ひげ図）
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x="グレード", y="is_win", 
                order=sorted(df["グレード"].unique()))
    plt.title("グレード別の勝率分布")
    plt.xlabel("グレード")
    plt.ylabel("勝率")
    plt.xticks(range(len(GRADE_LEVELS)), 
               [GRADE_LEVELS[g]['name'] for g in sorted(GRADE_LEVELS.keys())])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grade_win_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 統計情報の保存
    grade_year_stats.to_csv(output_dir / "grade_win_trends.csv", index=False, encoding="utf-8")

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
    
    # 基本統計量の表示と保存
    print("\n基本統計量:")
    stats_columns = ['最高レベル', '平均レベル', '勝利数', '複勝数', '出走回数', 'win_rate', 'place_rate']
    stats_df = horse_stats[stats_columns].describe()
    print("\n馬ごとの統計量:")
    print(stats_df)
    
    # グレード別の基本統計量
    print("\nグレード別の統計量:")
    grade_stats_columns = ['勝率', 'レース数', '複勝率', '平均レベル', 'レベル標準偏差']
    grade_stats_df = grade_stats[grade_stats_columns].describe()
    print(grade_stats_df)
    
    # 統計量をCSVに保存
    stats_df.to_csv(output_dir / f"basic_stats_turf_{date_str}.csv", encoding="utf-8")
    grade_stats_df.to_csv(output_dir / f"grade_basic_stats_turf_{date_str}.csv", encoding="utf-8")
    
    # 勝利時のレースレベル分布の分析
    analyze_win_level_distribution(df, output_dir, date_str)
    
    # グレード別の勝率傾向の分析
    analyze_grade_win_trends(df, output_dir)
    
    # 相関分析の実行
    correlation, model, r2 = perform_correlation_analysis(df, horse_stats)
    
    # 結果の可視化と保存
    if correlation is not None and model is not None and r2 is not None:
        visualize_correlation_results(horse_stats, correlation, model, r2, output_dir)
        
        # 相関分析結果の保存
        correlation_results = pd.DataFrame({
            '分析項目': ['相関係数', '決定係数', '回帰係数', '切片'],
            '値': [correlation, r2, model.coef_[0], model.intercept_]
        })
        correlation_results.to_csv(output_dir / f"correlation_analysis_turf_{date_str}.csv", 
                                index=False, encoding="utf-8")
        
        print(f"\n相関分析の結果：")
        print(f"相関係数: {correlation:.3f}")
        print(f"決定係数 (R²): {r2:.3f}")
        print(f"回帰係数: {model.coef_[0]:.3f}")
        print(f"切片: {model.intercept_:.3f}")
    else:
        print("\n警告: 相関分析を完了できませんでした。")
    
    # 結果の保存
    horse_stats.to_csv(output_dir / f"horse_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    grade_stats.to_csv(output_dir / f"grade_stats_turf_{date_str}.csv", index=False, encoding="utf-8")
    
    print(f"\n芝レースの分析が完了しました。結果は {output_dir} に保存されています：")
    print(f"- {output_dir}/horse_stats_turf_{date_str}.csv: 馬ごとの統計")
    print(f"- {output_dir}/grade_stats_turf_{date_str}.csv: グレードごとの統計")
    print(f"- {output_dir}/correlation_analysis_turf_{date_str}.csv: 相関分析結果")
    print(f"- {output_dir}/grade_win_rate.png: グレード別勝率")
    print(f"- {output_dir}/race_level_correlation.png: レースレベルと勝率の相関分析")
    print(f"- {output_dir}/grade_level_analysis.png: グレード別平均レースレベル")
    print(f"- {output_dir}/win_level_distribution.png: 勝利時のレースレベル分布")
    print(f"- {output_dir}/grade_win_trends.png: グレード別の勝率推移")
    print(f"- {output_dir}/grade_win_distribution.png: グレード別の勝率分布")
    print(f"- {output_dir}/win_level_stats_{date_str}.csv: 勝利時のレースレベル統計")
    print(f"- {output_dir}/grade_win_trends.csv: グレード別の勝率推移データ")

if __name__ == "__main__":
    main() 
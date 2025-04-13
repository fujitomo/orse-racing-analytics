import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import japanize_matplotlib

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def load_data(file_path):
    """単一のCSVファイルを読み込んで前処理を行う"""
    print(f"データを読み込んでいます: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 必要な列の確認
    required_cols = ['馬名', '着順', '1着賞金', 'レース名']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {missing_cols}")
    
    # データの前処理
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['1着賞金'] = pd.to_numeric(df['1着賞金'], errors='coerce')
    
    # 欠損値の除去
    df = df.dropna(subset=['着順', '1着賞金'])
    
    # 勝利フラグの追加
    df['is_win'] = df['着順'] == 1
    df['is_placed'] = df['着順'] <= 3
    
    return df

def load_all_data(input_path):
    """
    指定されたパスからデータを読み込む。
    パスがディレクトリの場合は、すべてのCSVファイルを読み込んで結合する。
    """
    input_path = Path(input_path)
    all_dfs = []
    processed_files = 0
    error_files = 0
    
    if input_path.is_file():
        # 単一のファイルの場合
        if input_path.suffix.lower() == '.csv':
            try:
                df = load_data(input_path)
                all_dfs.append(df)
                processed_files += 1
            except Exception as e:
                print(f"エラー - ファイル {input_path.name} の処理中: {str(e)}")
                error_files += 1
    elif input_path.is_dir():
        # ディレクトリ内のすべてのCSVファイルを処理
        csv_files = list(input_path.glob("*.csv"))
        total_files = len(csv_files)
        
        if total_files == 0:
            raise ValueError(f"指定されたディレクトリ '{input_path}' にCSVファイルが見つかりません。")
        
        print(f"\n検出されたCSVファイル数: {total_files}")
        
        for file in csv_files:
            try:
                print(f"\n処理中 ({processed_files + 1}/{total_files}): {file.name}")
                df = load_data(file)
                all_dfs.append(df)
                processed_files += 1
            except Exception as e:
                print(f"エラー - ファイル {file.name} の処理中: {str(e)}")
                error_files += 1
    else:
        raise ValueError(f"指定されたパス '{input_path}' が見つかりません。")
    
    if not all_dfs:
        raise ValueError("処理可能なデータが見つかりませんでした。")
    
    # 処理結果の表示
    print(f"\n処理結果:")
    print(f"- 処理完了: {processed_files} ファイル")
    print(f"- エラー: {error_files} ファイル")
    
    # すべてのデータフレームを結合
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"- 総レコード数: {len(combined_df)}")
    
    return combined_df

def analyze_prize_levels(df):
    """賞金レベル別の分析を行う"""
    # 賞金レベルの定義（1=1万円）
    prize_ranges = [
        (0, 500, '500万円未満'),
        (500, 1000, '500-1000万円'),
        (1000, 3000, '1000-3000万円'),
        (3000, 5000, '3000-5000万円'),
        (5000, 10000, '5000万-1億円'),
        (10000, float('inf'), '1億円以上')
    ]
    
    # 賞金レベル列の作成
    df['prize_level'] = 'その他'
    for min_prize, max_prize, label in prize_ranges:
        mask = (df['1着賞金'] >= min_prize) & (df['1着賞金'] < max_prize)
        df.loc[mask, 'prize_level'] = label
    
    # レベル別の統計
    stats = df.groupby('prize_level').agg({
        'is_win': ['count', 'mean'],
        'is_placed': 'mean',
        '着順': ['mean', 'std']
    }).round(4)
    
    stats.columns = ['出走数', '勝率', '複勝率', '平均着順', '着順標準偏差']
    stats = stats.reset_index()
    
    # 賞金レベル順に並び替え
    level_order = [level for _, _, level in prize_ranges]
    stats['prize_level'] = pd.Categorical(stats['prize_level'], categories=level_order, ordered=True)
    stats = stats.sort_values('prize_level')
    
    # 統計情報の追加
    stats['勝率'] = stats['勝率'] * 100  # パーセント表示
    stats['複勝率'] = stats['複勝率'] * 100  # パーセント表示
    
    return stats

def visualize_results(stats, output_dir):
    """分析結果を可視化する"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 賞金レベル別の勝率・複勝率
    plt.figure(figsize=(15, 8))
    x = np.arange(len(stats))
    width = 0.35
    
    plt.bar(x - width/2, stats['勝率'], width, label='単勝率 (%)', color='skyblue')
    plt.bar(x + width/2, stats['複勝率'], width, label='複勝率 (%)', color='lightcoral')
    
    plt.title('賞金レベル別 勝率・複勝率分析')
    plt.xlabel('賞金レベル（1=1万円）')
    plt.ylabel('確率 (%)')
    plt.xticks(x, stats['prize_level'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 数値を表示
    for i in range(len(stats)):
        plt.text(i - width/2, stats['勝率'].iloc[i], f'{stats["勝率"].iloc[i]:.1f}%', 
                ha='center', va='bottom')
        plt.text(i + width/2, stats['複勝率'].iloc[i], f'{stats["複勝率"].iloc[i]:.1f}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prize_win_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 賞金レベル別の平均着順
    plt.figure(figsize=(15, 8))
    plt.errorbar(x, stats['平均着順'], yerr=stats['着順標準偏差'],
                fmt='o-', capsize=5, color='blue', alpha=0.6)
    
    plt.fill_between(x,
                     stats['平均着順'] - stats['着順標準偏差'],
                     stats['平均着順'] + stats['着順標準偏差'],
                     alpha=0.2)
    
    plt.title('賞金レベル別 平均着順分析')
    plt.xlabel('賞金レベル（1=1万円）')
    plt.ylabel('平均着順')
    plt.xticks(x, stats['prize_level'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 数値を表示
    for i in range(len(stats)):
        plt.text(i, stats['平均着順'].iloc[i], 
                f'{stats["平均着順"].iloc[i]:.1f}±{stats["着順標準偏差"].iloc[i]:.1f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prize_average_order.png', dpi=300, bbox_inches='tight')
    plt.close()

def prepare_features(df):
    """機械学習のための特徴量を準備"""
    features = pd.DataFrame()
    
    # 基本特徴量
    features['賞金'] = df['1着賞金']  # 1=1万円
    features['賞金_log'] = np.log1p(df['1着賞金'])
    features['賞金_sqrt'] = np.sqrt(df['1着賞金'])
    
    # レース条件
    if '芝ダ障害コード' in df.columns:
        features = pd.concat([features, pd.get_dummies(df['芝ダ障害コード'], prefix='コース')], axis=1)
    
    if '距離' in df.columns:
        features['距離'] = df['距離']
        features['距離_log'] = np.log1p(df['距離'])
    
    if '馬番' in df.columns:
        features['馬番'] = df['馬番']
        # 内・中・外枠のカテゴリ化
        features['枠_内'] = (df['馬番'] <= 4).astype(int)
        features['枠_中'] = ((df['馬番'] > 4) & (df['馬番'] <= 12)).astype(int)
        features['枠_外'] = (df['馬番'] > 12).astype(int)
    
    # 季節性（月から導出）
    if '年月日' in df.columns:
        df['月'] = pd.to_datetime(df['年月日']).dt.month
        features = pd.concat([features, pd.get_dummies(df['月'], prefix='月')], axis=1)
    
    # 馬場状態
    if '馬場状態' in df.columns:
        features = pd.concat([features, pd.get_dummies(df['馬場状態'], prefix='馬場')], axis=1)
    
    # 斤量
    if '斤量' in df.columns:
        features['斤量'] = df['斤量']
        features['斤量_標準'] = features['斤量'] - features['斤量'].mean()
    
    # 交互作用の追加
    if '距離' in features.columns and '斤量' in features.columns:
        features['距離×斤量'] = features['距離'] * features['斤量']
    
    if '賞金' in features.columns and '距離' in features.columns:
        features['賞金×距離'] = features['賞金'] * features['距離']
    
    return features

def analyze_by_various_factors(df):
    """様々な要因による分析"""
    results = []
    
    # コース種別による分析
    if '芝ダ障害コード' in df.columns:
        course_stats = df.groupby('芝ダ障害コード').agg({
            'is_win': ['count', 'mean'],
            'is_placed': 'mean',
            '着順': ['mean', 'std']
        }).round(4)
        results.append(('コース種別', course_stats))
    
    # 距離帯による分析
    if '距離' in df.columns:
        df['距離帯'] = pd.cut(df['距離'], 
                           bins=[0, 1200, 1600, 2000, 2400, float('inf')],
                           labels=['短距離', 'マイル', '中距離', '中長距離', '長距離'])
        distance_stats = df.groupby('距離帯').agg({
            'is_win': ['count', 'mean'],
            'is_placed': 'mean',
            '着順': ['mean', 'std']
        }).round(4)
        results.append(('距離帯', distance_stats))
    
    # 馬場状態による分析
    if '馬場状態' in df.columns:
        track_stats = df.groupby('馬場状態').agg({
            'is_win': ['count', 'mean'],
            'is_placed': 'mean',
            '着順': ['mean', 'std']
        }).round(4)
        results.append(('馬場状態', track_stats))
    
    return results

def visualize_factor_analysis(stats_list, output_dir):
    """各要因の分析結果を可視化"""
    for factor_name, stats in stats_list:
        # 勝率・複勝率の可視化
        plt.figure(figsize=(12, 6))
        x = np.arange(len(stats.index))
        width = 0.35
        
        win_rates = stats[('is_win', 'mean')] * 100
        place_rates = stats[('is_placed', 'mean')] * 100
        
        plt.bar(x - width/2, win_rates, width, label='単勝率 (%)', color='skyblue')
        plt.bar(x + width/2, place_rates, width, label='複勝率 (%)', color='lightcoral')
        
        plt.title(f'{factor_name}別 勝率・複勝率分析')
        plt.xlabel(factor_name)
        plt.ylabel('確率 (%)')
        plt.xticks(x, stats.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数値を表示
        for i in range(len(stats)):
            plt.text(i - width/2, win_rates.iloc[i], f'{win_rates.iloc[i]:.1f}%', 
                    ha='center', va='bottom')
            plt.text(i + width/2, place_rates.iloc[i], f'{place_rates.iloc[i]:.1f}%', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{factor_name}_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 平均着順の可視化
        plt.figure(figsize=(12, 6))
        mean_orders = stats[('着順', 'mean')]
        std_orders = stats[('着順', 'std')]
        
        plt.errorbar(x, mean_orders, yerr=std_orders,
                    fmt='o-', capsize=5, color='blue', alpha=0.6)
        
        plt.fill_between(x,
                        mean_orders - std_orders,
                        mean_orders + std_orders,
                        alpha=0.2)
        
        plt.title(f'{factor_name}別 平均着順分析')
        plt.xlabel(factor_name)
        plt.ylabel('平均着順')
        plt.xticks(x, stats.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 数値を表示
        for i in range(len(stats)):
            plt.text(i, mean_orders.iloc[i], 
                    f'{mean_orders.iloc[i]:.1f}±{std_orders.iloc[i]:.1f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{factor_name}_orders.png', dpi=300, bbox_inches='tight')
        plt.close()

def perform_advanced_regression(df, output_dir):
    """改良版回帰分析"""
    print("\n高度な回帰分析を実行中...")
    
    # 特徴量の準備
    X = prepare_features(df)
    y = df['着順']
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. 線形回帰
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_pred = linear_model.predict(X_test_scaled)
    linear_r2 = r2_score(y_test, linear_pred)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
    
    # 2. ランダムフォレスト
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    # 特徴量の重要度を可視化
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.barh(np.arange(len(feature_importance)), feature_importance['importance'])
    plt.yticks(np.arange(len(feature_importance)), feature_importance['feature'])
    plt.title('ランダムフォレスト：特徴量の重要度')
    plt.xlabel('重要度')
    plt.tight_layout()
    plt.savefig(output_dir / 'rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n回帰分析の結果:")
    print("線形回帰:")
    print(f"R² score: {linear_r2:.3f}")
    print(f"RMSE: {linear_rmse:.3f}")
    print("\nランダムフォレスト:")
    print(f"R² score: {rf_r2:.3f}")
    print(f"RMSE: {rf_rmse:.3f}")
    
    return feature_importance, (linear_r2, linear_rmse), (rf_r2, rf_rmse)

def perform_advanced_classification(df, output_dir):
    """改良版分類分析"""
    print("\n高度な分類分析を実行中...")
    
    # 特徴量の準備
    X = prepare_features(df)
    y = df['is_win']
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. ロジスティック回帰
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    log_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]
    
    # 2. ランダムフォレスト
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # ROC曲線の作成
    plt.figure(figsize=(10, 8))
    
    # ロジスティック回帰のROC
    fpr_log, tpr_log, _ = roc_curve(y_test, log_pred_proba)
    roc_auc_log = auc(fpr_log, tpr_log)
    plt.plot(fpr_log, tpr_log, color='darkorange', lw=2,
             label=f'ロジスティック回帰 (AUC = {roc_auc_log:.2f})')
    
    # ランダムフォレストのROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2,
             label=f'ランダムフォレスト (AUC = {roc_auc_rf:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率')
    plt.ylabel('真陽性率')
    plt.title('モデル別 ROC曲線比較')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nロジスティック回帰の分類レポート:")
    print(classification_report(y_test, log_pred))
    print("\nランダムフォレストの分類レポート:")
    print(classification_report(y_test, rf_pred))
    
    return (roc_auc_log, roc_auc_rf)

def analyze_high_prize_winners(df):
    """高賞金レース勝利馬の分析"""
    print("\n高賞金レース勝利馬の分析を実行中...")
    
    # 日付の処理を修正
    try:
        # 年月日の数値を確実に2桁にパディング
        df['月'] = df['月'].astype(str).str.zfill(2)
        df['日'] = df['日'].astype(str).str.zfill(2)
        df['年月日'] = pd.to_datetime(df[['年', '月', '日']].astype(str).agg(lambda x: '-'.join(x), axis=1), 
                                  format='%Y-%m-%d', errors='coerce')
        
        # 無効な日付を持つレコードを除外
        invalid_dates = df['年月日'].isna()
        if invalid_dates.any():
            print(f"警告: {invalid_dates.sum()}件の無効な日付を除外しました")
            df = df.dropna(subset=['年月日'])
        
    except Exception as e:
        print(f"日付の処理中にエラーが発生: {str(e)}")
        print("日付形式の詳細:")
        print(df[['年', '月', '日']].head())
        return None, None
    
    # 馬ごとのレース履歴を時系列で並べ替え
    df = df.sort_values(['馬名', '年月日'])
    
    # 各馬の最高賞金勝利レースを特定
    winners = df[df['着順'] == 1].copy()
    if len(winners) == 0:
        print("勝利レースが見つかりません")
        return None, None
        
    max_prize_wins = winners.loc[winners.groupby('馬名')['1着賞金'].idxmax()]
    
    # 賞金レベルの定義
    prize_levels = [
        (0, 500, '500万円未満'),
        (500, 1000, '500-1000万円'),
        (1000, 3000, '1000-3000万円'),
        (3000, 5000, '3000-5000万円'),
        (5000, 10000, '5000万-1億円'),
        (10000, float('inf'), '1億円以上')
    ]
    
    # 最高賞金勝利後の成績分析
    results = []
    for _, win_race in max_prize_wins.iterrows():
        horse_races = df[df['馬名'] == win_race['馬名']]
        
        # 勝利レース以降のレースを抽出
        future_races = horse_races[horse_races['年月日'] > win_race['年月日']]
        
        if len(future_races) > 0:
            # 勝利後の成績を集計
            future_stats = {
                '馬名': win_race['馬名'],
                '最高勝利賞金': win_race['1着賞金'],
                'その後のレース数': len(future_races),
                'その後の勝利数': len(future_races[future_races['着順'] == 1]),
                'その後の複勝数': len(future_races[future_races['着順'] <= 3]),
                'その後の平均着順': future_races['着順'].mean(),
                'その後の標準偏差': future_races['着順'].std()
            }
            results.append(future_stats)
    
    if not results:
        print("十分なデータが見つかりませんでした。")
        return None, None
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # 賞金レベル列の追加
    results_df['賞金レベル'] = 'その他'
    for min_prize, max_prize, label in prize_levels:
        mask = (results_df['最高勝利賞金'] >= min_prize) & (results_df['最高勝利賞金'] < max_prize)
        results_df.loc[mask, '賞金レベル'] = label
    
    # 賞金レベル別の集計
    level_stats = results_df.groupby('賞金レベル').agg({
        '馬名': 'count',
        'その後の勝利数': ['sum', 'mean'],
        'その後のレース数': ['sum', 'mean'],
        'その後の複勝数': ['sum', 'mean'],
        'その後の平均着順': 'mean',
        'その後の標準偏差': 'mean'
    }).round(3)
    
    # 勝率と複勝率の計算
    level_stats['後続レース勝率'] = (level_stats['その後の勝利数']['sum'] / level_stats['その後のレース数']['sum'] * 100).round(1)
    level_stats['後続レース複勝率'] = (level_stats['その後の複勝数']['sum'] / level_stats['その後のレース数']['sum'] * 100).round(1)
    
    return results_df, level_stats

def visualize_high_prize_results(results_df, level_stats, output_dir):
    """高賞金レース勝利馬の分析結果を可視化"""
    output_dir = Path(output_dir)
    
    # 1. 賞金レベル別の後続レース成績
    plt.figure(figsize=(15, 8))
    x = np.arange(len(level_stats))
    width = 0.35
    
    plt.bar(x - width/2, level_stats['後続レース勝率'], width, 
            label='勝率 (%)', color='skyblue')
    plt.bar(x + width/2, level_stats['後続レース複勝率'], width, 
            label='複勝率 (%)', color='lightcoral')
    
    plt.title('高賞金レース勝利後の成績分析')
    plt.xlabel('勝利時の賞金レベル（1=1万円）')
    plt.ylabel('確率 (%)')
    plt.xticks(x, level_stats.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 数値を表示
    for i in range(len(level_stats)):
        plt.text(i - width/2, level_stats['後続レース勝率'].iloc[i], 
                f'{level_stats["後続レース勝率"].iloc[i]}%', 
                ha='center', va='bottom')
        plt.text(i + width/2, level_stats['後続レース複勝率'].iloc[i], 
                f'{level_stats["後続レース複勝率"].iloc[i]}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_prize_winners_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 賞金レベル別の平均着順推移
    plt.figure(figsize=(15, 8))
    plt.errorbar(x, level_stats['その後の平均着順']['mean'], 
                yerr=level_stats['その後の標準偏差']['mean'],
                fmt='o-', capsize=5, color='blue', alpha=0.6)
    
    plt.fill_between(x,
                     level_stats['その後の平均着順']['mean'] - level_stats['その後の標準偏差']['mean'],
                     level_stats['その後の平均着順']['mean'] + level_stats['その後の標準偏差']['mean'],
                     alpha=0.2)
    
    plt.title('高賞金レース勝利後の平均着順推移')
    plt.xlabel('勝利時の賞金レベル（1=1万円）')
    plt.ylabel('平均着順')
    plt.xticks(x, level_stats.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 数値を表示
    for i in range(len(level_stats)):
        plt.text(i, level_stats['その後の平均着順']['mean'].iloc[i],
                f'{level_stats["その後の平均着順"]["mean"].iloc[i]:.1f}±{level_stats["その後の標準偏差"]["mean"].iloc[i]:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_prize_winners_order.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='賞金レベル別の勝率分析を行います')
    parser.add_argument('input_path', help='入力CSVファイルのパス、またはCSVファイルを含むディレクトリのパス')
    parser.add_argument('--output-dir', '-o', default='export/prize_analysis',
                      help='分析結果の出力ディレクトリ（デフォルト: export/prize_analysis）')
    args = parser.parse_args()
    
    try:
        # データの読み込み
        df = load_all_data(args.input_path)
        
        # 出力ディレクトリの準備
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 基本的な賞金レベル分析
        stats = analyze_prize_levels(df)
        print("\n賞金レベル別統計:")
        print(stats.to_string(index=False))
        
        # 2. 様々な要因による分析
        factor_stats = analyze_by_various_factors(df)
        visualize_factor_analysis(factor_stats, output_dir)
        
        # 3. 高度な回帰分析
        feature_importance, linear_metrics, rf_metrics = perform_advanced_regression(df, output_dir)
        
        # 4. 高度な分類分析
        roc_metrics = perform_advanced_classification(df, output_dir)
        
        # 高賞金レース勝利馬の分析
        results_df, level_stats = analyze_high_prize_winners(df)
        print("\n高賞金レース勝利馬の後続レース成績:")
        print(level_stats)
        
        # 結果の可視化
        visualize_results(stats, output_dir)
        visualize_high_prize_results(results_df, level_stats, output_dir)
        
        # 結果の保存
        stats.to_csv(output_dir / 'prize_level_stats.csv', index=False, encoding='utf-8')
        feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8')
        results_df.to_csv(output_dir / 'high_prize_winners_results.csv', index=False, encoding='utf-8')
        level_stats.to_csv(output_dir / 'high_prize_winners_stats.csv', encoding='utf-8')
        
        print(f"\n分析結果を保存しました: {args.output_dir}")
        print("生成されたファイル:")
        print("- prize_level_stats.csv: 賞金レベル別統計")
        print("- feature_importance.csv: 特徴量の重要度")
        print("- rf_feature_importance.png: ランダムフォレストの特徴量重要度")
        print("- model_comparison_roc.png: モデル別ROC曲線比較")
        print("- 各要因別の分析グラフ")
        print("- high_prize_winners_performance.png: 高賞金レース勝利馬の後続成績")
        print("- high_prize_winners_order.png: 高賞金レース勝利馬の着順推移")
        print("- high_prize_winners_results.csv: 高賞金レース勝利馬の詳細データ")
        print("- high_prize_winners_stats.csv: 高賞金レース勝利馬の統計データ")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 
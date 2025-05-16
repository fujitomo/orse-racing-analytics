#!/usr/bin/env python
"""
馬の脚質とトラックバイアスの相関分析スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import japanize_matplotlib
import os
import argparse
from pathlib import Path
import statsmodels.api as sm

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def load_data(input_path):
    """データの読み込み"""
    # ファイルの存在確認
    if not os.path.exists(input_path):
        raise ValueError(f"指定されたファイル '{input_path}' が存在しません。正しいパスを指定してください。")
        
    # ディレクトリ指定の場合の処理
    if os.path.isdir(input_path):
        # 最新のCSVファイルを探す
        csv_files = sorted([f for f in os.listdir(input_path) if f.endswith('_formatted.csv')], reverse=True)
        if not csv_files:
            raise ValueError(f"'{input_path}' ディレクトリ内に '_formatted.csv' で終わるCSVファイルが見つかりません。")
        
        print(f"\nディレクトリ内のCSVファイル:")
        for i, file in enumerate(csv_files[:5], 1):  # 最新5件を表示
            print(f"{i}. {file}")
        if len(csv_files) > 5:
            print(f"... 他 {len(csv_files) - 5} 件")
        
        # すべてのファイルを読み込んで結合
        df_list = []
        for file in csv_files:
            file_path = os.path.join(input_path, file)
            try:
                # 試行するエンコーディングのリスト
                encodings = ['shift_jis', 'utf-8', 'cp932', 'euc-jp']
                
                # 各エンコーディングで試行
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        df_list.append(df)
                        print(f"読み込み成功: {file} ({encoding})")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"警告: {file} の読み込みに失敗しました: {str(e)}")
                        break
            except Exception as e:
                print(f"警告: {file} の処理中にエラーが発生しました: {str(e)}")
                continue
        
        if not df_list:
            raise ValueError("有効なCSVファイルが見つかりませんでした。")
        
        # データフレームを結合
        df = pd.concat(df_list, ignore_index=True)
        print(f"\n合計 {len(df)} 行のデータを読み込みました")
        return df
    else:
        # 単一ファイルを読み込む
        encodings = ['shift_jis', 'utf-8', 'cp932', 'euc-jp']
        
        # 各エンコーディングで試行
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                print(f"ファイルを {encoding} エンコーディングで読み込みました")
                print(f"合計 {len(df)} 行のデータを読み込みました")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue
        
        raise ValueError("ファイルを読み込めませんでした。サポートされているエンコーディングで試行しましたが、すべて失敗しました。")

def get_favorable_styles_by_course():
    """競馬場・芝ダート別の有利脚質マスタを返す"""
    # 競馬場・芝ダート別の有利脚質リスト（ユーザー提供データ）
    favorable_styles = {
        ('東京', '芝'): ['差し', '追い込み'],
        ('中山', '芝'): ['逃げ', '先行'],
        ('京都', '芝'): ['逃げ', '先行'],
        ('阪神', '芝', '内'): ['先行'],
        ('阪神', '芝', '外'): ['差し'],
        ('新潟', '芝'): ['追い込み'],
        ('札幌', '芝'): ['先行', '逃げ'],
        ('函館', '芝'): ['逃げ', '先行'],
        ('福島', '芝'): ['逃げ', '先行'],
        ('小倉', '芝'): ['逃げ', '先行'],
        ('中京', '芝'): ['差し', '追い込み'],
        # ダートは便宜上先行有利としておく（未提供）
        ('東京', 'ダート'): ['逃げ', '先行'],
        ('中山', 'ダート'): ['逃げ', '先行'],
        ('京都', 'ダート'): ['逃げ', '先行'],
        ('阪神', 'ダート'): ['逃げ', '先行'],
        ('新潟', 'ダート'): ['逃げ', '先行'],
        ('札幌', 'ダート'): ['逃げ', '先行'],
        ('函館', 'ダート'): ['逃げ', '先行'],
        ('福島', 'ダート'): ['逃げ', '先行'],
        ('小倉', 'ダート'): ['逃げ', '先行'],
        ('中京', 'ダート'): ['逃げ', '先行'],
    }
    return favorable_styles

def preprocess_data(df):
    """データの前処理"""
    # レース脚質カラムのみを使用
    if 'レース脚質' not in df.columns:
        raise ValueError('レース脚質カラムが見つかりません')
    running_style_col = 'レース脚質'

    # 着順カラム名の特定
    rank_col = None
    for col in ['着順', '順位', '着']:
        if col in df.columns:
            rank_col = col
            break
    if rank_col is None:
        raise ValueError('着順カラムが見つかりません')
    
    # 場コードから競馬場名への変換
    track_map = {
        1: '札幌',
        2: '函館',
        3: '福島',
        4: '新潟',
        5: '東京',
        6: '中山',
        7: '中京',
        8: '京都',
        9: '阪神',
        10: '小倉'
    }
    
    # 競馬場カラムの作成（場コードから変換、またはそのまま利用）
    if '競馬場' not in df.columns:
        if '場コード' in df.columns:
            def code_to_track(x):
                try:
                    # 数値ならマッピング
                    return track_map.get(int(float(str(x))), '不明')
                except (ValueError, TypeError):
                    # 文字列ならそのまま（例: 既に「福島」など）
                    s = str(x).strip()
                    if s in track_map.values():
                        return s
                    return '不明'
            df['競馬場'] = df['場コード'].map(code_to_track)
            print(f"場コードから競馬場情報を生成しました")
        else:
            raise ValueError('競馬場情報（場コード）が見つかりません')
    
    # 芝・ダート判定
    if '芝ダ障害コード' in df.columns:
        df['芝ダート'] = df['芝ダ障害コード'].map(lambda x: '芝' if str(x).startswith('芝') or str(x) == '1' else 'ダート')
    elif 'トラック種別' in df.columns:
        df['芝ダート'] = df['トラック種別']
    else:
        raise ValueError('芝ダート判定に必要なカラムが見つかりません')
    
    # コース（内・外）情報があれば利用
    has_course_info = 'コース' in df.columns
    
    # 数値型または数値文字列の脚質を名称に変換
    style_map = {
        1: '逃げ',
        2: '先行',
        3: '差し',
        4: '追い込み',
        5: '好位差し',
        6: '自在'
    }
    def style_code_to_name(x):
        try:
            x_int = int(float(str(x).strip()))
            return style_map.get(x_int, f'不明({x})')
        except (ValueError, TypeError):
            return x
    
    # 脚質名称の統一
    df[running_style_col] = df[running_style_col].apply(style_code_to_name)
    
    # 有利脚質の判定（競馬場・芝ダート別）
    favorable_styles_master = get_favorable_styles_by_course()
    
    def get_favorable_styles(row):
        """レースごとの有利脚質を判定"""
        track = row['競馬場']
        surface = row['芝ダート']
        
        # 阪神の内・外の判定
        if track == '阪神' and surface == '芝' and has_course_info:
            course_type = '内' if '内' in str(row['コース']) else '外'
            key = (track, surface, course_type)
            return favorable_styles_master.get(key, [])
        
        # 通常の競馬場・芝ダート判定
        key = (track, surface)
        return favorable_styles_master.get(key, [])
    
    # 各レースで有利脚質リストを判定
    df['有利脚質リスト'] = df.apply(get_favorable_styles, axis=1)
    
    # 各レースで馬の脚質が有利脚質に含まれるか判定
    df['有利脚質該当'] = df.apply(lambda row: row[running_style_col] in row['有利脚質リスト'], axis=1)
    
    return df, running_style_col, rank_col

def analyze_running_style(df, running_style_col, rank_col, output_dir):
    """脚質分析と結果の出力"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 脚質ごとの勝率
    style_stats = df.copy()
    style_stats['勝利'] = style_stats[rank_col] == 1
    style_stats['好成績'] = style_stats[rank_col].isin([1, 2, 3])  # 1-3着を好成績とする
    
    # 勝率・連対率・3着内率を計算
    style_summary = style_stats.groupby(running_style_col).agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum')
    )
    style_summary['勝率'] = style_summary['勝利数'] / style_summary['レース数']
    style_summary['好成績率'] = style_summary['好成績数'] / style_summary['レース数']
    style_summary = style_summary.sort_values('勝率', ascending=False)
    style_summary.to_csv(f"{output_dir}/running_style_stats.csv", encoding='utf-8')
    
    # 馬別の成績集計
    horse_stats = style_stats.groupby('馬名').agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum'),
        有利脚質該当数=('有利脚質該当', 'sum')
    )
    
    # 馬ごとの勝率・好成績率・有利脚質該当率を計算
    horse_stats['勝率'] = horse_stats['勝利数'] / horse_stats['レース数']
    horse_stats['好成績率'] = horse_stats['好成績数'] / horse_stats['レース数']
    horse_stats['有利脚質該当率'] = horse_stats['有利脚質該当数'] / horse_stats['レース数']
    horse_stats = horse_stats.sort_values('勝率', ascending=False)
    
    # 出走回数による絞り込み（2回以上出走した馬のみ）
    valid_horse_stats = horse_stats[horse_stats['レース数'] >= 2].copy()
    
    if len(valid_horse_stats) < 10:
        print(f"注意: {2}回以上出走した馬が{len(valid_horse_stats)}頭しか見つかりませんでした。十分なデータがあることを確認してください。")
    
    # 馬ごとの脚質出現回数集計
    valid_horse_style = style_stats.groupby(['馬名', running_style_col]).agg(
        レース数=('勝利', 'count'),
        勝利数=('勝利', 'sum'),
        好成績数=('好成績', 'sum')
    ).reset_index()
    valid_horse_style['勝率'] = valid_horse_style['勝利数'] / valid_horse_style['レース数']
    valid_horse_style['好成績率'] = valid_horse_style['好成績数'] / valid_horse_style['レース数']
    
    # 馬ごとに最頻脚質（または好成績脚質）を得意脚質とする
    # その脚質での勝率・好成績率・レース数も持たせる
    best_styles = valid_horse_style.sort_values(['馬名', 'レース数', '好成績率', '勝率'], ascending=[True, False, False, False])
    best_styles = best_styles.groupby('馬名').first().reset_index()
    horse_best_style = best_styles[['馬名', running_style_col, '勝率', '好成績率', 'レース数']].copy()
    horse_best_style = horse_best_style.rename(columns={running_style_col: '得意脚質'})
    
    # 相関分析用のデータ作成
    horse_analysis = valid_horse_stats.copy().reset_index()
    
    print(f"相関分析対象馬: {len(horse_analysis)}頭")
    
    # 勝率と有利脚質該当率の相関を分析
    x = horse_analysis['有利脚質該当率'].values
    y = horse_analysis['勝率'].values
    
    if len(x) > 1 and len(y) > 1:
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        # スピアマンの順位相関係数（ノンパラメトリック）
        spearman_corr, p_value = stats.spearmanr(x, y)
        
        print(f"有利脚質該当率と勝率の相関係数: r = {correlation:.4f}")
        print(f"決定係数 R^2: {r_squared:.4f}")
        print(f"スピアマン順位相関係数: rho = {spearman_corr:.4f}, p値 = {p_value:.4f}")
        
        # 線形回帰
        X = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # プロット作成
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(x, y, c=horse_analysis['レース数'], cmap='Blues', 
                            s=horse_analysis['レース数'] * 2, alpha=0.6)
        plt.plot(x, predictions, color='red', linewidth=2)
        plt.xlabel('有利脚質該当率')
        plt.ylabel('勝率')
        plt.title(f'有利脚質該当率と勝率の関係\n(r={correlation:.4f}, p={p_value:.4f})')
        plt.colorbar(scatter, label='レース数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/favorable_style_win_rate.png", dpi=200)
        plt.close()
        
        # ===== 新規追加：脚質ごとの勝率分析 =====
        print("\n脚質と勝率の関係分析を実行中...")
        
        # 各脚質の勝率集計
        style_win_rates = style_summary[['勝率']].reset_index()
        
        # 脚質と勝率の散布図（水平バープロット）
        plt.figure(figsize=(10, 6))
        plt.barh(style_win_rates[running_style_col], style_win_rates['勝率'], color='skyblue')
        plt.xlabel('勝率')
        plt.ylabel('脚質')
        plt.title('脚質別の平均勝率')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/running_style_win_rate_bar.png", dpi=200)
        plt.close()
        
        # 馬ごとの脚質利用率と勝率の関係分析
        # 馬ごとの各脚質使用率を計算
        horse_style_usage = pd.pivot_table(
            valid_horse_style,
            values='レース数',
            index='馬名',
            columns=running_style_col,
            aggfunc='sum',
            fill_value=0
        )
        
        # 各馬のレース総数で割って使用率を計算
        total_races = horse_style_usage.sum(axis=1)
        for style in horse_style_usage.columns:
            horse_style_usage[f'{style}_率'] = horse_style_usage[style] / total_races
            
        # 勝率と結合
        style_analysis = pd.merge(
            horse_analysis[['馬名', '勝率', 'レース数']],
            horse_style_usage,
            on='馬名',
            how='left'
        )
        
        # 脚質使用率と勝率の相関分析（使用率カラムのみ）
        usage_cols = [col for col in style_analysis.columns if col.endswith('_率')]
        correlation_results = []
        
        plt.figure(figsize=(15, 10))
        for i, style_col in enumerate(usage_cols):
            style_name = style_col.replace('_率', '')
            
            # 相関係数とp値を計算
            x_style = style_analysis[style_col].values
            y_style = style_analysis['勝率'].values
            mask = ~(np.isnan(x_style) | np.isnan(y_style))
            
            if sum(mask) > 10:  # 十分なデータがある場合のみ
                corr, p = stats.pearsonr(x_style[mask], y_style[mask])
                spearman, sp = stats.spearmanr(x_style[mask], y_style[mask])
                
                # 回帰直線
                X_style = x_style[mask].reshape(-1, 1)
                Y_style = y_style[mask]
                if len(X_style) > 1:
                    reg = LinearRegression().fit(X_style, Y_style)
                    pred = reg.predict(X_style)
                    
                    # 散布図と回帰直線（サブプロット）
                    plt.subplot(2, 3, i+1)
                    plt.scatter(x_style[mask], y_style[mask], 
                                c=style_analysis['レース数'].values[mask], 
                                cmap='viridis', alpha=0.6, s=style_analysis['レース数'].values[mask])
                    plt.plot(X_style, pred, 'r-', linewidth=2)
                    plt.xlabel(f'{style_name}使用率')
                    plt.ylabel('勝率')
                    plt.title(f'{style_name}使用率と勝率の関係\n(r={corr:.3f}, p={p:.4f})')
                    plt.grid(True, alpha=0.3)
                    
                    # 結果を保存
                    correlation_results.append({
                        '脚質': style_name,
                        'ピアソン相関係数': corr,
                        'p値': p,
                        'スピアマン順位相関': spearman,
                        'スピアマンp値': sp,
                        '回帰係数': reg.coef_[0],
                        '切片': reg.intercept_,
                        '決定係数': reg.score(X_style, Y_style)
                    })
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/style_win_rate_correlation.png", dpi=200)
        plt.close()
        
        # 相関結果の出力
        if correlation_results:
            corr_df = pd.DataFrame(correlation_results)
            corr_df = corr_df.sort_values('ピアソン相関係数', ascending=False)
            corr_df.to_csv(f"{output_dir}/style_win_rate_correlation.csv", index=False, encoding='utf-8')
            
            print("\n脚質使用率と勝率の相関分析結果:")
            print(corr_df[['脚質', 'ピアソン相関係数', 'p値', '決定係数']].to_string(index=False))
        
        # ロジスティック回帰分析
        # 勝率を2値に変換（例：0.15以上を成功として1、それ以外を0）
        threshold = horse_analysis['勝率'].median()  # 中央値をしきい値とする
        horse_analysis['高勝率'] = (horse_analysis['勝率'] >= threshold).astype(int)
        
        # 訓練データとテストデータに分割
        X = horse_analysis['有利脚質該当率'].values.reshape(-1, 1)
        y = horse_analysis['高勝率'].values
        
        if len(np.unique(y)) > 1:  # クラスが2つ以上ある場合のみ実行
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # ロジスティック回帰モデルのトレーニング
            log_reg = LogisticRegression()
            log_reg.fit(X_train, y_train)
            
            # モデル評価
            train_score = log_reg.score(X_train, y_train)
            test_score = log_reg.score(X_test, y_test)
            y_pred = log_reg.predict(X_test)
            
            print(f"ロジスティック回帰 - トレーニングスコア: {train_score:.4f}")
            print(f"ロジスティック回帰 - テストスコア: {test_score:.4f}")
            
            # ROC曲線
            y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('偽陽性率')
            plt.ylabel('真陽性率')
            plt.title('有利脚質該当率による高勝率予測のROC曲線')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/logistic_regression_roc.png", dpi=200)
            plt.close()
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('予測値')
            plt.ylabel('実際の値')
            plt.title('混同行列')
            plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=200)
            plt.close()
            
            # 分類レポート
            report = classification_report(y_test, y_pred)
            print("分類レポート:")
            print(report)
            
            # statsmodelsを使った詳細な回帰分析
            X_sm = sm.add_constant(X)  # 切片を追加
            logit_model = sm.Logit(y, X_sm)
            try:
                result = logit_model.fit()
                print(result.summary())
            except Exception as e:
                print(f"statsmodelsでの解析エラー: {e}")
        else:
            print("警告: 高勝率クラスが1つしかないため、ロジスティック回帰分析をスキップします")
        
        # 複数の指標との相関プロット
        metrics = ['勝率', '好成績率']
        fig, ax = plt.subplots(1, len(metrics), figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            ax[i].scatter(horse_analysis['有利脚質該当率'], horse_analysis[metric], 
                         s=horse_analysis['レース数'], alpha=0.6)
            ax[i].set_xlabel('有利脚質該当率')
            ax[i].set_ylabel(metric)
            ax[i].set_title(f'有利脚質該当率と{metric}の関係')
            ax[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/favorable_style_multiple_metrics.png", dpi=200)
        plt.close()
        
        # 有利脚質該当・非該当の比較
        plt.figure(figsize=(10, 6))
        horse_analysis['有利脚質該当'] = (horse_analysis['有利脚質該当率'] > 0.5).astype(str)
        for metric in ['勝率', '好成績率']:
            sns.boxplot(x='有利脚質該当', y=metric, data=horse_analysis)
            plt.title(f'有利脚質該当別の{metric}分布')
            plt.savefig(f"{output_dir}/favorable_style_{metric}_boxplot.png", dpi=200)
            plt.close()
            
            # t検定
            group_true = horse_analysis[horse_analysis['有利脚質該当'] == 'True'][metric]
            group_false = horse_analysis[horse_analysis['有利脚質該当'] == 'False'][metric]
            
            if len(group_true) > 0 and len(group_false) > 0:
                t_stat, p_val = stats.ttest_ind(group_true, group_false, equal_var=False)
                print(f"{metric}の有利脚質該当/非該当グループ間のt検定: t={t_stat:.4f}, p={p_val:.4f}")
        
        return horse_analysis, correlation, p_value, model
    else:
        print("警告: 相関分析に十分なデータがありません")
        return horse_analysis, None, None, None

def main():
    parser = argparse.ArgumentParser(description='馬の脚質とトラックバイアスの相関分析を行うスクリプト')
    parser.add_argument('input_path', help='入力CSVファイルのパス')
    parser.add_argument('--output-dir', default='export/analysis/trackbias', help='出力ディレクトリのパス')
    
    args = parser.parse_args()
    
    try:
        # データの読み込み
        print(f"データを読み込んでいます: {args.input_path}")
        df = load_data(args.input_path)
        print(f"読み込み完了: {len(df)}行のデータ")
        
        # データの前処理
        print("データを前処理しています...")
        df, running_style_col, rank_col = preprocess_data(df)
        
        # 脚質分析
        print("脚質とトラックバイアスの分析を実行中...")
        horse_analysis, correlation, p_value, model = analyze_running_style(df, running_style_col, rank_col, args.output_dir)
        
        print(f"\n分析概要:")
        print(f"- 分析対象馬: {len(horse_analysis)}頭")
        print(f"- 有利脚質該当率と勝率の相関係数: {correlation:.3f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
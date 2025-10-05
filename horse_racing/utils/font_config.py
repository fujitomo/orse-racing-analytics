"""
フォント設定ユーティリティ
matplotlibの日本語フォント警告を解決するための統一設定
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import platform
import logging

logger = logging.getLogger(__name__)

def setup_japanese_fonts(suppress_warnings=True):
    """
    日本語フォントの設定を行う
    
    Args:
        suppress_warnings (bool): フォント警告を抑制するかどうか
    """
    
    if suppress_warnings:
        # フォントマネージャーの警告を抑制
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # システムにインストールされているフォントを確認
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Windows用の日本語フォント候補（優先順）
    windows_fonts = [
        'Yu Gothic',
        'Meiryo',
        'MS Gothic',
        'MS UI Gothic',
        'BIZ UDGothic',
        'BIZ UDPGothic',
        'Hiragino Sans',  # 追加
        'Noto Sans CJK JP'  # 追加
    ]
    
    # Linux/Mac用の日本語フォント候補（優先順）
    unix_fonts = [
        'Noto Sans CJK JP',
        'Takao Gothic',
        'DejaVu Sans',
        'Liberation Sans',
        'Hiragino Sans'  # 追加
    ]
    
    # システムに応じて利用可能なフォントを選択
    if platform.system() == 'Windows':
        font_candidates = windows_fonts
    else:
        font_candidates = unix_fonts
    
    # 利用可能なフォントを検索
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            logger.info(f"📝 日本語フォントを設定: {font}")
            break
    
    # フォントが見つからない場合はデフォルトを使用
    if selected_font is None:
        selected_font = 'DejaVu Sans'  # matplotlib のデフォルト
        logger.warning(f"⚠️ 日本語フォントが見つからないため、デフォルトフォント '{selected_font}' を使用")
    
    # matplotlibの設定（より強力な設定）
    plt.rcParams['font.family'] = [selected_font, 'DejaVu Sans', 'sans-serif']
    mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
    
    # 文字化け防止の追加設定
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # フォントサイズの設定
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 文字化け防止の追加設定
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # より強力な文字化け防止設定
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'none'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = 'png'
    
    # フォントキャッシュのクリア（文字化け対策）
    try:
        fm._rebuild()
    except:
        pass
    
    return selected_font

def get_safe_font_list():
    """
    システムで利用可能な安全なフォントリストを取得
    
    Returns:
        list: 利用可能なフォント名のリスト
    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # よく使われるフォントの中から利用可能なものを選択
    common_fonts = [
        'Yu Gothic', 'Meiryo', 'MS Gothic', 'MS UI Gothic',
        'Noto Sans CJK JP', 'Takao Gothic', 'DejaVu Sans',
        'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif'
    ]
    
    safe_fonts = [font for font in common_fonts if font in available_fonts]
    
    # 最低限 sans-serif は含める
    if 'sans-serif' not in safe_fonts:
        safe_fonts.append('sans-serif')
    
    return safe_fonts

def apply_plot_style(fig_size=(12, 8)):
    """
    プロット用の統一スタイルを適用
    
    Args:
        fig_size (tuple): 図のサイズ (width, height)
    """
    plt.style.use('default')  # デフォルトスタイルをリセット
    
    # 背景とグリッドの設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    
    # 図のサイズ
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # 線とマーカーの設定
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    
    return True

# モジュール読み込み時に自動設定
if __name__ != '__main__':
    setup_japanese_fonts(suppress_warnings=True)


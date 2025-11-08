# horse_racing/data/constants/features.py

"""
特徴量計算に使用する定数を定義します。
"""

# グレードレベルのマッピング
# 書籍の定義「G1:9, G2:4, G3:3, 重賞:2, 特別:1」とリステッド(1.5)を統合
GRADE_LEVEL_MAPPING = {
    1: 9.0,  # G1
    2: 4.0,  # G2
    3: 3.0,  # G3
    4: 2.0,  # 重賞
    5: 1.0,  # 特別
    6: 1.5,  # リステッド
}

# 賞金に基づいたグレードレベルの閾値 (単位: 万円)
PRIZE_MONEY_THRESHOLDS = {
    "G1": 1650,
    "G2": 855,
    "G3": 570,
    "LISTED": 300,
    "SPECIAL": 120,
}

# 賞金とグレードレベルのマッピング
PRIZE_TO_GRADE_LEVEL = {
    "G1": 9.0,
    "G2": 4.0,
    "G3": 3.0,
    "LISTED": 2.0,  # レポートの定義ではリステッドは2.0
    "SPECIAL": 1.0,
}

# 競馬場レベルのマッピング
# 書籍引用「東京、中山、阪神、京都、札幌 > 中京、函館、新潟 > 福島、小倉」準拠
VENUE_LEVELS = {
    "group1": 9.0,
    "group2": 7.0,
    "group3": 4.0,
}

# 競馬場グループ
VENUE_GROUPS = {
    "group1": ['東京', '中山', '阪神', '京都', '札幌'],
    "group2": ['中京', '函館', '新潟'],
    "group3": ['福島', '小倉'],
}

# 競馬場コードとグループのマッピング
VENUE_CODE_GROUPS = {
    "group1": ['01', '02', '05', '06', '08'],
    "group2": ['03', '04', '07'],
    "group3": ['09', '10'],
}

# 距離レベルのマッピング
DISTANCE_LEVELS = {
    "sprint": 0.85,      # <= 1400m
    "mile": 1.0,         # <= 1800m (基準)
    "intermediate": 1.35,# <= 2000m
    "long": 1.45,        # <= 2400m
    "extended": 1.25,    # > 2400m
}

# 距離の閾値 (m)
DISTANCE_THRESHOLDS = {
    "sprint": 1400,
    "mile": 1800,
    "intermediate": 2000,
    "long": 2400,
}


# REQI計算のデフォルト重み (レポート5.1.3節)
DEFAULT_REQI_WEIGHTS = {
    'grade_weight': 0.636,
    'venue_weight': 0.323,
    'distance_weight': 0.041,
}

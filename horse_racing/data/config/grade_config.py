"""
グレード推定用の設定クラス
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class GradeThresholds:
    """グレード推定用の賞金閾値設定（formattedデータ分析結果に基づく実証的基準）"""
    G1_MIN: int = 3407    # G1: 3,407万円以上（G1レース平均）
    G2_MIN: int = 2177    # G2: 2,177万円以上（G2レース平均）
    G3_MIN: int = 1438    # G3: 1,438万円以上（G3レース平均）
    LISTED_MIN: int = 903  # L（リステッド）: 903万円以上（Lレース平均）
    SPECIAL_MIN: int = 552 # 特別/OP: 552万円以上（特別レース平均）
    
    # グレード名マッピング
    GRADE_NAME_MAPPING: Dict[int, str] = field(default_factory=lambda: {
        1: 'Ｇ１',
        2: 'Ｇ２',
        3: 'Ｇ３',
        4: '重賞',
        5: '特別',
        6: 'Ｌ（リステッド）'
    })
    
    def to_thresholds_list(self) -> List[Tuple[int, int]]:
        """賞金しきい値を降順リストに変換します。

        Returns:
            List[Tuple[int, int]]: 最低賞金と対応するグレード値のタプルのリスト。
        """
        return [
            (self.G1_MIN, 1),
            (self.G2_MIN, 2),
            (self.G3_MIN, 3),
            (self.LISTED_MIN, 6),
            (self.SPECIAL_MIN, 5)
        ]

@dataclass
class RacePatterns:
    """レース名パターン定義"""
    G1_PATTERNS: List[str] = field(default_factory=lambda: [
        'ジャパンカップ', '有馬記念', '大阪杯', '東京優駿',
        '天皇賞', '宝塚記念', '皐月賞', '菊花賞',
        '安田記念', 'マイルチャンピオンシップ',
        '高松宮記念', 'スプリンターズステークス',
        '優駿牝馬', '桜花賞', 'ヴィクトリアマイル',
        'エリザベス女王杯', 'ジャパンカップダート',
        'ＮＨＫマイルカップ', 'チャンピオンズカップ',
        'フェブラリーステークス', '秋華賞', 'ＪＢＣクラシック',
        '中山グランドジャンプ', '中山大障害',
        '朝日杯フューチュリティステークス', 'ＪＢＣスプリント',
        'ダービー', 'オークス', 'マイル', 'フューチュリティ'
    ])
    G2_PATTERNS: List[str] = field(default_factory=lambda: ['札幌記念', '阪神カップ', '記念', '大賞典'])
    G3_PATTERNS: List[str] = field(default_factory=lambda: ['賞', '特別'])
    STAKES_PATTERNS: List[str] = field(default_factory=lambda: ['重賞', 'リステッド', 'L'])
    CONDITIONS_PATTERNS: List[str] = field(default_factory=lambda: ['条件', '新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス'])


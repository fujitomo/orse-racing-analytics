"""
データ列名の集中定義とユーティリティ
"""
class ColumnNames:
    """データ列名の集中定義とユーティリティ。

    既存の日本語列名を対象に、推定ロジックが参照する列名を提供する。
    """
    # 基本列
    RACE_NAME = 'レース名'
    DISTANCE = '距離'
    HORSE_COUNT = '頭数'
    POSITION = '着順'
    HORSE_NAME = '馬名'
    HORSE_AGE = '馬齢'
    IDM = 'IDM'
    GRADE = 'グレード'
    GRADE_Y = 'グレード_y'
    GRADE_NAME = 'グレード名'

    # 日付・識別
    REGISTRATION_NUMBER = '血統登録番号'
    RACE_DATE = '年月日'

    # 賞金関連
    PRIZE_1ST_WITH_BONUS = '1着賞金(1着算入賞金込み)'
    PRIZE_MAIN = '本賞金'

    def get_grade_columns(self):
        return [self.GRADE, 'grade', 'レースグレード']

    def get_prize_columns(self):
        return [
            '2着賞金', '3着賞金', '4着賞金', '5着賞金',
            '1着算入賞金', '2着算入賞金',
            self.PRIZE_1ST_WITH_BONUS, '2着賞金(2着算入賞金込み)', '平均賞金',
            self.PRIZE_MAIN
        ]


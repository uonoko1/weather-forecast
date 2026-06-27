"""天気予測ドメインの型定義。

外部ライブラリに依存しない純粋なデータ構造とルールを定義する。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WeatherCategory(Enum):
    """天気の3分類。"""
    SUNNY = "晴れ"
    CLOUDY = "曇り"
    RAINY = "雨"


@dataclass(frozen=True)
class DailyWeatherRecord:
    """1日分の気象観測データ。"""
    date: str
    avg_temperature: float      # 平均気温 (℃)
    max_temperature: float      # 最高気温 (℃)
    min_temperature: float      # 最低気温 (℃)
    precipitation: float        # 降水量 (mm)
    avg_humidity: float         # 平均湿度 (%)
    avg_pressure: float         # 平均海面気圧 (hPa)
    avg_wind_speed: float       # 平均風速 (m/s)
    sunshine_hours: float       # 日照時間 (h)
    weather_category: Optional[WeatherCategory] = None  # 天気概況から変換


# 気象庁の天気概況テキストを3分類にマッピングするルール
WEATHER_KEYWORDS: dict[WeatherCategory, list[str]] = {
    WeatherCategory.RAINY: ["雨", "雷", "みぞれ", "雪"],
    WeatherCategory.CLOUDY: ["曇"],
    WeatherCategory.SUNNY: ["晴", "快晴"],
}


def classify_weather(description: str) -> WeatherCategory:
    """天気概況テキストを3カテゴリに分類する。

    優先順位: 雨 > 曇り > 晴れ
    「晴一時雨」→ 雨、「晴時々曇」→ 曇り、「晴」→ 晴れ
    """
    for category in [WeatherCategory.RAINY, WeatherCategory.CLOUDY, WeatherCategory.SUNNY]:
        for keyword in WEATHER_KEYWORDS[category]:
            if keyword in description:
                return category
    return WeatherCategory.CLOUDY  # 判定不能なら曇りとする


# 特徴量として使うカラム名（モデル入力）
FEATURE_COLUMNS = [
    "avg_temperature",
    "max_temperature",
    "min_temperature",
    "precipitation",
    "avg_humidity",
    "avg_pressure",
    "avg_wind_speed",
    "sunshine_hours",
]
# test

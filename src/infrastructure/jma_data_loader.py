"""気象庁CSVデータの読み込み・パース。

気象庁の「過去の気象データ・ダウンロード」ページからDLしたCSVを読み込む。
CSVは独特なフォーマット（ヘッダー複数行、品質情報カラムなど）なので、
ここで吸収してクリーンな DataFrame を返す。
"""
import pandas as pd
from pathlib import Path

from src.domain.weather import (
    DailyWeatherRecord,
    WeatherCategory,
    classify_weather,
)


# 気象庁CSVのカラム名マッピング（CSVの列順に対応）
# 気象庁CSVは列名が日本語で、品質情報・均質番号の列が交互に入る
JMA_COLUMN_MAP = {
    "平均気温(℃)": "avg_temperature",
    "最高気温(℃)": "max_temperature",
    "最低気温(℃)": "min_temperature",
    "降水量の合計(mm)": "precipitation",
    "平均湿度(％)": "avg_humidity",
    "平均海面気圧(hPa)": "avg_pressure",
    "平均風速(m/s)": "avg_wind_speed",
    "日照時間(時間)": "sunshine_hours",
    "天気概況(昼：06時～18時)": "weather_description",
}


def load_jma_csv(file_path: str | Path) -> pd.DataFrame:
    """気象庁CSVを読み込み、整形された DataFrame を返す。

    気象庁CSVの特徴:
    - 最初の数行がヘッダー（地点情報など）
    - 品質情報・均質番号のカラムが交互に入る
    - 欠損値が「--」「×」「///」などで表現される

    Returns:
        整形済みの DataFrame。カラム名は英語に統一。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {path}")

    # 気象庁CSVはShift_JISエンコーディングで、ヘッダーが複数行ある
    # まず生のCSVを読んでヘッダー行を特定する
    df = pd.read_csv(
        path,
        encoding="shift_jis",
        header=[0, 1, 2],  # 3行ヘッダー
        index_col=0,        # 最初の列（年月日）をインデックスに
        na_values=["--", "×", "///", ""],
    )

    # MultiIndex のヘッダーをフラットにする
    # 気象庁CSVのヘッダーは3段構成なので、1段目を使う
    df.columns = [col[0] for col in df.columns]

    # 品質情報・均質番号のカラム（番号だけのカラム）を除去
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # カラム名を英語に変換
    rename_map = {}
    for jma_name, eng_name in JMA_COLUMN_MAP.items():
        for col in df.columns:
            if jma_name in col:
                rename_map[col] = eng_name
                break

    df = df.rename(columns=rename_map)

    # 必要なカラムだけ残す
    available_columns = [c for c in JMA_COLUMN_MAP.values() if c in df.columns]
    df = df[available_columns]

    # インデックスを日付型に変換
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # 数値カラムを float に変換
    numeric_cols = [c for c in df.columns if c != "weather_description"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def create_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """動作確認用のサンプルデータを生成する。

    気象庁CSVがなくても学習パイプラインをテストできるよう、
    それっぽい気象データをランダムに生成する。
    """
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # 季節変動を模擬（sin波で気温を変動させる）
    day_of_year = dates.dayofyear.values
    seasonal = np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 7月頃がピーク

    avg_temp = 15 + 12 * seasonal + np.random.normal(0, 2, n_days)
    max_temp = avg_temp + 5 + np.random.normal(0, 1, n_days)
    min_temp = avg_temp - 5 + np.random.normal(0, 1, n_days)
    precipitation = np.maximum(0, np.random.exponential(3, n_days) - 2)
    humidity = 60 + 15 * seasonal + np.random.normal(0, 8, n_days)
    humidity = np.clip(humidity, 20, 100)
    pressure = 1013 - 5 * seasonal + np.random.normal(0, 5, n_days)
    wind_speed = np.maximum(0.5, 3 + np.random.normal(0, 1.5, n_days))
    sunshine = np.maximum(0, 6 - 3 * (precipitation > 1).astype(float) + np.random.normal(0, 2, n_days))

    # 天気カテゴリ: 降水量と湿度から決定
    weather = []
    for p, h in zip(precipitation, humidity):
        if p > 1.0:
            weather.append(WeatherCategory.RAINY.value)
        elif h > 75:
            weather.append(WeatherCategory.CLOUDY.value)
        else:
            weather.append(WeatherCategory.SUNNY.value)

    df = pd.DataFrame({
        "avg_temperature": avg_temp,
        "max_temperature": max_temp,
        "min_temperature": min_temp,
        "precipitation": precipitation,
        "avg_humidity": humidity,
        "avg_pressure": pressure,
        "avg_wind_speed": wind_speed,
        "sunshine_hours": sunshine,
        "weather_description": weather,
    }, index=dates)
    df.index.name = "date"

    return df


def dataframe_to_records(df: pd.DataFrame) -> list[DailyWeatherRecord]:
    """DataFrame を DailyWeatherRecord のリストに変換する。"""
    records = []
    for date, row in df.iterrows():
        category = None
        if "weather_description" in row and pd.notna(row["weather_description"]):
            desc = str(row["weather_description"])
            category = classify_weather(desc)

        records.append(DailyWeatherRecord(
            date=str(date.date()) if hasattr(date, "date") else str(date),
            avg_temperature=float(row.get("avg_temperature", 0)),
            max_temperature=float(row.get("max_temperature", 0)),
            min_temperature=float(row.get("min_temperature", 0)),
            precipitation=float(row.get("precipitation", 0)),
            avg_humidity=float(row.get("avg_humidity", 0)),
            avg_pressure=float(row.get("avg_pressure", 0)),
            avg_wind_speed=float(row.get("avg_wind_speed", 0)),
            sunshine_hours=float(row.get("sunshine_hours", 0)),
            weather_category=category,
        ))
    return records

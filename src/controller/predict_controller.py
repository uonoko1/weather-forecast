"""予測の実行フロー。

保存済みモデルを読み込んで、新しいデータに対して予測を実行する。
"""
from src.domain.weather import FEATURE_COLUMNS
from src.infrastructure.model_repository import load_model
from src.usecase.predict import predict_weather, predict_temperature

import pandas as pd


def run_prediction(data: dict[str, float]) -> dict:
    """1日分の気象データから翌日の天気と気温を予測する。

    Args:
        data: 気象データの辞書。キーは FEATURE_COLUMNS の値。
            例: {"avg_temperature": 20.0, "max_temperature": 25.0, ...}

    Returns:
        {"weather": "晴れ", "max_temperature": 26.5} のような辞書
    """
    # 入力をDataFrameに変換（モデルが期待する形式）
    df = pd.DataFrame([data])

    result = {}

    # 天気分類
    try:
        classifier = load_model("weather_classifier")
        weather = predict_weather(classifier, df)
        result["weather"] = weather[0]
    except FileNotFoundError:
        result["weather"] = "モデルが見つかりません（先に学習を実行してください）"

    # 気温予測
    try:
        regressor = load_model("temp_regressor")
        temp = predict_temperature(regressor, df)
        result["max_temperature"] = round(temp[0], 1)
    except FileNotFoundError:
        result["max_temperature"] = "モデルが見つかりません（先に学習を実行してください）"

    return result

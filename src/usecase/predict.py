"""予測ロジック。

学習済みモデルを使って予測を実行する。
"""
import pandas as pd

from src.domain.weather import FEATURE_COLUMNS


def predict_weather(model: object, data: pd.DataFrame) -> list[str]:
    """天気カテゴリを予測する。

    Args:
        model: 学習済みの分類モデル
        data: 特徴量を含む DataFrame

    Returns:
        予測された天気ラベルのリスト（例: ["晴れ", "雨", "曇り", ...]）
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    X = data[feature_cols]
    return list(model.predict(X))


def predict_temperature(model: object, data: pd.DataFrame) -> list[float]:
    """翌日の最高気温を予測する。

    Args:
        model: 学習済みの回帰モデル
        data: 特徴量を含む DataFrame

    Returns:
        予測された気温のリスト
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    X = data[feature_cols]
    return list(model.predict(X))

"""前処理ロジック。

生の気象データを機械学習モデルに入力できる形に変換する。
"""
import pandas as pd
import numpy as np

from src.domain.weather import (
    FEATURE_COLUMNS,
    WeatherCategory,
    classify_weather,
)


def add_weather_labels(df: pd.DataFrame) -> pd.DataFrame:
    """天気概況テキストを3カテゴリのラベルに変換する。"""
    if "weather_description" not in df.columns:
        raise ValueError("weather_description カラムが必要です")

    df = df.copy()
    df["weather_label"] = df["weather_description"].apply(
        lambda x: classify_weather(str(x)).value if pd.notna(x) else None
    )
    return df


def create_next_day_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """翌日のデータを予測ターゲットとして追加する。

    当日の気象データから翌日の天気/気温を予測するため、
    ターゲットを1日ずらす（shift）。
    """
    df = df.copy()
    df[f"target_{target_col}"] = df[target_col].shift(-1)
    # 最終日はターゲットがないので除去
    df = df.iloc[:-1]
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値を含む行を除去する。"""
    return df.dropna()


def normalize_features(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """特徴量を0〜1の範囲に正規化する。

    Args:
        df: 入力DataFrame
        columns: 正規化するカラム名のリスト。Noneなら FEATURE_COLUMNS を使う

    Returns:
        (正規化済みDataFrame, {カラム名: (min, max)} の辞書)
        辞書は予測時に逆変換するために使う。
    """
    if columns is None:
        columns = [c for c in FEATURE_COLUMNS if c in df.columns]

    df = df.copy()
    stats: dict[str, tuple[float, float]] = {}

    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        stats[col] = (col_min, col_max)

    return df, stats


def split_train_test(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """時系列データを訓練とテストに分割する。

    時系列なのでシャッフルせず、末尾をテストデータとする。
    （未来のデータで学習して過去を予測するのは不正なので）
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def prepare_classification_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """分類モデル用のデータ準備を一括で行う。

    Returns:
        (訓練データ, テストデータ, 正規化統計情報)
    """
    df = add_weather_labels(df)
    df = create_next_day_target(df, "weather_label")
    df = drop_missing(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    target_col = "target_weather_label"
    keep_cols = feature_cols + [target_col]
    df = df[keep_cols]

    df, stats = normalize_features(df, feature_cols)
    train, test = split_train_test(df)

    return train, test, stats


def prepare_regression_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """回帰モデル用（気温予測）のデータ準備を一括で行う。

    Returns:
        (訓練データ, テストデータ, 正規化統計情報)
    """
    df = create_next_day_target(df, "max_temperature")
    df = drop_missing(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    target_col = "target_max_temperature"
    keep_cols = feature_cols + [target_col]
    df = df[keep_cols]

    df, stats = normalize_features(df, feature_cols)
    train, test = split_train_test(df)

    return train, test, stats

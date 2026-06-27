"""モデル学習のユースケース。

scikit-learn のモデルを学習する。
usecase層なので、データの読み込みやモデルの保存は行わない。
DataFrame を受け取り、学習済みモデルを返す。
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.domain.weather import FEATURE_COLUMNS


def train_classifier(
    train_df: pd.DataFrame,
    target_col: str = "target_weather_label",
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """天気分類モデルを学習する。

    Args:
        train_df: 訓練データ（特徴量 + ターゲット）
        target_col: ターゲットカラム名
        n_estimators: 決定木の本数（多いほど精度↑、速度↓）
        random_state: 乱数シード（再現性のため固定）

    Returns:
        学習済みの RandomForestClassifier
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    X = train_df[feature_cols]
    y = train_df[target_col]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def train_regressor(
    train_df: pd.DataFrame,
    target_col: str = "target_max_temperature",
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestRegressor:
    """気温回帰モデルを学習する。

    Args:
        train_df: 訓練データ（特徴量 + ターゲット）
        target_col: ターゲットカラム名
        n_estimators: 決定木の本数
        random_state: 乱数シード

    Returns:
        学習済みの RandomForestRegressor
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    X = train_df[feature_cols]
    y = train_df[target_col]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X, y)
    return model

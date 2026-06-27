"""学習済みモデルの保存・読み込み。

scikit-learn のモデルを pickle 形式で永続化する。
"""
import pickle
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def save_model(model: object, name: str) -> Path:
    """学習済みモデルを保存する。

    Args:
        model: scikit-learn のモデルオブジェクト
        name: モデル名（例: "weather_classifier", "temp_regressor"）

    Returns:
        保存先のパス
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> object:
    """保存済みモデルを読み込む。

    Args:
        name: モデル名

    Returns:
        scikit-learn のモデルオブジェクト
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

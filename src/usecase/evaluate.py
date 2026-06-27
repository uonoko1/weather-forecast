"""評価ロジック。

予測結果と実測値を比較して、モデルの性能を評価する。
"""
from dataclasses import dataclass

from src.domain.metrics import accuracy, mean_absolute_error, confusion_matrix_dict
from src.domain.weather import WeatherCategory


@dataclass
class ClassificationResult:
    """分類モデルの評価結果。"""
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    labels: list[str]

    def summary(self) -> str:
        lines = [
            f"=== 天気分類モデルの評価 ===",
            f"正解率 (Accuracy): {self.accuracy:.1%}",
            f"",
            f"混同行列:",
            f"{'':>8s}" + "".join(f"{l:>8s}" for l in self.labels) + "  ← 予測",
        ]
        for actual in self.labels:
            row = f"{actual:>8s}"
            for pred in self.labels:
                row += f"{self.confusion_matrix[actual][pred]:>8d}"
            lines.append(row)
        lines.append(f"↑ 実際")
        return "\n".join(lines)


@dataclass
class RegressionResult:
    """回帰モデルの評価結果。"""
    mae: float
    y_true: list[float]
    y_pred: list[float]

    def summary(self) -> str:
        return (
            f"=== 気温予測モデルの評価 ===\n"
            f"平均絶対誤差 (MAE): {self.mae:.2f}℃\n"
            f"（予測は平均 {self.mae:.2f}℃ ズレる）"
        )


def evaluate_classification(y_true: list[str], y_pred: list[str]) -> ClassificationResult:
    """分類モデルを評価する。"""
    labels = [c.value for c in WeatherCategory]
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix_dict(y_true, y_pred, labels)
    return ClassificationResult(accuracy=acc, confusion_matrix=cm, labels=labels)


def evaluate_regression(y_true: list[float], y_pred: list[float]) -> RegressionResult:
    """回帰モデルを評価する。"""
    mae = mean_absolute_error(y_true, y_pred)
    return RegressionResult(mae=mae, y_true=y_true, y_pred=y_pred)

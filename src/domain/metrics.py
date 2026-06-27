"""評価指標の計算ロジック。

scikit-learnに依存せず、純粋なPythonで評価指標を計算する。
"""


def accuracy(y_true: list, y_pred: list) -> float:
    """分類の正解率を計算する。"""
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """平均絶対誤差 (MAE) を計算する。「平均何度ズレたか」。"""
    if len(y_true) == 0:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix_dict(
    y_true: list, y_pred: list, labels: list
) -> dict[str, dict[str, int]]:
    """混同行列を辞書形式で返す。

    Returns:
        {実際のラベル: {予測ラベル: 件数}} の形式
    """
    matrix: dict[str, dict[str, int]] = {}
    for label in labels:
        matrix[label] = {l: 0 for l in labels}

    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    return matrix

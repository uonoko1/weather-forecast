"""matplotlib を使った可視化。

評価結果をグラフとして出力する。
"""
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# GUIバックエンドがない環境でも動くようにする
matplotlib.use("Agg")

# 日本語フォント設定
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    labels: list[str],
    title: str = "混同行列",
) -> Path:
    """混同行列をヒートマップとして保存する。"""
    _ensure_output_dir()

    data = [[matrix[actual][pred] for pred in labels] for actual in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("予測")
    ax.set_ylabel("実際")
    ax.set_title(title)

    # セル内に数値を表示
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(data[i][j]), ha="center", va="center",
                    color="white" if data[i][j] > max(max(row) for row in data) / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()

    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_vs_actual(
    y_true: list[float],
    y_pred: list[float],
    title: str = "予測 vs 実測（気温）",
) -> Path:
    """予測値と実測値の散布図を保存する。"""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    # 理想的な予測ライン（y = x）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="理想 (y=x)")

    ax.set_xlabel("実測値 (℃)")
    ax.set_ylabel("予測値 (℃)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "prediction_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    title: str = "特徴量重要度",
) -> Path:
    """特徴量の重要度を棒グラフで保存する。"""
    _ensure_output_dir()

    # 重要度順にソート
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("重要度")
    ax.set_title(title)
    ax.invert_yaxis()  # 上から重要な順に
    fig.tight_layout()

    path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_time_series(
    dates: list,
    y_true: list[float],
    y_pred: list[float],
    title: str = "気温予測の時系列",
) -> Path:
    """実測と予測の時系列比較グラフを保存する。"""
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, label="実測", alpha=0.7)
    ax.plot(dates, y_pred, label="予測", alpha=0.7)
    ax.set_xlabel("日付")
    ax.set_ylabel("気温 (℃)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "time_series.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

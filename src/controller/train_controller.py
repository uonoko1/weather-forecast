"""学習の実行フロー。

infrastructure層とusecase層を組み合わせて、
データ読み込み → 前処理 → 学習 → 評価 → 保存 を実行する。
"""
from pathlib import Path

from src.domain.weather import FEATURE_COLUMNS
from src.infrastructure.jma_data_loader import load_jma_csv, create_sample_data
from src.infrastructure.model_repository import save_model
from src.infrastructure.visualizer import (
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    plot_feature_importance,
)
from src.usecase.preprocess import prepare_classification_data, prepare_regression_data
from src.usecase.train import train_classifier, train_regressor
from src.usecase.predict import predict_weather, predict_temperature
from src.usecase.evaluate import evaluate_classification, evaluate_regression


def run_training(csv_path: str | None = None, use_sample: bool = False) -> None:
    """学習パイプライン全体を実行する。

    Args:
        csv_path: 気象庁CSVのパス。Noneかつuse_sample=Trueならサンプルデータを使う
        use_sample: サンプルデータを使うかどうか
    """
    # --- データ読み込み ---
    print("📊 データを読み込み中...")
    if csv_path:
        df = load_jma_csv(csv_path)
    elif use_sample:
        df = create_sample_data(n_days=1500)
        print(f"  サンプルデータを生成しました（{len(df)}日分）")
    else:
        raise ValueError("csv_path か use_sample=True のどちらかを指定してください")

    print(f"  データ件数: {len(df)}行")
    print(f"  期間: {df.index[0]} 〜 {df.index[-1]}")
    print()

    # --- 分類モデル（天気予測） ---
    print("=" * 50)
    print("🌤️  天気分類モデルの学習")
    print("=" * 50)

    train_cls, test_cls, stats_cls = prepare_classification_data(df)
    print(f"  訓練データ: {len(train_cls)}件 / テストデータ: {len(test_cls)}件")

    classifier = train_classifier(train_cls)
    print("  学習完了！")

    y_true_cls = list(test_cls["target_weather_label"])
    y_pred_cls = predict_weather(classifier, test_cls)
    cls_result = evaluate_classification(y_true_cls, y_pred_cls)
    print()
    print(cls_result.summary())

    # モデル保存
    path = save_model(classifier, "weather_classifier")
    print(f"\n  モデル保存: {path}")

    # 可視化
    cm_path = plot_confusion_matrix(
        cls_result.confusion_matrix, cls_result.labels
    )
    print(f"  混同行列グラフ: {cm_path}")

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_cls.columns]
    fi_path = plot_feature_importance(
        feature_cols, list(classifier.feature_importances_),
        title="天気予測 - 特徴量重要度",
    )
    print(f"  特徴量重要度グラフ: {fi_path}")

    # --- 回帰モデル（気温予測） ---
    print()
    print("=" * 50)
    print("🌡️  気温予測モデルの学習")
    print("=" * 50)

    train_reg, test_reg, stats_reg = prepare_regression_data(df)
    print(f"  訓練データ: {len(train_reg)}件 / テストデータ: {len(test_reg)}件")

    regressor = train_regressor(train_reg)
    print("  学習完了！")

    y_true_reg = list(test_reg["target_max_temperature"])
    y_pred_reg = predict_temperature(regressor, test_reg)
    reg_result = evaluate_regression(y_true_reg, y_pred_reg)
    print()
    print(reg_result.summary())

    # モデル保存
    path = save_model(regressor, "temp_regressor")
    print(f"\n  モデル保存: {path}")

    # 可視化
    scatter_path = plot_prediction_vs_actual(y_true_reg, y_pred_reg)
    print(f"  予測vs実測グラフ: {scatter_path}")

    fi_path2 = plot_feature_importance(
        feature_cols, list(regressor.feature_importances_),
        title="気温予測 - 特徴量重要度",
    )
    print(f"  特徴量重要度グラフ: {fi_path2}")

    print()
    print("✅ 学習パイプライン完了！")

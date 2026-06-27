# アーキテクチャ

## オニオンアーキテクチャ

ビジネスロジック（計算・ML）とインフラ（ファイルI/O・外部データ取得）を分離するため、オニオンアーキテクチャを採用している。

```
┌─────────────────────────────────────────────┐
│  controller（最外殻）                        │
│  ┌─────────────────────────────────────────┐ │
│  │  infrastructure                         │ │
│  │  ┌───────────────────────────────────┐   │ │
│  │  │  usecase                          │   │ │
│  │  │  ┌─────────────────────────────┐   │   │ │
│  │  │  │  domain（最内殻）            │   │   │ │
│  │  │  └─────────────────────────────┘   │   │ │
│  │  └───────────────────────────────────┘   │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

依存の方向: `controller → usecase → domain ← infrastructure`

内側の層は外側の層を知らない。これにより、例えばデータソースをCSVからAPIに変えても、usecase/domain層は変更不要。

## 各層の責務

### domain（最内殻）

外部ライブラリに一切依存しない純粋なPythonコード。

| ファイル | 責務 |
|---------|------|
| `weather.py` | `WeatherCategory` enum（晴れ/曇り/雨）、`DailyWeatherRecord` データクラス、天気概況テキストの分類ルール、特徴量カラム名の定義 |
| `metrics.py` | 評価指標（accuracy、MAE、混同行列）の計算ロジック |

### usecase

機械学習のビジネスロジック。domain層にのみ依存する。

| ファイル | 責務 |
|---------|------|
| `preprocess.py` | 天気ラベル付与、翌日ターゲット作成（shift）、正規化、時系列分割 |
| `train.py` | RandomForestClassifier / RandomForestRegressor の学習 |
| `predict.py` | 学習済みモデルによる予測実行 |
| `evaluate.py` | 予測結果と実測の比較、`ClassificationResult` / `RegressionResult` の生成 |

### infrastructure

外部システム（ファイル、ライブラリ）とのやり取り。domain層に依存する。

| ファイル | 責務 |
|---------|------|
| `jma_data_loader.py` | 気象庁CSVの読み込み・パース（Shift_JIS、3行ヘッダー対応）、サンプルデータ生成 |
| `model_repository.py` | 学習済みモデルのpickle保存・読み込み |
| `visualizer.py` | matplotlib による混同行列、散布図、特徴量重要度、時系列グラフの出力 |

### controller（最外殻）

usecase層とinfrastructure層を組み合わせてフローを実行する。

| ファイル | 責務 |
|---------|------|
| `train_controller.py` | データ読み込み → 前処理 → 学習 → 評価 → モデル保存 → グラフ出力 |
| `predict_controller.py` | モデル読み込み → 予測実行 → 結果返却 |

## データフロー（学習時）

```
気象庁CSV or サンプルデータ
    │
    ▼  infrastructure/jma_data_loader.py
  DataFrame（生データ）
    │
    ▼  usecase/preprocess.py
  正規化済みDataFrame + ターゲット列（翌日の天気/気温）
    │
    ├─ 80% → 訓練データ
    └─ 20% → テストデータ
                │
                ▼  usecase/train.py
          学習済みモデル（RandomForest）
                │
                ▼  usecase/predict.py → usecase/evaluate.py
          評価結果（accuracy, MAE）
                │
                ▼  infrastructure/visualizer.py
          グラフ画像（output/*.png）
```

## データフロー（予測時）

```
CLI引数（気温、湿度、気圧など）
    │
    ▼  controller/predict_controller.py
  DataFrame（1行）
    │
    ▼  infrastructure/model_repository.py
  学習済みモデル読み込み
    │
    ▼  usecase/predict.py
  予測結果 {"weather": "晴れ", "max_temperature": 26.5}
```
